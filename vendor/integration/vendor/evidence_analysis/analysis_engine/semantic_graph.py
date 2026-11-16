"""Topic-wide source semantics and reversible graph retrieval for Semantic."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

from analysis_core.models import CandidateAnalysisDimension, CandidateSet, DimensionLevel
from analysis_core.plan_resolver import runtime_profile_for_analysis

from .evidence_graph import EvidenceGraph, _metadata_constraints, _best_match, _normalise
from .retrieval import HybridGraphRetriever
from .runtime_facets import LLMRuntimeFacetAnnotator


def semantic_dimensions(plan):
    """Compile retrieval predicates from declarations alone, without reading records.

    Each predicate has its own namespace. Subgroups may overlap; predicates do
    not partition the dataset. Repeated source questions share graph nodes.
    """
    dimensions = {}
    predicates_by_topic = {}
    for topic in plan.analyses:
        questions = [
            ("outcome", topic.outcome_terms or [topic.outcome_group]),
            ("exposure", topic.exposure_terms),
            ("comparison", topic.comparison_terms),
            ("population_or_subgroup", topic.population_terms),
        ]
        if topic.population_policy == "overall_only":
            questions.append(("population_scope", ["Overall source population, rather than an overlapping subset of that same population."]))
        if topic.dimension_field and topic.dimension_value_id:
            questions.append((topic.dimension_field, topic.dimension_terms or [topic.dimension_value_id]))
        questions.extend(_metadata_constraints(topic.selection_rules))
        # Relative policies such as longest_eligible require comparing records
        # later in review; they cannot become an absolute source-time filter.
        ids = []
        for axis, values in questions:
            terms = sorted(set(value.strip() for value in values if value.strip()))
            if not terms:
                continue
            key = "semantic_" + hashlib.sha256(json.dumps([axis, terms], ensure_ascii=False).encode()).hexdigest()[:24]
            dimensions[key] = CandidateAnalysisDimension(
                dimension_id=key, field=key, label=f"Source {axis}", kind="subgroup",
                levels=[DimensionLevel(value_id="scope", label=" | ".join(terms), terms=terms)],
            )
            ids.append(key)
        predicates_by_topic[topic.analysis_id] = list(dict.fromkeys(ids))
    return list(dimensions.values()), predicates_by_topic


class SourceSemantics(LLMRuntimeFacetAnnotator):
    mode = "llm_source_semantics_semantic"

    @staticmethod
    def _prompt(dimensions, batch):
        return (
            "Annotate source semantics for retrieval across main, secondary and subgroup questions. "
            "The dimensions below are runtime source predicates compiled from existing topic wording. "
            "Their internal IDs carry no meaning or preference. For each predicate determine source support, "
            "explicit contradiction, mixed scope, or missing information. A narrower/composite outcome, "
            "reversed comparison or incomplete description is not automatically a contradiction. "
            "Use an explicit contradiction only when supplied source wording rules out the requested scope. "
            "Leave numerical sufficiency, competing models and final inclusion to the Reviewer.\n\n"
            + LLMRuntimeFacetAnnotator._prompt(dimensions, batch)
        )

    def annotate(self, plan, records):
        dimensions, _ = semantic_dimensions(plan)
        # This temporary plan is used only by the annotator. It adds no topics
        # to the executable plan and is never written into participant input.
        return super().annotate(plan.model_copy(update={"candidate_dimensions": dimensions}), records)


class SemanticEvidenceGraph(EvidenceGraph):
    def __init__(self, records, *, plan, runtime_facets=None):
        super().__init__(records, runtime_facets=None)
        self.plan = plan
        _, self.predicates_by_topic = semantic_dimensions(plan)
        self.topic_audit = {}
        for topic in plan.analyses:
            topic_node = "runtime_topic:" + topic.analysis_id
            self.graph.add_node(topic_node, type="Analysis Topic", name=topic.title)
            for predicate in self.predicates_by_topic[topic.analysis_id]:
                node = self._add_entity(f"runtime_facet_{predicate}", "scope")
                self.graph.add_edge(topic_node, node, relation="QUERIES_SOURCE_SCOPE")

    def topic_candidates(self, definition, plan, *, expanded=False):
        """Retrieve via outcome and subgroup graph nodes, without LLM facets.

        Unknown fields stay reviewable. Declared target and data-family nodes
        recover records missed by outcome wording even when another record
        already matched; the Reviewer decides their actual eligibility.
        Explicit full-graph expansion remains available after review.
        """
        profile = runtime_profile_for_analysis(definition, plan)
        broad = super().retrieve(profile, pre_review_filter_mode="reviewer_decides")
        all_ids = {r.evidence_id for r in self.records}
        terms = definition.outcome_terms or [definition.outcome_group]
        nodes, _ = self._nodes_for_terms("outcome", terms, profile.semantic_match_threshold)
        # Shared words such as 'glenoid deviation' must not route inclination
        # to version when a declared alternative matches the node better.
        broad_union = "Graph retrieval: broad_union_outcomes" in definition.selection_rules
        competing = [a for a in plan.analyses
                     if "Graph retrieval: broad_union_outcomes" not in a.selection_rules]
        if not broad_union:
            nodes = [node for node in nodes if _best_match(self.graph.nodes[node]['name'], terms)[0]
                     >= max((_best_match(self.graph.nodes[node]['name'], a.outcome_terms or [a.outcome_group])[0]
                             for a in competing), default=0)]
        matched = self._evidence_ids_from_nodes(nodes)
        unknown = {r.evidence_id for r in self.records if not r.outcome.strip()}
        if definition.eligible_outcome_codes:
            matched |= self._evidence_ids_from_nodes(self._nodes_for_exact_values(
                "outcome_code", definition.eligible_outcome_codes))
        selected = (matched | unknown) if any(terms) else all_ids
        fallback = None
        recovered_ids = set()
        # Some broad, explicitly enumerated topics already declare every
        # eligible outcome alias.  For those opt-in analyses, typed-anchor
        # recovery would reintroduce unrelated outcomes solely because they
        # share the extraction target and data family.  Keep the recovery
        # default for all existing profiles.
        outcome_matches_only = "Graph retrieval: outcome_matches_only" in definition.selection_rules
        if not expanded and any(terms) and not outcome_matches_only:
            # A single lexical outcome hit must not prevent other records for
            # the same declared target and data family from reaching review.
            # Distinct analyses can share a broad extraction target, so keep
            # records that clearly match another topic's outcome out of this
            # topic's typed-anchor recovery.
            target_nodes = self._nodes_for_exact_values("target", [definition.target_id])
            family_nodes = self._nodes_for_exact_values(
                "data_family", definition.eligible_data_families)
            anchored = (self._evidence_ids_from_nodes(target_nodes)
                        & self._evidence_ids_from_nodes(family_nodes))
            by_id = {record.evidence_id: record for record in self.records}
            for evidence_id in anchored - selected:
                record = by_id[evidence_id]
                this_score = _best_match(record.outcome, terms)[0]
                other_score = max((
                    _best_match(record.outcome, other.outcome_terms or [other.outcome_group])[0]
                    for other in plan.analyses if other.analysis_id != definition.analysis_id
                ), default=0)
                if other_score > this_score and other_score > 0:
                    continue
                recovered_ids.add(evidence_id)
            if recovered_ids:
                fallback = ("zero_outcome_match_target_and_data_family" if not selected
                            else "partial_outcome_match_target_and_data_family")
                selected |= recovered_ids
        constraints = list(_metadata_constraints(definition.selection_rules))
        if definition.dimension_field:
            constraints.append((definition.dimension_field, definition.dimension_terms or [definition.dimension_value_id]))
        steps = []
        kinds = {'population_or_subgroup': 'subgroup', 'population': 'subgroup',
                 'exposure': 'exposure', 'intervention_or_exposure': 'exposure',
                 'comparison': 'comparator', 'comparator': 'comparator',
                 'timepoint': 'timepoint', 'follow_up': 'timepoint', 'outcome': 'outcome',
                 'data_family': 'data_family', 'estimand': 'estimand'}
        for field, values in constraints:
            kind = kinds.get(field, 'metadata_' + field)
            matches, _ = self._nodes_for_terms(kind, values, threshold=0.5)
            known_nodes = [n for n, data in self.graph.nodes(data=True)
                           if data.get('type') == kind.replace('_', ' ').title()
                           and data.get('name') != 'unspecified']
            known = self._evidence_ids_from_nodes(known_nodes)
            selected &= self._evidence_ids_from_nodes(matches) | (all_ids - known)
            steps.append({'field': field, 'retained': len(selected), 'unknown': len(all_ids - known)})
        if expanded:
            selected = all_ids
        if "Analysis eligibility: categorical_highest_vs_lowest" in definition.selection_rules:
            from .eligibility import is_categorical_high_low
            selected = {record.evidence_id for record in self.records
                        if record.evidence_id in selected and is_categorical_high_low(record)}
        compacted_ids = set()
        if not expanded and "Graph retrieval: two_per_study_outcome" in definition.selection_rules:
            # Retain the two best-described source contrasts for each literal
            # study/outcome label.  This is an opt-in pre-review projection for
            # broad topics with many alternative models, not a source deletion.
            # Every omitted ID remains in the full graph and can be recovered
            # by the explicit expansion path.
            by_key = defaultdict(list)
            for record in self.records:
                if record.evidence_id in selected:
                    by_key[(record.study_id, _normalise(record.outcome))].append(record)

            def priority(record):
                description = " ".join((record.intervention_or_exposure,
                    record.comparator, record.estimand,
                    str(record.raw_data.get("exposure_unit") or ""))).casefold()
                high = bool(re.search(r"high|top|upper|q4|q5|fourth|fifth", description))
                low = bool(re.search(r"low|bottom|lower|q1|first|least|reference", description))
                overall = not bool(re.search(r"subgroup|\bmen\b|\bwomen\b|\bmale\b|\bfemale\b",
                                              record.population_or_subgroup.casefold()))
                complete = sum(value is not None for value in
                               (record.reported_effect_value, record.ci_lower, record.ci_upper))
                adjustment = str(record.raw_data.get("adjustment_set") or "")
                return (int(high) + int(low), int(overall), complete,
                        min(len(adjustment), 1000), record.evidence_id)

            retained = {record.evidence_id for records in by_key.values()
                        for record in sorted(records, key=priority, reverse=True)[:2]}
            compacted_ids = selected - retained
            selected = retained
        groups = []
        for group in broad.groups:
            records = [r for r in group.records if r.evidence_id in selected]
            if records:
                groups.append(group.model_copy(update={'records': records,
                    'evidence_ids': [r.evidence_id for r in records],
                    'article_ids': sorted({r.article_id for r in records})}))
        self.topic_audit[definition.analysis_id] = {
            'mode': 'graph_outcome_subgroup', 'source_count': len(all_ids),
            'reviewable_count': len(selected), 'outcome_matched': len(matched),
            'unknown_outcome_count': len(unknown), 'constraints': steps,
            'expanded': expanded, 'excluded_evidence_ids': sorted(all_ids - selected),
            'runtime_facet_calls': 0, 'fallback': fallback,
            'typed_anchor_recovered_ids': sorted(recovered_ids),
            'pre_review_compacted_ids': sorted(compacted_ids),
        }
        print(f"[graph-filter] {definition.analysis_id}: {len(all_ids)} -> {len(selected)} Evidence; expanded={expanded}; fallback={fallback or 'none'}", flush=True)
        return CandidateSet(target_id=broad.target_id, groups=groups, retrieval_trace=[
            *broad.retrieval_trace,
            *(["Current typed-anchor recall traversed matching Target and DataFamily graph nodes."]
              if fallback else []),
            f"Current outcome/subgroup graph traversal retained {len(selected)}/{len(all_ids)}; expanded={expanded}."])



class SemanticRetriever(HybridGraphRetriever):
    def __init__(self, graph, embeddings, *, expanded=False, audit_callback=None):
        # No silent max_groups truncation: each graph-retained context can
        # reach review. Filtering, rather than an arbitrary top-k, localises it.
        super().__init__(graph, embeddings, max_groups=max(1, len(graph.records)),
                         pre_review_filter_mode="semantic_semantic")
        self.expanded = expanded
        self.audit_callback = audit_callback

    def inventory(self, plan):
        result = super().inventory(plan)
        if self.audit_callback:
            self.audit_callback()
        return result

    def _graph_candidates(self, definition, plan):
        return self.graph.topic_candidates(definition, plan, expanded=self.expanded)

    def expand(self, definition, plan):
        expanded = SemanticRetriever(self.graph, self.embeddings, expanded=True)
        sets, _ = expanded.retrieve_all(plan.model_copy(update={"analyses": [definition]}))
        return sets[definition.analysis_id]
