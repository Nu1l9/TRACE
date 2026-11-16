"""External query hints for the frozen Analysis graph, never edits to source Evidence.

The Analysis source filter deliberately retains all noncontradictory contexts. Alias
hints therefore improve ordering rather than secretly relaxing the study scope.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

from .common import activate, digest, require, write

activate()
from analysis_core.plan_resolver import resolve_analysis_plan_structured
from analysis_engine.retrieval_artifacts import graph_artifact
from analysis_engine.retrieval import HashEmbeddingBackend
from analysis_engine.evidence_graph import load_evidence_records
from analysis_engine.semantic_graph import SemanticEvidenceGraph, SemanticRetriever
from .statistical_method_agent import ReleaseAnalysisAgent


SCHEMA = "brain_graph_patch/1.0"
FIELDS = {
    "outcome": "outcome",
    "exposure": "intervention_or_exposure",
    "comparison": "comparator",
    "population": "population_or_subgroup",
}
MAX_OPERATIONS = 16
MAX_TERMS = 20


def ledger_fingerprint(extractor_payloads, study_by_article):
    """Bind query hints to the exact analysis input and article-study mapping."""
    return digest({"extractor_payloads": extractor_payloads, "study_by_article": study_by_article})


def validate_graph_patch(patch, *, protocol_fingerprint, ledger_fingerprint, analysis_ids):
    require(isinstance(patch, dict), "graph patch must be an object")
    allowed = {"schema_version", "analysis_id", "base_protocol_fingerprint",
               "base_ledger_fingerprint", "operations", "patch_id"}
    require(not set(patch) - allowed, "graph patch contains unsupported fields; source/selection/scope mutation is forbidden")
    require(patch.get("schema_version") == SCHEMA, f"graph patch schema_version must be {SCHEMA}")
    require(patch.get("base_protocol_fingerprint") == protocol_fingerprint, "graph patch has a stale protocol fingerprint")
    require(patch.get("base_ledger_fingerprint") == ledger_fingerprint, "graph patch has a stale ledger fingerprint")
    require(isinstance(patch.get("analysis_id"), str) and patch["analysis_id"] in analysis_ids,
            "graph patch analysis_id is not in the declared plan")
    if "patch_id" in patch:
        require(isinstance(patch["patch_id"], str) and 0 < len(patch["patch_id"]) <= 160, "patch_id must be a short string")
    operations = patch.get("operations")
    require(isinstance(operations, list) and 1 <= len(operations) <= MAX_OPERATIONS,
            f"graph patch needs 1..{MAX_OPERATIONS} operations")
    result = copy.deepcopy(patch)
    for operation in result["operations"]:
        require(isinstance(operation, dict), "graph operation must be an object")
        require(set(operation) == {"operation", "field", "terms", "basis", "scope_change"},
                "only query aliases are supported; numeric/source/selection/scope fields are forbidden")
        require(operation["operation"] == "add_query_aliases", "only add_query_aliases is supported")
        require(isinstance(operation["field"], str) and operation["field"] in FIELDS,
                "graph query field is not whitelisted")
        require(operation["scope_change"] is False, "query expansion cannot change the research scope")
        require(isinstance(operation["basis"], str) and 0 < len(operation["basis"].strip()) <= 2000,
                "query aliases require a nonempty basis (at most 2000 characters)")
        terms = operation["terms"]
        require(isinstance(terms, list) and 1 <= len(terms) <= MAX_TERMS,
                f"terms must contain 1..{MAX_TERMS} strings")
        require(all(isinstance(term, str) and 0 < len(term.strip()) <= 160 for term in terms),
                "aliases must be nonempty strings, at most 160 characters each")
        operation["terms"] = list(dict.fromkeys(term.strip() for term in terms))
        operation["basis"] = operation["basis"].strip()
    return result


def validate_graph_patches(patches, **context):
    if patches is None:
        return []
    values = patches if isinstance(patches, list) else [patches]
    require(len(values) <= 32, "at most 32 graph patches per run")
    return [validate_graph_patch(value, **context) for value in values]


def _normalise(value):
    return " ".join(re.findall(r"\w+", str(value).casefold(), flags=re.UNICODE))


def _matches(text, term):
    # Phrase/token boundaries avoid matching e.g. 'risk' inside 'brisk'.
    text, term = _normalise(text), _normalise(term)
    return bool(text and term and f" {term} " in f" {text} ")


def _known(value):
    return _normalise(value) not in {"", "unknown", "not reported", "unspecified", "na", "n a"}


class QueryHintGraph(SemanticEvidenceGraph):
    """Adds only query nodes; parent source predicates and Evidence stay intact."""

    def __init__(self, records, *, plan, runtime_facets=None, patches=()):
        super().__init__(records, plan=plan, runtime_facets=runtime_facets)
        self.query_aliases = {}
        self.query_patches = copy.deepcopy(list(patches))
        for patch in patches:
            topic_id = patch["analysis_id"]
            fields = self.query_aliases.setdefault(topic_id, {})
            for operation in patch["operations"]:
                field = operation["field"]
                terms = fields.setdefault(field, [])
                for term in operation["terms"]:
                    if term not in terms:
                        terms.append(term)
                    node_id = "brain_query_alias:" + digest([topic_id, field, term])[:24]
                    self.graph.add_node(node_id, type="Brain Query Alias", name=term,
                                        field=field, soft_hint=True, basis=operation["basis"])
                    self.graph.add_edge("runtime_topic:" + topic_id, node_id,
                                        relation="HAS_SOFT_QUERY_ALIAS")

    def matches(self, topic_id, record):
        return {field: [term for term in terms if _matches(getattr(record, FIELDS[field]), term)]
                for field, terms in self.query_aliases.get(topic_id, {}).items()
                if any(_matches(getattr(record, FIELDS[field]), term) for term in terms)}


class QueryHintRetriever(SemanticRetriever):
    """Uses hints in actual review input, retaining the frozen candidate set."""

    def __init__(self, graph, embeddings, *, expanded=False, audit_callback=None, feedback_callback=None):
        super().__init__(graph, embeddings, expanded=expanded, audit_callback=audit_callback)
        self.feedback_callback = feedback_callback
        self.ranking_audit = {}

    def retrieve_all(self, plan):
        candidates, inventories = super().retrieve_all(plan)
        for definition in plan.analyses:
            topic_id = definition.analysis_id
            before = candidates[topic_id]
            if not self.graph.query_aliases.get(topic_id):
                continue
            scores = {}
            matches_by_group = {}
            for group in before.groups:
                matches = {record.evidence_id: self.graph.matches(topic_id, record)
                           for record in group.records}
                matches = {key: value for key, value in matches.items() if value}
                # Each field contributes at most once; repeated aliases cannot
                # overwhelm another group's signal by padding the patch.
                scores[group.group_id] = max((len(value) for value in matches.values()), default=0)
                matches_by_group[group.group_id] = matches
            ordered = sorted(before.groups, key=lambda group: scores[group.group_id], reverse=True)
            ranked = [group.model_copy(update={"retrieval_reasons": [
                *group.retrieval_reasons,
                f"brain_alias_rank={rank}; matched_query_fields={scores[group.group_id]}; soft_hint_only",
            ]}) for rank, group in enumerate(ordered, start=1)]
            candidates[topic_id] = before.model_copy(update={
                "groups": ranked,
                "retrieval_trace": [*before.retrieval_trace,
                    "External Brain query aliases reranked contexts; all original candidates and source predicates retained."],
            })
            self.ranking_audit[topic_id] = {
                "analysis_id": topic_id, "expanded": self.expanded,
                "before_group_order": [group.group_id for group in before.groups],
                "after_group_order": [group.group_id for group in ranked],
                "before_evidence_ids": sorted({r.evidence_id for g in before.groups for r in g.records}),
                "after_evidence_ids": sorted({r.evidence_id for g in ranked for r in g.records}),
                "matches_by_group": matches_by_group,
                "criteria_unchanged": True,
            }
        if self.feedback_callback:
            self.feedback_callback(plan, candidates, self.ranking_audit)
        return candidates, inventories

    def expand(self, definition, plan):
        # Keep the frozen full-graph retry, with hints still only affecting rank.
        retriever = QueryHintRetriever(self.graph, self.embeddings, expanded=True,
                                       feedback_callback=self.feedback_callback)
        candidates, _ = retriever.retrieve_all(plan.model_copy(update={"analyses": [definition]}))
        return candidates[definition.analysis_id]


def build_graph_feedback(graph, plan, candidate_sets, ranking_audit, *, protocol_fingerprint, ledger_fingerprint):
    """Count-only diagnostics suitable for Brain input; no effects or selections."""
    topics = []
    for definition in plan.analyses:
        topic_id = definition.analysis_id
        candidates = candidate_sets.get(topic_id)
        if candidates is None:
            # Planner can decline detailed review while a nonempty graph
            # inventory exists. Report that inventory, not a false zero.
            candidates = graph.topic_candidates(definition, plan)
        records = [record for group in candidates.groups for record in group.records]
        audit = ranking_audit.get(topic_id, {})
        source_audit = graph.topic_audit.get(topic_id, {})
        fields = {}
        for field, attribute in FIELDS.items():
            fields[field] = {
                "known_count": sum(_known(getattr(record, attribute)) for record in graph.records),
                "unknown_count": sum(not _known(getattr(record, attribute)) for record in graph.records),
                "alias_match_count": sum(field in graph.matches(topic_id, record) for record in records),
            }
        count = len(records)
        study_count = len({record.study_id for record in records})
        patches = [patch for patch in graph.query_patches if patch["analysis_id"] == topic_id]
        topics.append({
            "analysis_id": topic_id, "source_count": len(graph.records),
            "before_candidate_count": len(audit.get("before_evidence_ids", [])) if audit else count,
            "after_candidate_count": count, "added_candidate_count": 0,
            "before_study_count": study_count, "after_study_count": study_count,
            "fields": fields,
            "source_contradiction_count": len(source_audit.get("contradictions", {})),
            "ranking_changed": audit.get("before_group_order", []) != audit.get("after_group_order", []),
            "patch_fingerprint": digest(patches),
        })
    return {"schema_version": "brain_graph_feedback/1.0",
            "base_protocol_fingerprint": protocol_fingerprint,
            "base_ledger_fingerprint": ledger_fingerprint,
            "topics": topics, "stop_scope": "retrieval_only_no_effect_values"}


def preview_graph_feedback(*, protocol, task, analysis_profile, extractor_payloads,
                           study_by_article, graph_patch=None, runtime_facets=None,
                           embeddings=None, output_dir=None):
    """Run the real graph/retrieval stages without Reviewer or statistical tools.

    Missing source annotations stay unknown. This preview never estimates source
    contradiction from query hints, and is explicitly marked when unannotated.
    """
    plan = resolve_analysis_plan_structured(protocol, task, analysis_profile)
    protocol_hash = digest(protocol)
    ledger_hash = ledger_fingerprint(extractor_payloads, study_by_article)
    patches = validate_graph_patches(graph_patch, protocol_fingerprint=protocol_hash,
        ledger_fingerprint=ledger_hash, analysis_ids={topic.analysis_id for topic in plan.analyses})
    graph = QueryHintGraph(load_evidence_records(extractor_payloads, study_by_article),
                            plan=plan, runtime_facets=runtime_facets, patches=patches)
    retriever = QueryHintRetriever(graph, embeddings or HashEmbeddingBackend())
    candidates, _ = retriever.retrieve_all(plan)
    feedback = build_graph_feedback(graph, plan, candidates, retriever.ranking_audit,
        protocol_fingerprint=protocol_hash, ledger_fingerprint=ledger_hash)
    feedback["source_semantics_mode"] = "provided_annotations" if runtime_facets is not None else "unannotated_preview"
    if output_dir:
        directory = Path(output_dir)
        write(directory / "graph_feedback.json", feedback)
        write(directory / "graph_query_audit.json", {
            "schema_version": "brain_graph_query_audit/1.0", "patches": patches,
            "topics": retriever.ranking_audit, "source_facts_modified": False,
            "analysis_criteria_modified": False,
        })
        write(directory / "graph.json", graph_artifact(graph))
    return feedback


class ExtendedAnalysisAgent(ReleaseAnalysisAgent):
    """Graph query adapter around the selected release statistical interface."""

    def __init__(self, *, graph_patch=None, graph_feedback_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.graph_patch = copy.deepcopy(graph_patch)
        self.graph_feedback_dir = Path(graph_feedback_dir) if graph_feedback_dir else None
        self.validated_graph_patches = []
        self.graph_feedback = None
        self.graph_ranking_audit = {}
        self.graph_feedback_history = []

    def run(self, **kwargs):
        self.graph_feedback = None
        self.graph_ranking_audit = {}
        self.graph_feedback_history = []
        self._protocol_fingerprint = digest(kwargs["protocol"])
        self._ledger_fingerprint = ledger_fingerprint(kwargs["extractor_payloads"], kwargs["study_by_article"])
        plan = resolve_analysis_plan_structured(kwargs["protocol"], kwargs["task"], kwargs["analysis_profile"])
        self.validated_graph_patches = validate_graph_patches(
            self.graph_patch, protocol_fingerprint=self._protocol_fingerprint,
            ledger_fingerprint=self._ledger_fingerprint,
            analysis_ids={topic.analysis_id for topic in plan.analyses},
        )
        result = super().run(**kwargs)
        # Topics blocked at graph inventory still get diagnostic coverage.
        if isinstance(self.last_graph, QueryHintGraph):
            self._feedback(plan, self.last_candidate_sets or {}, self.graph_ranking_audit)
        return result

    def _build_graph(self, records, plan, runtime_facets):
        if not self.validated_graph_patches:
            return super()._build_graph(records, plan, runtime_facets)
        self.retrieval_audit = {}
        graph = QueryHintGraph(records, plan=plan, runtime_facets=runtime_facets,
                               patches=self.validated_graph_patches)
        if self.audit_dir:
            write(self.audit_dir / "graph.json", graph_artifact(graph))
        return graph

    def _make_retriever(self, graph):
        if not self.validated_graph_patches:
            return super()._make_retriever(graph)
        return QueryHintRetriever(graph, self.embeddings, audit_callback=self._save_audit,
                                  feedback_callback=self._feedback)

    def _feedback(self, plan, candidates, ranking_audit):
        self.graph_ranking_audit.update(copy.deepcopy(ranking_audit))
        self.graph_feedback = build_graph_feedback(
            self.last_graph, plan, candidates, self.graph_ranking_audit,
            protocol_fingerprint=self._protocol_fingerprint,
            ledger_fingerprint=self._ledger_fingerprint,
        )
        self.graph_feedback_history.append(copy.deepcopy(self.graph_feedback))
        if self.graph_feedback_dir:
            write(self.graph_feedback_dir / "graph_feedback.json", self.graph_feedback)
            write(self.graph_feedback_dir / "graph_query_audit.json", {
                "schema_version": "brain_graph_query_audit/1.0",
                "patches": self.validated_graph_patches,
                "topics": self.graph_ranking_audit,
                "feedback_history": self.graph_feedback_history,
                "source_facts_modified": False,
                "analysis_criteria_modified": False,
            })
