from __future__ import annotations

"""Typed GraphRAG for the graph analysis agent.

The graph is an executable retrieval index, not a visual mirror of a table.
Topic retrieval starts at protocol entity nodes and traverses reverse edges to
Evidence, then expands the local Evidence -> Article -> Study context. An
embedding model only ranks the resulting local subgraphs.
"""

from collections import Counter, defaultdict
import re
from typing import Any

import networkx as nx

from analysis_core.models import (
    CandidateGroup,
    CandidateSet,
    EvidenceInventory,
    EvidenceRecord,
    ResolvedAnalysisProfile,
)


def _text(value: object) -> str:
    return str(value or "").strip()


def _effect_value(item: dict[str, Any], field: str) -> object:
    effect = item.get("reported_effect") or item.get("effect") or {}
    return item.get(field) if item.get(field) is not None else effect.get(field)


_ANALYSIS_METADATA_FIELDS = (
    "outcome_axis",
    "measurement_semantics",
    "unit",
    "intervention_class",
    "comparator_class",
    "rsid",
    "study_design",
    "phenotype",
    "genotype_role",
)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _outcome_parts(item: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Read either the flat extractor outcome or the public nested form."""
    outcome = item.get("outcome")
    if not isinstance(outcome, dict):
        return _text(outcome), _text(item.get("outcome_code")), {}
    semantics = _mapping(outcome.get("analysis_semantics"))
    return (
        _text(outcome.get("label") or outcome.get("description") or outcome.get("code")),
        _text(outcome.get("code") or item.get("outcome_code")),
        semantics,
    )


def _normalised_raw_data(item: dict[str, Any], outcome_semantics: dict[str, Any]) -> dict[str, Any]:
    """Preserve the extractor contract while exposing source semantics to the graph.

    The public benchmark keeps a fixed top-level Evidence envelope.  Extractors
    can return the same optional semantics inside ``outcome.analysis_semantics``
    or in ``raw_data``.  This function only normalises those already supplied
    source fields; it does not infer a target, eligibility label, or Gold ID.
    """
    raw_data = _mapping(item.get("raw_data"))
    metadata = _mapping(raw_data.get("_analysis_metadata"))
    for source in (item, raw_data, outcome_semantics):
        for field in _ANALYSIS_METADATA_FIELDS:
            value = source.get(field)
            if field not in metadata and value not in (None, "", [], {}):
                metadata[field] = value
    if metadata:
        raw_data["_analysis_metadata"] = metadata
    return raw_data


def _source_location(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {"source_id": _text(value)} if _text(value) else {}


def load_evidence_records(
    extractor_payloads: list[dict[str, Any]], study_by_article: dict[str, str]
) -> list[EvidenceRecord]:
    """Convert extractor payloads into source-grounded graph evidence nodes."""
    records: list[EvidenceRecord] = []
    for payload in extractor_payloads:
        article_id = _text(payload.get("article_id") or payload.get("document_id") or "unknown_article")
        study_id = _text(payload.get("study_id") or study_by_article.get(article_id) or article_id)
        evidence_items = payload.get("evidence") or payload.get("candidate_evidence") or []
        for index, item in enumerate(evidence_items, start=1):
            if not isinstance(item, dict):
                continue
            outcome, outcome_code, outcome_semantics = _outcome_parts(item)
            evidence_id = _text(
                item.get("evidence_id") or item.get("candidate_id") or f"{article_id}:evidence:{index}"
            )
            records.append(
                EvidenceRecord(
                    evidence_id=evidence_id,
                    article_id=_text(item.get("article_id") or article_id),
                    study_id=_text(item.get("study_id") or study_id),
                    target_id=_text(item.get("target_id")),
                    intervention_or_exposure=_text(
                        item.get("intervention_or_exposure") or item.get("exposure")
                    ),
                    comparator=_text(item.get("comparator") or item.get("comparison")),
                    outcome=outcome,
                    outcome_code=outcome_code,
                    estimand=_text(item.get("estimand")),
                    adjustment_model=_text(item.get("adjustment_model") or item.get("model")),
                    population_or_subgroup=_text(item.get("population_or_subgroup")),
                    follow_up=_text(item.get("follow_up") or item.get("timepoint")),
                    data_family=_text(item.get("data_family")),
                    raw_data=_normalised_raw_data(item, outcome_semantics),
                    derivation_status=_text(item.get("derivation_status")),
                    quantitative_usability=_text(item.get("quantitative_usability")),
                    reported_effect_type=_effect_value(item, "effect_type") or item.get("reported_effect_type"),
                    reported_effect_value=_effect_value(item, "value") or item.get("reported_effect_value"),
                    ci_lower=_effect_value(item, "ci_lower"),
                    ci_upper=_effect_value(item, "ci_upper"),
                    source_location=_source_location(item.get("source_location")),
                    original=dict(item),
                )
            )
    return records


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _tokens(value: str) -> set[str]:
    return {token for token in _normalise(value).split() if len(token) > 1}


def _lexical_similarity(left: str, right: str) -> float:
    """Deterministic semantic fallback until an embedding backend is enabled."""
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    normal_left, normal_right = _normalise(left), _normalise(right)
    # A protocol alias such as "adenotonsillectomy" should fully match the
    # source phrase "adenotonsillectomy versus expectant management".
    if normal_left == normal_right or normal_left in normal_right or normal_right in normal_left:
        return 1.0
    return len(a & b) / len(a | b)


def _best_match(value: str, query_terms: list[str]) -> tuple[float, str]:
    if not query_terms:
        return 1.0, "no profile-term restriction"
    scored = [(_lexical_similarity(value, term), term) for term in query_terms]
    score, term = max(scored, default=(0.0, ""))
    return score, f"{score:.2f} lexical match to '{term}'"


def _canonical_timepoint(value: str) -> str:
    """Merge superficial wording variants of the same reported duration."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(months?|weeks?|years?)\b", value.casefold())
    return f"{match.group(1)} {match.group(2)}" if match else _normalise(value)


def _source_anchor(record: EvidenceRecord) -> str:
    """Produce a provenance node without pretending missing locations are known."""
    location = record.source_location
    parts = [
        str(location.get(key)).strip()
        for key in ("table_id", "table", "figure_id", "figure", "page", "section")
        if location.get(key) is not None and str(location.get(key)).strip()
    ]
    return f"{record.article_id}::{' | '.join(parts) if parts else 'unlocated'}"


def _cohort_anchor(record: EvidenceRecord) -> str:
    """Use an explicit cohort identifier when supplied; otherwise retain study scope."""
    explicit = (
        record.original.get("cohort_id")
        or record.raw_data.get("cohort_id")
        or record.original.get("participant_overlap_group")
        or record.raw_data.get("participant_overlap_group")
    )
    return f"{record.study_id}::{str(explicit).strip() if explicit else 'study_scope'}"


def _metadata_constraints(selection_rules: list[str]) -> list[tuple[str, list[str]]]:
    """Read explicit profile constraints without inventing candidate labels."""
    prefix = "Metadata constraint: "
    marker = " must match one of: "
    constraints: list[tuple[str, list[str]]] = []
    for rule in selection_rules:
        if not rule.startswith(prefix) or marker not in rule:
            continue
        field, values = rule[len(prefix):].split(marker, 1)
        terms = [item.strip().rstrip(".") for item in values.split(",") if item.strip()]
        if field.strip() and terms:
            constraints.append((field.strip(), terms))
    return constraints


def _record_metadata(record: EvidenceRecord) -> dict[str, Any]:
    raw = record.raw_data if isinstance(record.raw_data, dict) else {}
    metadata = raw.get("_analysis_metadata")
    merged = dict(metadata) if isinstance(metadata, dict) else {}
    for field in _ANALYSIS_METADATA_FIELDS:
        value = raw.get(field)
        if field not in merged and value not in (None, "", [], {}):
            merged[field] = value
    return merged


class EvidenceGraph:
    """Executable typed graph for protocol-constrained evidence retrieval.

    The graph retains study, article, evidence, PICO, model, source, and
    cohort-hint nodes. ``cohort_hint`` never asserts non-overlap unless the
    extractor supplied an explicit cohort/participant-overlap identifier.
    """

    def __init__(
        self,
        records: list[EvidenceRecord],
        *,
        runtime_facets: dict[str, dict[str, Any]] | None = None,
    ):
        self.records = records
        # Runtime facets are a separate, source-grounded annotation artifact.
        # Do not write them into EvidenceRecord.raw_data: the public benchmark
        # contract and its extractor-compatible output stay unchanged.
        self.runtime_facets = runtime_facets or {}
        self.graph = nx.MultiDiGraph()
        self._record_by_evidence_id = {record.evidence_id: record for record in records}
        self._evidence_node_by_id: dict[str, str] = {}
        self._build_graph()

    @staticmethod
    def _node(kind: str, value: str) -> str:
        return f"{kind}:{_normalise(value) or 'unspecified'}"

    def _add_entity(self, kind: str, value: str, *, allow_unspecified: bool = False) -> str | None:
        if not value and not allow_unspecified:
            return None
        name = value or "unspecified"
        node = self._node(kind, name)
        self.graph.add_node(node, type=kind.replace("_", " ").title(), name=name)
        return node

    def _build_graph(self) -> None:
        for record in self.records:
            study = self._node("study", record.study_id)
            article = self._node("article", record.article_id)
            evidence = self._node("evidence", record.evidence_id)
            self.graph.add_node(study, type="Study", name=record.study_id)
            self.graph.add_node(article, type="Article", name=record.article_id)
            self.graph.add_node(evidence, type="Evidence", evidence_id=record.evidence_id)
            self._evidence_node_by_id[record.evidence_id] = evidence
            self.graph.add_edge(study, article, relation="HAS_ARTICLE")
            self.graph.add_edge(article, evidence, relation="REPORTS_EVIDENCE")
            for kind, value, relation in (
                ("target", record.target_id, "ON_TARGET"),
                ("data_family", record.data_family, "USES_DATA_FAMILY"),
                ("exposure", record.intervention_or_exposure, "MEASURES_EXPOSURE"),
                ("outcome", record.outcome, "MEASURES_OUTCOME"),
                ("outcome_code", record.outcome_code, "HAS_OUTCOME_CODE"),
                ("comparator", record.comparator, "USES_COMPARATOR"),
                ("subgroup", record.population_or_subgroup, "IN_SUBGROUP"),
                ("timepoint", record.follow_up, "AT_TIMEPOINT"),
                ("estimand", record.estimand, "ESTIMATES"),
                ("adjustment_model", record.adjustment_model, "USES_ADJUSTMENT_MODEL"),
                ("effect_type", _text(record.reported_effect_type), "REPORTS_EFFECT_TYPE"),
                ("source", _source_anchor(record), "EXTRACTED_FROM"),
                ("cohort_hint", _cohort_anchor(record), "IN_COHORT_HINT"),
            ):
                entity = self._add_entity(
                    kind,
                    value,
                    allow_unspecified=kind in {
                        "data_family", "outcome_code", "estimand", "adjustment_model", "effect_type",
                    },
                )
                if entity:
                    self.graph.add_edge(evidence, entity, relation=relation)
            for field in _ANALYSIS_METADATA_FIELDS:
                entity = self._add_entity(
                    f"metadata_{field}", _text(_record_metadata(record).get(field)), allow_unspecified=True
                )
                if entity:
                    self.graph.add_edge(evidence, entity, relation=f"HAS_{field.upper()}")
            for facet in self._runtime_facet_dimensions(record):
                field = facet["field"]
                status = facet["status"]
                status_entity = self._add_entity(
                    f"runtime_facet_status_{field}", status, allow_unspecified=True
                )
                if status_entity:
                    self.graph.add_edge(
                        evidence,
                        status_entity,
                        relation=f"HAS_RUNTIME_FACET_STATUS_{_normalise(field).upper()}",
                    )
                if status in {"explicit_level", "explicit_other"}:
                    for level_id in facet["contradicted_level_ids"]:
                        entity = self._add_entity(f"runtime_contradiction_{field}", level_id)
                        if entity:
                            self.graph.add_edge(
                                evidence, entity, relation="SOURCE_CONTRADICTS_LEVEL",
                                source_path=facet["source_path"], source_quote=facet["source_quote"],
                            )
                if status != "explicit_level":
                    continue
                for level_id in facet["level_ids"]:
                    level_entity = self._add_entity(
                        f"runtime_facet_{field}", level_id, allow_unspecified=False
                    )
                    if level_entity:
                        self.graph.add_edge(
                            evidence,
                            level_entity,
                            relation=f"HAS_RUNTIME_FACET_{_normalise(field).upper()}",
                            source_path=facet["source_path"], source_quote=facet["source_quote"],
                        )

    def _runtime_facet_dimensions(self, record: EvidenceRecord) -> list[dict[str, Any]]:
        from .runtime_facets import RuntimeFacetAnnotation, _evidence_payload, _normalise_annotation

        raw = self.runtime_facets.get(record.evidence_id)
        if not isinstance(raw, dict):
            return []
        values = raw.get("dimensions")
        if not isinstance(values, list):
            return []
        try:
            annotation = RuntimeFacetAnnotation(evidence_id=record.evidence_id, dimensions=values)
        except ValueError:
            return []
        fields = {item.field for item in annotation.dimensions}
        allowed = {
            field: {
                value for item in annotation.dimensions if item.field == field
                for value in [*item.level_ids, *item.contradicted_level_ids]
            }
            for field in fields
        }
        checked = _normalise_annotation(
            annotation, requested_fields=fields, levels_by_field=allowed,
            source=_evidence_payload(record),
        )
        return [item.model_dump() for item in checked.dimensions]

    def inventory(self) -> EvidenceInventory:
        return EvidenceInventory(
            record_count=len(self.records),
            article_count=len({record.article_id for record in self.records}),
            study_count=len({record.study_id for record in self.records}),
            target_counts=dict(Counter(record.target_id for record in self.records)),
            data_family_counts=dict(Counter(record.data_family for record in self.records)),
            graph_node_count=self.graph.number_of_nodes(),
            graph_edge_count=self.graph.number_of_edges(),
        )

    def _evidence_ids_from_nodes(self, nodes: list[str]) -> set[str]:
        evidence_ids: set[str] = set()
        for node in nodes:
            for predecessor in self.graph.predecessors(node):
                data = self.graph.nodes[predecessor]
                if data.get("type") == "Evidence":
                    evidence_id = data.get("evidence_id")
                    if isinstance(evidence_id, str):
                        evidence_ids.add(evidence_id)
        return evidence_ids

    def _nodes_for_exact_values(
        self, kind: str, values: list[str], *, include_unspecified: bool = False
    ) -> list[str]:
        expected = {self._node(kind, value) for value in values if value}
        if include_unspecified:
            expected.add(self._node(kind, "unspecified"))
        return [node for node in expected if node in self.graph]

    def _nodes_for_terms(
        self, kind: str, terms: list[str], threshold: float
    ) -> tuple[list[str], list[str]]:
        if not terms:
            return [], [f"No {kind} term restriction declared."]
        matched: list[tuple[str, float, str]] = []
        node_type = kind.replace("_", " ").title()
        for node, data in self.graph.nodes(data=True):
            if data.get("type") != node_type:
                continue
            score, reason = _best_match(str(data.get("name") or ""), terms)
            if score >= threshold:
                matched.append((node, score, reason))
        matched.sort(key=lambda item: item[1], reverse=True)
        return (
            [node for node, _, _ in matched],
            [f"Graph {kind} node {node}: {reason}" for node, _, reason in matched],
        )

    def _study_context_reason(self, record: EvidenceRecord) -> str:
        evidence = self._evidence_node_by_id[record.evidence_id]
        articles = [node for node in self.graph.predecessors(evidence) if self.graph.nodes[node].get("type") == "Article"]
        studies = [
            study
            for article in articles
            for study in self.graph.predecessors(article)
            if self.graph.nodes[study].get("type") == "Study"
        ]
        return (
            f"Graph local path Study({len(studies)}) -> Article({len(articles)}) -> "
            f"Evidence({record.evidence_id})."
        )

    def retrieve(
        self,
        profile: ResolvedAnalysisProfile,
        *,
        pre_review_filter_mode: str = "strict_protocol",
        dimension_field: str | None = None,
        dimension_value_id: str | None = None,
        dimension_levels_by_field: dict[str, list[str]] | None = None,
    ) -> CandidateSet:
        """Traverse typed protocol anchors to local Evidence subgraphs.

        The only scan here is over entity nodes to resolve protocol aliases.
        Candidate Evidence itself is recovered from graph edges, never from a
        second table-field filtering pass.
        """
        all_ids = set(self._record_by_evidence_id)
        trace = [
            f"Graph contains {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.",
        ]
        if pre_review_filter_mode not in {
            "strict_protocol", "data_family_only", "reviewer_decides", "semantic_protocol",
            "semantic_enriched_protocol",
        }:
            raise ValueError(f"Unsupported pre-review filter mode: {pre_review_filter_mode}")
        target_nodes = self._nodes_for_exact_values("target", [profile.target_id])
        family_nodes = self._nodes_for_exact_values("data_family", [profile.data_family])
        target_ids = self._evidence_ids_from_nodes(target_nodes)
        family_ids = self._evidence_ids_from_nodes(family_nodes)
        if pre_review_filter_mode == "strict_protocol":
            candidate_ids = all_ids & target_ids & family_ids
            trace.append(
                f"Graph traversal Target({len(target_nodes)}) <- Evidence -> DataFamily({len(family_nodes)}) retained {len(candidate_ids)} records."
            )
        elif pre_review_filter_mode == "data_family_only":
            candidate_ids = all_ids & family_ids
            trace.append(
                f"Blind candidate mode retained {len(candidate_ids)} records using DataFamily({len(family_nodes)}) only; target, outcome, estimand, exposure, and metadata remain for Reviewer adjudication."
            )
        elif pre_review_filter_mode in {"semantic_protocol", "semantic_enriched_protocol"}:
            unknown_family_nodes = self._nodes_for_exact_values(
                "data_family", [], include_unspecified=True
            )
            candidate_ids = all_ids & (
                family_ids | self._evidence_ids_from_nodes(unknown_family_nodes)
            )
            trace.append(
                f"Semantic graph traversal through DataFamily({len(family_nodes)}) plus unknown family retained {len(candidate_ids)} records."
            )
            if profile.accepted_effect_types:
                effect_nodes = self._nodes_for_exact_values(
                    "effect_type", profile.accepted_effect_types, include_unspecified=True
                )
                candidate_ids &= self._evidence_ids_from_nodes(effect_nodes)
                trace.append(
                    f"Semantic graph traversal through EffectType({len(effect_nodes)}) retained {len(candidate_ids)} records."
                )
            for field, terms in _metadata_constraints(profile.selection_rules):
                metadata_kind = f"metadata_{field}"
                matched_nodes, _ = self._nodes_for_terms(metadata_kind, terms, threshold=0.5)
                unknown_nodes = self._nodes_for_exact_values(
                    metadata_kind, [], include_unspecified=True
                )
                before = len(candidate_ids)
                candidate_ids &= self._evidence_ids_from_nodes([*matched_nodes, *unknown_nodes])
                trace.append(
                    f"Semantic graph traversal through {field}({len(matched_nodes)} matching nodes + unknown) retained {len(candidate_ids)} records from {before}."
                )
            if pre_review_filter_mode == "semantic_enriched_protocol":
                candidate_ids = self._apply_runtime_dimension_filter(
                    candidate_ids,
                    dimension_field=dimension_field,
                    dimension_value_id=dimension_value_id,
                    dimension_levels_by_field=dimension_levels_by_field or {},
                    trace=trace,
                )
        else:
            candidate_ids = all_ids
            trace.append(
                f"Blind candidate mode retained all {len(candidate_ids)} records; data family, target, outcome, estimand, exposure, and metadata remain for Reviewer adjudication."
            )

        if pre_review_filter_mode == "strict_protocol" and profile.eligible_outcome_codes:
            outcome_code_nodes = self._nodes_for_exact_values(
                "outcome_code", profile.eligible_outcome_codes, include_unspecified=True
            )
            candidate_ids &= self._evidence_ids_from_nodes(outcome_code_nodes)
            trace.append(
                f"Graph traversal through OutcomeCode({len(outcome_code_nodes)}) retained {len(candidate_ids)} records."
            )
        if pre_review_filter_mode == "strict_protocol" and profile.eligible_estimands:
            estimand_nodes = self._nodes_for_exact_values(
                "estimand", profile.eligible_estimands, include_unspecified=True
            )
            candidate_ids &= self._evidence_ids_from_nodes(estimand_nodes)
            trace.append(
                f"Graph traversal through Estimand({len(estimand_nodes)}) retained {len(candidate_ids)} records."
            )

        exposure_nodes, exposure_reasons = self._nodes_for_terms(
            "exposure", profile.exposure_query_terms, profile.semantic_match_threshold
        )
        outcome_nodes, outcome_reasons = self._nodes_for_terms(
            "outcome", profile.outcome_query_terms, profile.semantic_match_threshold
        )
        if pre_review_filter_mode == "strict_protocol":
            if profile.exposure_query_terms:
                candidate_ids &= self._evidence_ids_from_nodes(exposure_nodes)
                trace.append(
                    f"Graph traversal through Exposure({len(exposure_nodes)}) retained {len(candidate_ids)} records."
                )
            if profile.outcome_query_terms:
                candidate_ids &= self._evidence_ids_from_nodes(outcome_nodes)
                trace.append(
                    f"Graph traversal through Outcome({len(outcome_nodes)}) retained {len(candidate_ids)} records."
                )

        matched = [
            (
                self._record_by_evidence_id[evidence_id],
                [*exposure_reasons, *outcome_reasons, self._study_context_reason(self._record_by_evidence_id[evidence_id])],
            )
            for evidence_id in sorted(candidate_ids)
        ]

        grouped: dict[tuple[str, str, str, str, str, str, str], list[tuple[EvidenceRecord, list[str]]]] = defaultdict(list)
        for record, reasons in matched:
            decision_key = (
                record.study_id,
                _normalise(record.intervention_or_exposure),
                _normalise(record.outcome),
                _normalise(record.comparator),
                _normalise(record.population_or_subgroup),
                _canonical_timepoint(record.follow_up),
                _normalise(record.estimand),
            )
            grouped[decision_key].append((record, reasons))

        groups: list[CandidateGroup] = []
        for decision_key, items in sorted(grouped.items()):
            records = [record for record, _ in items]
            reasons = list(dict.fromkeys(reason for _, item_reasons in items for reason in item_reasons))
            triplet = decision_key[:3]
            selection_context = {
                "comparison": records[0].comparator,
                "population_or_subgroup": records[0].population_or_subgroup,
                "follow_up": records[0].follow_up,
                "estimand": records[0].estimand,
            }
            groups.append(
                CandidateGroup(
                    group_id=(
                        f"triplet:{triplet[0]}:{triplet[1] or 'unspecified'}:"
                        f"{triplet[2] or 'unspecified'}:"
                        f"context:{decision_key[3] or 'unspecified'}:"
                        f"{decision_key[4] or 'overall'}:{decision_key[5] or 'unspecified'}:"
                        f"{decision_key[6] or 'unspecified'}"
                    ),
                    study_id=triplet[0],
                    article_ids=sorted({record.article_id for record in records}),
                    evidence_ids=[record.evidence_id for record in records],
                    records=records,
                    exposure=records[0].intervention_or_exposure,
                    outcome=records[0].outcome,
                    comparator=records[0].comparator,
                    triplet=triplet,
                    selection_context=selection_context,
                    retrieval_reasons=reasons,
                )
            )
        return CandidateSet(
            target_id=profile.target_id,
            groups=groups,
            retrieval_trace=[
                *trace,
                f"Typed graph traversal returned {len(matched)} records in {len(groups)} local study-exposure-outcome subgraphs.",
            ],
        )

    def _apply_runtime_dimension_filter(
        self,
        candidate_ids: set[str],
        *,
        dimension_field: str | None,
        dimension_value_id: str | None,
        dimension_levels_by_field: dict[str, list[str]],
        trace: list[str],
    ) -> set[str]:
        """Remove only source-explicit facet contradictions for one topic.

        An unreported or mixed source dimension remains in the local graph for
        Reviewer adjudication.  This is deliberately asymmetric: a facet can
        prove an explicit mismatch, but cannot prove a record irrelevant.
        """
        if not dimension_field or not dimension_value_id:
            trace.append("Runtime facet filter: this topic has no declared dimension level.")
            return candidate_ids
        all_levels = [
            value for value in dimension_levels_by_field.get(dimension_field, []) if value
        ]
        if dimension_value_id not in all_levels:
            trace.append(
                f"Runtime facet filter: {dimension_field}/{dimension_value_id} is absent from the declared dimension map; no records removed."
            )
            return candidate_ids

        facet_kind = f"runtime_facet_{dimension_field}"
        target_nodes = self._nodes_for_exact_values(facet_kind, [dimension_value_id])
        target_ids = self._evidence_ids_from_nodes(target_nodes)
        contradiction_nodes = self._nodes_for_exact_values(
            f"runtime_contradiction_{dimension_field}", [dimension_value_id]
        )
        # Different levels can overlap: only an independently stated source
        # contradiction removes a record. Conflicting positive edges win.
        explicit_contradiction_ids = self._evidence_ids_from_nodes(contradiction_nodes) - target_ids
        before = len(candidate_ids)
        excluded_here = len(candidate_ids & explicit_contradiction_ids)
        candidate_ids = candidate_ids - explicit_contradiction_ids
        trace.append(
            "Runtime facet graph traversal through "
            f"{dimension_field}={dimension_value_id} retained {len(candidate_ids)} records from {before}; "
            f"kept {len(candidate_ids & target_ids)} explicit target records and retained mixed/unreported records, "
            f"while excluding {excluded_here} explicit non-target records."
        )
        return candidate_ids
