from __future__ import annotations

"""Study--exposure--outcome GraphRAG for the analysis agent.

This is the benchmark-compatible successor to ``MetaGraphRAG``. It preserves
the original graph idea (paper/study -> evidence -> exposure/outcome) while
accepting normalized extractor records rather than a project-specific CSV.
Hard protocol constraints are applied before lexical semantic expansion, so an
embedding model is an optional future improvement rather than a prerequisite
for a reproducible analysis run.
"""

from collections import Counter, defaultdict
import re
from typing import Any

import networkx as nx

from .models import CandidateGroup, CandidateSet, EvidenceInventory, EvidenceRecord, ResolvedAnalysisProfile


def _text(value: object) -> str:
    return str(value or "").strip()


def _effect_value(item: dict[str, Any], field: str) -> object:
    effect = item.get("reported_effect") or item.get("effect") or {}
    return item.get(field) if item.get(field) is not None else effect.get(field)


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
            evidence_id = _text(item.get("evidence_id") or f"{article_id}:evidence:{index}")
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
                    outcome=_text(item.get("outcome")),
                    outcome_code=_text(item.get("outcome_code")),
                    estimand=_text(item.get("estimand")),
                    adjustment_model=_text(item.get("adjustment_model") or item.get("model")),
                    population_or_subgroup=_text(item.get("population_or_subgroup")),
                    follow_up=_text(item.get("follow_up") or item.get("timepoint")),
                    data_family=_text(item.get("data_family")),
                    raw_data=dict(item.get("raw_data") or {}),
                    derivation_status=_text(item.get("derivation_status")),
                    quantitative_usability=_text(item.get("quantitative_usability")),
                    reported_effect_type=_effect_value(item, "effect_type") or item.get("reported_effect_type"),
                    reported_effect_value=_effect_value(item, "value") or item.get("reported_effect_value"),
                    ci_lower=_effect_value(item, "ci_lower"),
                    ci_upper=_effect_value(item, "ci_upper"),
                    source_location=dict(item.get("source_location") or {}),
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


class EvidenceGraph:
    """Explicit GraphRAG structure for study-level evidence selection.

    Nodes: Study, Article, Evidence, Exposure, Outcome, Comparator, Subgroup,
    Timepoint. The key retrieval unit remains the original triad:
    ``(study, exposure, outcome)``. Extra relation nodes preserve context for
    the reviewer rather than collapsing competing models or follow-ups early.
    """

    def __init__(self, records: list[EvidenceRecord]):
        self.records = records
        self.graph = nx.MultiDiGraph()
        self._build_graph()

    @staticmethod
    def _node(kind: str, value: str) -> str:
        return f"{kind}:{_normalise(value) or 'unspecified'}"

    def _add_entity(self, kind: str, value: str) -> str | None:
        if not value:
            return None
        node = self._node(kind, value)
        self.graph.add_node(node, type=kind.title(), name=value)
        return node

    def _build_graph(self) -> None:
        for record in self.records:
            study = self._node("study", record.study_id)
            article = self._node("article", record.article_id)
            evidence = self._node("evidence", record.evidence_id)
            self.graph.add_node(study, type="Study", name=record.study_id)
            self.graph.add_node(article, type="Article", name=record.article_id)
            self.graph.add_node(evidence, type="Evidence", evidence_id=record.evidence_id)
            self.graph.add_edge(study, article, relation="HAS_ARTICLE")
            self.graph.add_edge(article, evidence, relation="REPORTS_EVIDENCE")
            for kind, value, relation in (
                ("exposure", record.intervention_or_exposure, "MEASURES_EXPOSURE"),
                ("outcome", record.outcome, "MEASURES_OUTCOME"),
                ("comparator", record.comparator, "USES_COMPARATOR"),
                ("subgroup", record.population_or_subgroup, "IN_SUBGROUP"),
                ("timepoint", record.follow_up, "AT_TIMEPOINT"),
                ("estimand", record.estimand, "ESTIMATES"),
            ):
                entity = self._add_entity(kind, value)
                if entity:
                    self.graph.add_edge(evidence, entity, relation=relation)

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

    def retrieve(self, profile: ResolvedAnalysisProfile) -> CandidateSet:
        """Retrieve PICO-compatible evidence subgraphs before LLM review.

        Target ID/data family/estimand are hard filters. Exposure and outcome
        anchors then use deterministic lexical expansion; an embedding backend
        can replace only that score later without changing the graph contract.
        """
        hard_matches = [
            record for record in self.records
            if record.target_id == profile.target_id
            and record.data_family == profile.data_family
            and (
                not profile.eligible_outcome_codes
                or not record.outcome_code
                or record.outcome_code in profile.eligible_outcome_codes
            )
            and (
                not profile.eligible_estimands
                or not record.estimand
                or record.estimand in profile.eligible_estimands
            )
        ]
        matched: list[tuple[EvidenceRecord, list[str]]] = []
        for record in hard_matches:
            exposure_score, exposure_reason = _best_match(
                record.intervention_or_exposure, profile.exposure_query_terms
            )
            outcome_score, outcome_reason = _best_match(record.outcome, profile.outcome_query_terms)
            if (
                exposure_score < profile.semantic_match_threshold
                or outcome_score < profile.semantic_match_threshold
            ):
                continue
            matched.append((record, [exposure_reason, outcome_reason]))

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
                f"Graph contains {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.",
                f"Hard target/data-family/estimand filters retained {len(hard_matches)} evidence records.",
                f"PICO lexical expansion retained {len(matched)} records in {len(groups)} study-exposure-outcome subgraphs.",
            ],
        )
