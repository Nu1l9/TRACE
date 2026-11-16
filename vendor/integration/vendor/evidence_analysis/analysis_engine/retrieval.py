from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from .evidence_graph import EvidenceGraph
from analysis_core.models import CandidateSet, EvidenceRecord, ResolvedAnalysisPlan
from analysis_core.plan_resolver import runtime_profile_for_analysis
from .eligibility import eligibility_reasons

from .models import TopicCandidateInventory


class EmbeddingBackend(Protocol):
    mode: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingBackend:
    """Deterministic offline stand-in used only in tests and no-API runs."""

    mode = "local_hash_fallback"

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in text.casefold().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:2], "big") % self.dimensions
                vector[index] += 1.0 if digest[2] % 2 else -1.0
            vectors.append(_unit(vector))
        return vectors


class OpenAIEmbeddingBackend:
    """Official OpenAI embedding client with optional local vector cache."""

    mode = "openai_embedding"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        cache_path: Path | None = None,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.cache_path = cache_path
        self.max_retries = max_retries
        self.embedding_requests = 0
        self.remote_batch_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        if not self.api_key:
            raise RuntimeError("OpenAI embedding retrieval needs OPENAI_API_KEY.")
        self.embedding_requests += 1
        cache = self._load_cache()
        keys = [self._cache_key(text) for text in texts]
        missing = [text for key, text in zip(keys, texts) if key not in cache]
        self.cache_hit_count += len(texts) - len(missing)
        self.cache_miss_count += len(missing)
        if missing:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=self.max_retries,
                timeout=120.0,
            )
            for start in range(0, len(missing), 96):
                batch = missing[start:start + 96]
                self.remote_batch_count += 1
                response = client.embeddings.create(model=self.model, input=batch)
                if not isinstance(response.data, list) or len(response.data) != len(batch):
                    raise RuntimeError(
                        "Embedding endpoint did not return an OpenAI-compatible vector list. "
                        "Use the official OpenAI /v1 endpoint with OPENAI_API_KEY; do not point "
                        "ANALYSIS_EMBEDDING_BASE_URL at a chat-only proxy."
                    )
                for text, item in zip(batch, response.data):
                    if not isinstance(item.embedding, list):
                        raise RuntimeError(
                            "Embedding endpoint returned a malformed vector. "
                            "Use the official OpenAI embedding endpoint."
                        )
                    cache[self._cache_key(text)] = item.embedding
            self._save_cache(cache)
        return [cache[key] for key in keys]

    def diagnostics(self) -> dict[str, int]:
        """Expose actual retrieval transport use for a run-level audit record."""
        return {
            "embedding_requests": self.embedding_requests,
            "remote_batch_count": self.remote_batch_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
        }

    def _cache_key(self, text: str) -> str:
        # A model alias can be served by different compatible providers.  Do
        # not reuse vectors across endpoints: that would make an "official
        # OpenAI" retrieval experiment partly depend on a third-party cache.
        namespace = f"{self.base_url.rstrip('/')}\0{self.model}"
        return hashlib.sha256(f"{namespace}\0{text}".encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, list[float]]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_cache(self, cache: dict[str, list[float]]) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache), encoding="utf-8")


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _evidence_text(record: EvidenceRecord) -> str:
    raw_data = record.raw_data if isinstance(record.raw_data, dict) else {}
    metadata = raw_data.get("_analysis_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata_text = [str(metadata.get(field) or raw_data.get(field) or "") for field in (
        "outcome_axis", "measurement_semantics", "unit", "intervention_class", "comparator_class",
        "rsid", "study_design", "phenotype", "genotype_role",
    )]
    return " ".join(filter(None, [
        record.target_id,
        record.intervention_or_exposure,
        record.outcome,
        record.outcome_code,
        record.comparator,
        record.population_or_subgroup,
        record.follow_up,
        record.estimand,
        record.adjustment_model,
        record.data_family,
        record.reported_effect_type or "",
        *metadata_text,
    ]))


def _analysis_query(definition) -> str:
    return " ".join(filter(None, [
        definition.target_id,
        definition.outcome_group,
        *definition.eligible_outcome_codes,
        *definition.exposure_terms,
        *definition.comparison_terms,
        *definition.population_terms,
        *definition.dimension_terms,
        *definition.estimands,
        *definition.accepted_effect_types,
        *definition.eligible_data_families,
        *definition.selection_rules,
    ]))


class HybridGraphRetriever:
    """Hard graph constraints followed by embedding ranking of local subgraphs."""

    def __init__(
        self,
        graph: EvidenceGraph,
        embeddings: EmbeddingBackend,
        max_groups: int = 192,
        minimum_groups_per_study: int = 3,
        pre_review_filter_mode: str = "strict_protocol",
    ):
        self.graph = graph
        self.embeddings = embeddings
        self.max_groups = max_groups
        self.minimum_groups_per_study = minimum_groups_per_study
        self.pre_review_filter_mode = pre_review_filter_mode

    @staticmethod
    def _dimension_levels_by_field(plan: ResolvedAnalysisPlan) -> dict[str, list[str]]:
        """Pass public dimension levels to the graph, never candidate labels."""
        by_field: dict[str, list[str]] = {}
        for dimension in plan.candidate_dimensions:
            levels = by_field.setdefault(dimension.field, [])
            for level in dimension.levels:
                if level.value_id not in levels:
                    levels.append(level.value_id)
        return by_field

    def _graph_candidates(self, definition, plan: ResolvedAnalysisPlan) -> CandidateSet:
        profile = runtime_profile_for_analysis(definition, plan)
        return self.graph.retrieve(
            profile,
            pre_review_filter_mode=self.pre_review_filter_mode,
            dimension_field=definition.dimension_field,
            dimension_value_id=definition.dimension_value_id,
            dimension_levels_by_field=self._dimension_levels_by_field(plan),
        )

    def inventory(self, plan: ResolvedAnalysisPlan) -> list[TopicCandidateInventory]:
        """Create a graph-only topic inventory before paying for embeddings.

        This is deliberately a coarse triage view: it contains counts and
        independent-study coverage, but does not rank or expose the local
        Evidence contexts that the Reviewer will later receive.
        """
        inventories: list[TopicCandidateInventory] = []
        for definition in plan.analyses:
            broad = self._graph_candidates(definition, plan)
            eligible_groups = []
            for group in broad.groups:
                eligible_records = (
                    [record for record in group.records if not eligibility_reasons(record, definition)]
                    if self.pre_review_filter_mode == "strict_protocol" else list(group.records)
                )
                if eligible_records:
                    eligible_groups.append(group.model_copy(update={
                        "records": eligible_records,
                        "evidence_ids": [record.evidence_id for record in eligible_records],
                        "article_ids": sorted({record.article_id for record in eligible_records}),
                    }))
            records = [record for group in eligible_groups for record in group.records]
            inventories.append(TopicCandidateInventory(
                analysis_id=definition.analysis_id,
                candidate_record_count=len(records),
                candidate_context_count=len(eligible_groups),
                candidate_study_count=len({record.study_id for record in records}),
                embedding_retrieval_summary=(
                    f"graph_constraint_inventory: {len(records)} Evidence in "
                    f"{len(eligible_groups)} local graph contexts."
                ),
            ))
        return inventories

    def retrieve_all(
        self, plan: ResolvedAnalysisPlan
    ) -> tuple[dict[str, CandidateSet], list[TopicCandidateInventory]]:
        definitions = plan.analyses
        queries = [_analysis_query(definition) for definition in definitions]
        query_vectors = self.embeddings.embed(queries)

        # First recover candidates through graph relations. Only those local
        # Evidence records are embedded; unrelated rows never enter ranking.
        broad_by_id: dict[str, CandidateSet] = {}
        eligible_by_id: dict[str, list] = {}
        candidate_records: dict[str, EvidenceRecord] = {}
        for definition in definitions:
            broad = self._graph_candidates(definition, plan)
            broad_by_id[definition.analysis_id] = broad
            eligible_groups = []
            for group in broad.groups:
                eligible_records = (
                    [record for record in group.records if not eligibility_reasons(record, definition)]
                    if self.pre_review_filter_mode == "strict_protocol" else list(group.records)
                )
                if eligible_records:
                    candidate_records.update({record.evidence_id: record for record in eligible_records})
                    eligible_groups.append(group.model_copy(update={
                        "records": eligible_records,
                        "evidence_ids": [record.evidence_id for record in eligible_records],
                        "article_ids": sorted({record.article_id for record in eligible_records}),
                    }))
            eligible_by_id[definition.analysis_id] = eligible_groups

        ordered_records = [candidate_records[evidence_id] for evidence_id in sorted(candidate_records)]
        evidence_vectors = {
            record.evidence_id: vector
            for record, vector in zip(
                ordered_records,
                self.embeddings.embed([_evidence_text(record) for record in ordered_records]),
            )
        }

        candidate_sets: dict[str, CandidateSet] = {}
        inventories: list[TopicCandidateInventory] = []
        for definition, query_vector in zip(definitions, query_vectors):
            broad = broad_by_id[definition.analysis_id]
            eligible_groups = eligible_by_id[definition.analysis_id]
            scored_groups = sorted(
                (
                    (
                        max(
                            _cosine(query_vector, evidence_vectors[record.evidence_id])
                            for record in group.records
                        ),
                        group,
                    )
                    for group in eligible_groups
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            # Do not silently erase entire studies before local review merely
            # because one large extraction run emitted many contexts for other
            # studies. Retain a small ranked slate for every study, then fill
            # the remaining review budget globally by embedding relevance.
            retained: list[tuple[float, object]] = []
            retained_group_ids: set[str] = set()
            by_study: dict[str, list[tuple[float, object]]] = defaultdict(list)
            for scored_group in scored_groups:
                by_study[scored_group[1].study_id].append(scored_group)
            for study_id in sorted(by_study):
                for score, group in by_study[study_id][:self.minimum_groups_per_study]:
                    if len(retained) >= self.max_groups:
                        break
                    retained.append((score, group))
                    retained_group_ids.add(group.group_id)
            for score, group in scored_groups:
                if len(retained) >= self.max_groups:
                    break
                if group.group_id not in retained_group_ids:
                    retained.append((score, group))
                    retained_group_ids.add(group.group_id)
            scored_groups = sorted(retained, key=lambda item: item[0], reverse=True)
            # Preserve rank and cosine score with the selected graph context.
            # This makes the retrieval decision inspectable after a run rather
            # than collapsing a real embedding call into a count-only log.
            ranked_groups = [
                group.model_copy(update={
                    "retrieval_reasons": [
                        *group.retrieval_reasons,
                        f"embedding_rank={rank}; cosine_score={score:.6f}",
                    ],
                })
                for rank, (score, group) in enumerate(scored_groups, start=1)
            ]
            candidates = CandidateSet(
                target_id=broad.target_id,
                groups=ranked_groups,
                retrieval_trace=[
                    *broad.retrieval_trace,
                    f"Pre-review mode={self.pre_review_filter_mode}; retained {len(eligible_groups)} contexts for local reviewer adjudication.",
                    f"Embedding backend={self.embeddings.mode}; retained {len(ranked_groups)} graph contexts "
                    f"with up to {self.minimum_groups_per_study} ranked contexts per study before global fill.",
                ],
            )
            candidate_sets[definition.analysis_id] = candidates
            records = [record for group in candidates.groups for record in group.records]
            inventories.append(TopicCandidateInventory(
                analysis_id=definition.analysis_id,
                candidate_record_count=len(records),
                candidate_context_count=len(candidates.groups),
                candidate_study_count=len({record.study_id for record in records}),
                embedding_retrieval_summary=(
                    f"{self.embeddings.mode}: {len(records)} Evidence in {len(candidates.groups)} local graph contexts."
                ),
            ))
        return candidate_sets, inventories
