from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from .dataset import DocumentStore
from .indexes import BM25Index, CrossEncoderReranker, DenseIndex


@dataclass(frozen=True)
class QueryVariant:
    name: str
    text: str
    weight: float = 1.0


@dataclass
class RetrievalCandidate:
    corpus_id: int
    score: float
    rank: int
    title: str = ""
    abstract: str = ""
    pmid: str = ""
    year: Optional[int] = None
    has_sections: bool = False
    trace: List[Dict[str, Any]] | None = None

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["corpus_id"] = int(value["corpus_id"])
        return value


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


class StructuredQueryPlanner:
    """Deterministic multi-query planner using only declared review inputs."""

    def plan(self, task: Dict[str, Any]) -> List[QueryVariant]:
        research = _clean(task.get("Research_Question") or task.get("Title"))
        population = _clean(task.get("Population"))
        intervention = _clean(task.get("Intervention") or task.get("Exposure"))
        comparison = _clean(task.get("Comparison"))
        outcome = _clean(task.get("Outcome"))
        structured = " ".join(
            part for part in (research, population, intervention, comparison, outcome) if part
        )
        focused = " ".join(part for part in (population, intervention, outcome) if part)
        high_recall = " ".join(part for part in (intervention, outcome) if part)
        variants = [
            QueryVariant("research_question", research, 1.0),
            QueryVariant("structured_pieco", structured, 1.2),
            QueryVariant("focused_pio", focused, 1.0),
            QueryVariant("high_recall_io", high_recall, 0.7),
        ]
        unique: Dict[str, QueryVariant] = {}
        for variant in variants:
            if variant.text:
                unique.setdefault(variant.text.casefold(), variant)
        return list(unique.values())

    @staticmethod
    def screening_query(task: Dict[str, Any]) -> str:
        fields = [
            f"Research: {_clean(task.get('Research_Question') or task.get('Title'))}",
            f"Population: {_clean(task.get('Population'))}",
            f"Intervention or exposure: {_clean(task.get('Intervention') or task.get('Exposure'))}",
            f"Comparison: {_clean(task.get('Comparison'))}",
            f"Outcome: {_clean(task.get('Outcome'))}",
        ]
        return "\n".join(value for value in fields if not value.endswith(": "))


class HybridRetriever:
    def __init__(
        self,
        store: DocumentStore,
        bm25: Optional[BM25Index] = None,
        dense: Optional[DenseIndex] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: int = 60,
    ):
        if bm25 is None and dense is None:
            raise ValueError("At least one retriever must be configured")
        self.store = store
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.rrf_k = int(rrf_k)

    def retrieve(
        self,
        task: Dict[str, Any],
        variants: Optional[Sequence[QueryVariant]] = None,
        per_query_k: int = 1000,
        fusion_k: int = 500,
        final_k: int = 200,
        max_year: Optional[int] = None,
    ) -> List[RetrievalCandidate]:
        variants = list(variants or StructuredQueryPlanner().plan(task))
        scores: Dict[int, float] = {}
        traces: Dict[int, List[Dict[str, Any]]] = {}
        retrievers = [("bm25", self.bm25), ("dense", self.dense)]
        for query in variants:
            for retriever_name, retriever in retrievers:
                if retriever is None:
                    continue
                ranking = retriever.search(query.text, top_k=per_query_k, max_year=max_year)
                for rank, (corpus_id, raw_score) in enumerate(ranking, start=1):
                    contribution = query.weight / (self.rrf_k + rank)
                    scores[corpus_id] = scores.get(corpus_id, 0.0) + contribution
                    traces.setdefault(corpus_id, []).append(
                        {
                            "retriever": retriever_name,
                            "query": query.name,
                            "rank": rank,
                            "raw_score": raw_score,
                            "rrf_contribution": contribution,
                        }
                    )
        fused_ids = sorted(scores, key=scores.get, reverse=True)[:fusion_k]
        metadata = self.store.get_many(fused_ids)
        by_id = {int(item["corpus_id"]): item for item in metadata}

        if self.reranker is not None and fused_ids:
            rerank_docs = [by_id[value] for value in fused_ids if value in by_id]
            reranked = self.reranker.rerank(
                StructuredQueryPlanner.screening_query(task), rerank_docs, top_k=final_k
            )
            ordered_ids = [value[0] for value in reranked]
            final_scores = {value[0]: value[1] for value in reranked}
        else:
            ordered_ids = fused_ids[:final_k]
            final_scores = scores

        candidates: List[RetrievalCandidate] = []
        for rank, corpus_id in enumerate(ordered_ids, start=1):
            item = by_id.get(corpus_id)
            if item is None:
                continue
            candidates.append(
                RetrievalCandidate(
                    corpus_id=corpus_id,
                    score=float(final_scores[corpus_id]),
                    rank=rank,
                    title=str(item.get("title") or ""),
                    abstract=str(item.get("abstract") or ""),
                    pmid=str(item.get("pmid") or ""),
                    year=item.get("year"),
                    has_sections=bool(item.get("has_sections")),
                    trace=traces.get(corpus_id, []),
                )
            )
        return candidates
