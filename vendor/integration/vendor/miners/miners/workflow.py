"""Public end-to-end MINERS workflow starting from Hybrid retrieval.

Hybrid retrieval produces the candidate pool before any PICO decision is made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .dataset import DocumentStore, ReviewDataset
from .evaluation import evaluate_result, load_result
from .indexes import BM25Index, CrossEncoderReranker, DenseIndex
from .paper_node_adapter import build_llm_callback
from .reporting import generate_synthesis_report, write_run_output
from .retrieval import HybridRetriever, StructuredQueryPlanner
from .rob_routing import RiskOfBiasRecommendationAgent
from .screening import ProtocolScreener, included_and_unresolved


RetrievalMode = Literal["bm25", "dense", "hybrid"]
FullTextPolicy = Literal["unsure", "all-available"]
UnsurePolicy = Literal["include", "exclude"]
ProtocolMode = Literal["strict", "oracle"]


@dataclass(frozen=True)
class CorpusPaths:
    """Local review corpus, cache, and index locations."""

    papers: Path = Path("data/papers.json")
    test_ids: Path = Path("data/test_ids.json")
    corpus: Path = Path("data/corpus.json")
    store: Path = Path("cache/miners/documents.sqlite")
    bm25_index: Path = Path("cache/miners/bm25")
    dense_index: Path = Path("cache/miners/dense")


@dataclass(frozen=True)
class HybridReviewConfig:
    """Configuration for the complete retrieval-to-screening review workflow."""

    paths: CorpusPaths
    output_dir: Path = Path("runs/miners")
    mode: RetrievalMode = "hybrid"
    per_query_k: int = 1000
    fusion_k: int = 500
    final_k: int = 200
    date_filter: bool = True
    reranker_model: str | None = None
    screen: bool = False
    fulltext: bool = False
    fulltext_policy: FullTextPolicy = "unsure"
    protocol_mode: ProtocolMode = "strict"
    unsure_policy: UnsurePolicy = "exclude"
    generate_report: bool = False
    route_rob: bool = False
    rob_review_type: str = "auto"
    expected_designs: tuple[str, ...] = ()
    max_rob_fulltext_chars: int = 120_000
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("mode must be one of: bm25, dense, hybrid")
        if self.per_query_k < 1 or self.fusion_k < 1 or self.final_k < 1:
            raise ValueError("per_query_k, fusion_k, and final_k must all be positive")
        if self.final_k > self.fusion_k:
            raise ValueError("final_k cannot exceed fusion_k")
        if self.fulltext_policy not in {"unsure", "all-available"}:
            raise ValueError("fulltext_policy must be 'unsure' or 'all-available'")
        if self.unsure_policy not in {"include", "exclude"}:
            raise ValueError("unsure_policy must be 'include' or 'exclude'")
        if self.generate_report and not self.screen:
            raise ValueError("generate_report requires screen=True")
        if self.fulltext and not self.screen:
            raise ValueError("fulltext requires screen=True")
        if self.route_rob and not self.screen:
            raise ValueError("route_rob requires screen=True")
        if self.max_rob_fulltext_chars < 1:
            raise ValueError("max_rob_fulltext_chars must be positive")


@dataclass(frozen=True)
class ReviewRunResult:
    """Artifacts created for one review task."""

    paper_id: int
    results_path: Path
    report_path: Path
    manual_review_path: Path
    metrics_path: Path
    metrics: dict[str, Any]
    risk_of_bias_path: Path | None = None


def prepare_document_store(
    corpus: str | Path,
    store: str | Path,
    *,
    limit: int | None = None,
    rebuild: bool = False,
) -> DocumentStore:
    """Create the SQLite document store required by all MINERS retrievers."""

    return DocumentStore.build(corpus, store, limit=limit, rebuild=rebuild)


def build_bm25_index(
    store: str | Path,
    index_dir: str | Path,
    *,
    max_features: int = 300_000,
    min_df: int = 2,
    rebuild: bool = False,
) -> BM25Index:
    """Build or open the sparse retrieval index."""

    return BM25Index.build(
        DocumentStore(store),
        index_dir,
        max_features=max_features,
        min_df=min_df,
        rebuild=rebuild,
    )


def build_dense_index(
    store: str | Path,
    index_dir: str | Path,
    *,
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 64,
    rebuild: bool = False,
) -> DenseIndex:
    """Build or open the dense embedding index used by Hybrid retrieval."""

    return DenseIndex.build(
        DocumentStore(store),
        index_dir,
        model_name=model_name,
        batch_size=batch_size,
        rebuild=rebuild,
    )


class HybridReviewWorkflow:
    """Run MINERS Hybrid retrieval, PICO screening, full-text resolution, and reporting.

    The workflow has one source of truth for the candidate order: ``HybridRetriever``.
    Screening never sees evaluation labels, and screening is optional so the retrieval
    layer can be evaluated independently.
    """

    def __init__(self, config: HybridReviewConfig):
        self.config = config
        self.dataset = ReviewDataset(config.paths.papers, config.paths.test_ids)
        self.store = DocumentStore(config.paths.store)
        self.retriever = self._build_retriever()
        self.screener = self._build_screener()
        self.rob_router = self._build_rob_router()

    def _build_retriever(self) -> HybridRetriever:
        bm25 = BM25Index(self.config.paths.bm25_index) if self.config.mode in {"bm25", "hybrid"} else None
        dense = DenseIndex(self.config.paths.dense_index) if self.config.mode in {"dense", "hybrid"} else None
        reranker = (
            CrossEncoderReranker(self.config.reranker_model)
            if self.config.reranker_model
            else None
        )
        return HybridRetriever(self.store, bm25=bm25, dense=dense, reranker=reranker)

    def _build_screener(self) -> ProtocolScreener | None:
        if not self.config.screen:
            return None
        return ProtocolScreener(
            model=self.config.model,
            api_key_env=self.config.api_key_env,
            base_url=self.config.base_url,
            protocol_mode=self.config.protocol_mode,
        )

    def _build_rob_router(self) -> RiskOfBiasRecommendationAgent | None:
        if not self.config.route_rob:
            return None
        callback = build_llm_callback(
            model=self.config.model,
            api_key_env=self.config.api_key_env,
            base_url=self.config.base_url,
            system_message=RiskOfBiasRecommendationAgent.system_message(),
        )
        return RiskOfBiasRecommendationAgent(callback)

    @staticmethod
    def _max_year(task: dict[str, Any], enabled: bool) -> int | None:
        if not enabled:
            return None
        value = task.get("search_end_date")
        try:
            return int(str(value)[:4]) if value else None
        except (TypeError, ValueError):
            return None

    def run(self, paper_ids: list[int]) -> list[ReviewRunResult]:
        """Run one or more tasks, reusing loaded models and indexes."""

        if not paper_ids:
            raise ValueError("paper_ids cannot be empty")
        return [self.run_one(int(paper_id)) for paper_id in paper_ids]

    def run_one(self, paper_id: int) -> ReviewRunResult:
        """Run one review task from Hybrid retrieval through final audit artifacts."""

        task = self.dataset.get(paper_id)
        variants = StructuredQueryPlanner().plan(task)
        candidates = self.retriever.retrieve(
            task,
            variants=variants,
            per_query_k=self.config.per_query_k,
            fusion_k=self.config.fusion_k,
            final_k=self.config.final_k,
            max_year=self._max_year(task, self.config.date_filter),
        )

        decisions: list[dict[str, Any]] = []
        included_ids: list[int] = []
        unresolved_ids: list[int] = []
        if self.screener is not None:
            decisions = self.screener.screen_abstracts(task, candidates)
            if self.config.fulltext:
                decisions = self.screener.resolve_unsure_with_fulltext(
                    task,
                    decisions,
                    self.store,
                    rescreen_all_available=self.config.fulltext_policy == "all-available",
                )
            else:
                for decision in decisions:
                    decision.setdefault("final_status", decision.get("status", "Unresolved"))
            included_ids, unresolved_ids = included_and_unresolved(
                decisions,
                unsure_policy=self.config.unsure_policy,
            )

        output_dir = self.config.output_dir / self.config.mode / f"paper_{paper_id}"
        run_config = {
            "pipeline": "miners_hybrid_retrieval_to_pico_screening",
            "mode": self.config.mode,
            "queries": [variant.__dict__ for variant in variants],
            "per_query_k": self.config.per_query_k,
            "fusion_k": self.config.fusion_k,
            "final_k": self.config.final_k,
            "date_filter": self.config.date_filter,
            "screen": self.config.screen,
            "fulltext": self.config.fulltext,
            "fulltext_policy": self.config.fulltext_policy,
            "protocol_mode": self.config.protocol_mode,
            "unsure_policy": self.config.unsure_policy,
        }
        generated_report = None
        if self.config.generate_report:
            generated_report = generate_synthesis_report(self.screener.callback, task, decisions)
        paths = write_run_output(
            output_dir,
            task,
            candidates,
            decisions,
            included_ids,
            unresolved_ids,
            run_config,
            generated_report=generated_report,
        )
        metrics = evaluate_result(task, load_result(paths["results"]))
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        risk_of_bias_path = self._route_included_studies(task, decisions, output_dir)
        return ReviewRunResult(
            paper_id=paper_id,
            results_path=paths["results"],
            report_path=paths["report"],
            manual_review_path=paths["manual_review"],
            metrics_path=metrics_path,
            metrics=metrics,
            risk_of_bias_path=risk_of_bias_path,
        )

    def _route_included_studies(
        self,
        task: dict[str, Any],
        decisions: list[dict[str, Any]],
        output_dir: Path,
    ) -> Path | None:
        """Route only final Includes, retaining full-text availability as an audit state."""

        if self.rob_router is None:
            return None
        recommendations: list[dict[str, Any]] = []
        for decision in decisions:
            if (decision.get("final_status") or decision.get("status")) != "Include":
                continue
            corpus_id = int(decision["corpus_id"])
            recommendation = self.rob_router.recommend(
                {
                    "study_id": str(corpus_id),
                    "corpus_id": corpus_id,
                    "pmid": decision.get("pmid", ""),
                    "title": decision.get("title", ""),
                    "abstract": decision.get("abstract", ""),
                },
                self.store.get_fulltext_sections(corpus_id),
                review_question=str(task.get("Research_Question") or task.get("Title") or ""),
                review_type=self.config.rob_review_type,
                expected_designs=self.config.expected_designs,
                max_total_chars=self.config.max_rob_fulltext_chars,
            )
            recommendation["corpus_id"] = corpus_id
            recommendation["screening_status"] = "Include"
            recommendations.append(recommendation)
        output_path = output_dir / "risk_of_bias_recommendations.json"
        output_path.write_text(
            json.dumps(
                {
                    "paper_id": int(task["ID"]),
                    "routing_scope": "final Include decisions only",
                    "recommendations": recommendations,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return output_path
