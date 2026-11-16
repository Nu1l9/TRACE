"""Public API for the MINERS evidence-synthesis workflow."""

from .case_workflow import CandidateCaseConfig, run_candidate_case
from .workflow import (
    HybridReviewConfig,
    HybridReviewWorkflow,
    CorpusPaths,
    ReviewRunResult,
    build_bm25_index,
    build_dense_index,
    prepare_document_store,
)

__version__ = "0.5.0"

__all__ = [
    "CandidateCaseConfig",
    "HybridReviewConfig",
    "HybridReviewWorkflow",
    "CorpusPaths",
    "ReviewRunResult",
    "build_bm25_index",
    "build_dense_index",
    "prepare_document_store",
    "run_candidate_case",
]
