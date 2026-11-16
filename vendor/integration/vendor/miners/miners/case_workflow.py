"""Supplementary workflow for a supplied mixed candidate set.

This mode is intentionally separate from ``HybridReviewWorkflow``.  It is for
experiments in which candidate PDFs have already been supplied, so it does not
claim that Hybrid retrieval was performed for that run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .case_screening_rob import (
    FullTextProtocolScreener,
    run_case_screening_rob,
)
from .paper_node_adapter import build_llm_callback
from .rob_routing import RiskOfBiasRecommendationAgent


@dataclass(frozen=True)
class CandidateCaseConfig:
    """Inputs for PICO screening and RoB routing of supplied candidate PDFs."""

    specs_zip: Path
    positives_zip: Path
    negatives_zip: Path
    case_id: str
    output_dir: Path = Path("runs/miners_case_review")
    model: str = "gpt-5.6-luna"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    review_type: str = "auto"
    expected_designs: tuple[str, ...] = ()
    max_study_pages: int = 50
    max_chars_per_page: int = 12_000
    max_screening_fulltext_chars: int = 120_000
    max_fulltext_adjudication_attempts: int = 3
    max_rob_fulltext_chars: int = 120_000
    resume_unsure: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        for label, path in (
            ("specs_zip", self.specs_zip),
            ("positives_zip", self.positives_zip),
            ("negatives_zip", self.negatives_zip),
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if self.max_fulltext_adjudication_attempts < 1:
            raise ValueError("max_fulltext_adjudication_attempts must be positive")


def _args_from_config(config: CandidateCaseConfig) -> SimpleNamespace:
    return SimpleNamespace(
        specs_zip=str(config.specs_zip),
        positives_zip=str(config.positives_zip),
        negatives_zip=str(config.negatives_zip),
        case_id=config.case_id,
        output_dir=str(config.output_dir),
        dry_run=config.dry_run,
        review_type=config.review_type,
        expected_design=list(config.expected_designs),
        max_study_pages=config.max_study_pages,
        max_chars_per_page=config.max_chars_per_page,
        max_screening_fulltext_chars=config.max_screening_fulltext_chars,
        max_fulltext_adjudication_attempts=config.max_fulltext_adjudication_attempts,
        resume_unsure=config.resume_unsure,
        top_rob_evidence_chunks=8,
        max_rob_fulltext_chars=config.max_rob_fulltext_chars,
    )


def run_candidate_case(config: CandidateCaseConfig) -> dict[str, Any]:
    """Screen supplied candidates and route final Includes to a RoB tool.

    Evaluation labels remain confined to the local output and are not sent to
    either LLM call.
    """

    args = _args_from_config(config)
    if config.dry_run:
        return run_case_screening_rob(args, None, None)
    screening_callback = build_llm_callback(
        model=config.model,
        api_key_env=config.api_key_env,
        base_url=config.base_url,
    )
    routing_callback = build_llm_callback(
        model=config.model,
        api_key_env=config.api_key_env,
        base_url=config.base_url,
        system_message=RiskOfBiasRecommendationAgent.system_message(),
    )
    return run_case_screening_rob(
        args,
        FullTextProtocolScreener(screening_callback),
        RiskOfBiasRecommendationAgent(routing_callback),
    )
