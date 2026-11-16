from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandidateReview(BaseModel):
    """Minimal LLM response: never repeat the candidate's extracted values."""

    model_config = ConfigDict(extra="forbid")

    prediction_id: str
    disposition: Literal[
        "keep_as_valid_candidate",
        "discard_clear_mismatch",
        "needs_further_review",
    ]
    evidence_role: Literal[
        "main_analysis_candidate",
        "supplementary_or_sensitivity",
        "source_only",
        "uncertain",
        "none",
    ] = Field(
        description=(
            "The candidate's role in the evidence pool. A candidate may be retained "
            "without being eligible for the main analysis."
        )
    )
    reason: str = Field(
        max_length=120,
        description="Concise audit reason in English, at most 120 characters. Do not repeat source data."
    )

    @model_validator(mode="after")
    def disposition_must_match_evidence_role(self) -> "CandidateReview":
        if self.disposition == "keep_as_valid_candidate" and self.evidence_role in {
            "main_analysis_candidate",
            "supplementary_or_sensitivity",
            "source_only",
        }:
            return self
        if (
            self.disposition == "needs_further_review"
            and self.evidence_role == "uncertain"
        ):
            return self
        if self.disposition == "discard_clear_mismatch" and self.evidence_role == "none":
            return self
        raise ValueError("disposition must be consistent with evidence_role")


class CandidateReviewBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviews: list[CandidateReview]
