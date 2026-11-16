"""Validated response contracts for LLM-backed review stages.

Screening returns only an eligibility decision and rationale. Routing evidence
quotes are checked against the supplied source text by the caller.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
T = TypeVar("T", bound=BaseModel)


class LLMResponseModel(BaseModel):
    """Base contract shared by all model responses in this package."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class AbstractScreeningResponse(LLMResponseModel):
    status: Literal["Include", "Exclude", "Unsure"]
    reason: NonEmptyString


class FullTextScreeningResponse(LLMResponseModel):
    status: Literal["Include", "Exclude"]
    reason: NonEmptyString


QuestionType = Literal[
    "intervention_effect",
    "exposure_effect",
    "diagnostic_accuracy",
    "prognostic_factor",
    "prediction_model",
    "other_or_unclear",
]

StudyDesign = Literal[
    "individual_rct",
    "cluster_rct",
    "crossover_rct",
    "nonrandomized_intervention",
    "cohort",
    "case_control",
    "cross_sectional",
    "diagnostic_accuracy",
    "prognostic_factor_study",
    "prediction_model_study",
    "other",
    "unclear",
]

AllocationType = Literal["randomized", "nonrandomized", "not_applicable", "unclear"]
ConfidenceLevel = Literal["high", "medium", "low"]
RiskOfBiasTool = Literal[
    "RoB 2",
    "ROBINS-I",
    "ROBINS-E",
    "QUADAS-3",
    "QUIPS",
    "PROBAST+AI",
]


class RiskOfBiasEvidence(LLMResponseModel):
    evidence_id: int = Field(ge=1)
    claim: Literal["study_design", "question_type", "allocation", "tool_rationale"]
    quote: NonEmptyString


class RiskOfBiasRoutingResponse(LLMResponseModel):
    question_type: QuestionType
    study_design: StudyDesign
    allocation: AllocationType
    recommended_tool: RiskOfBiasTool | None
    confidence: ConfidenceLevel
    reason: NonEmptyString
    evidence: list[RiskOfBiasEvidence]
    missing_information: list[NonEmptyString]


def parse_llm_json(response: str, model: type[T]) -> T:
    """Extract and validate exactly one JSON object from a model response.

    Some OpenAI-compatible providers occasionally wrap otherwise valid JSON in a
    Markdown fence. The decoder accepts that wrapper but rejects a second object
    or any non-whitespace text after the object.
    """

    if not isinstance(response, str):
        raise TypeError("LLM response must be a string.")
    payload = response.strip()
    if payload.startswith("```"):
        first_newline = payload.find("\n")
        if first_newline < 0:
            raise ValueError("Markdown fence did not contain JSON.")
        payload = payload[first_newline + 1 :]
        if payload.rstrip().endswith("```"):
            payload = payload.rstrip()[:-3].rstrip()

    decoder = json.JSONDecoder()
    start = payload.find("{")
    if start < 0:
        raise ValueError("LLM response did not contain a JSON object.")
    raw_object, end = decoder.raw_decode(payload[start:])
    if payload[start + end :].strip():
        raise ValueError("LLM response contained text after the JSON object.")
    if not isinstance(raw_object, dict):
        raise ValueError("LLM response JSON root must be an object.")
    return model.model_validate(raw_object)


__all__ = [
    "AbstractScreeningResponse",
    "FullTextScreeningResponse",
    "RiskOfBiasRoutingResponse",
    "ValidationError",
    "parse_llm_json",
]
