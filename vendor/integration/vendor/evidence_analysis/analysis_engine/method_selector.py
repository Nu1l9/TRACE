"""Auditable statistical-method selection for method-blind benchmark ablations.

The public Format profile intentionally does not prescribe a fixed/random model,
heterogeneity estimator, or confidence-interval method.  This tool makes that
choice after evidence selection, before the deterministic metafor calculation.
"""

from __future__ import annotations

import json
import os
from hashlib import sha256
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from analysis_core.models import AnalysisDefinition, StudyEffect

from .llm_transport import create_chat_completion


class MethodSelectionDecision(BaseModel):
    pooling_model: Literal["fixed_effect", "random_effects"]
    heterogeneity_estimator: Literal["dersimonian_laird", "reml", "paule_mandel", "not_applicable"]
    confidence_interval: Literal["normal_approximation", "hartung_knapp_t"]
    rationale: str
    selector_mode: Literal["deterministic", "llm"] = "llm"


class StatisticalMethodSelector(Protocol):
    mode: str

    def select(
        self, analysis: AnalysisDefinition, effects: list[StudyEffect]
    ) -> MethodSelectionDecision: ...


class DeterministicMethodSelector:
    """Offline fallback for unit tests; production Format uses the LLM selector."""

    mode = "deterministic"

    def select(self, analysis: AnalysisDefinition, effects: list[StudyEffect]) -> MethodSelectionDecision:
        return MethodSelectionDecision(
            pooling_model="random_effects",
            heterogeneity_estimator="reml",
            confidence_interval="normal_approximation",
            rationale="Offline fallback: random-effects REML with normal-approximation interval.",
            selector_mode=self.mode,
        )


class LLMStatisticalMethodSelector:
    """Choose one generic synthesis method without seeing source Gold."""

    mode = "llm"

    def __init__(
        self,
        *,
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        prompt_style: Literal["legacy_legacy", "neutral_neutral"] = "legacy_legacy",
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_retries = max_retries
        self.prompt_style = prompt_style
        self.calls_made = 0

    @staticmethod
    def _rotate(values: list[str], analysis_id: str, salt: str) -> list[str]:
        """Use a stable per-topic order so one option is not always displayed first."""
        offset = int(sha256(f"{analysis_id}:{salt}".encode()).hexdigest(), 16) % len(values)
        return values[offset:] + values[:offset]

    def _prompt_and_payload(
        self, analysis: AnalysisDefinition, effects: list[StudyEffect]
    ) -> tuple[str, dict[str, object]]:
        allowed_methods = {
            "pooling_model": ["fixed_effect", "random_effects"],
            "heterogeneity_estimator": [
                "dersimonian_laird", "reml", "paule_mandel", "not_applicable"
            ],
            "confidence_interval": ["normal_approximation", "hartung_knapp_t"],
        }
        if self.prompt_style == "neutral_neutral":
            allowed_methods = {
                key: self._rotate(values, analysis.analysis_id, key)
                for key, values in allowed_methods.items()
            }
        payload: dict[str, object] = {
            "analysis": {
                "analysis_id": analysis.analysis_id,
                "outcome": analysis.outcome_group,
                "data_family": analysis.eligible_data_families,
                "effect_types": analysis.accepted_effect_types,
                "minimum_studies": analysis.minimum_studies,
            },
            "independent_effects": [
                {
                    "study_id": effect.study_id,
                    "effect_type": effect.effect_type,
                    "estimate": effect.value,
                    "analysis_scale_estimate": effect.log_effect,
                    "standard_error": effect.standard_error,
                    "ci_lower": effect.ci_lower,
                    "ci_upper": effect.ci_upper,
                }
                for effect in effects
            ],
            "allowed_methods": allowed_methods,
        }
        if self.prompt_style == "legacy_legacy":
            prompt = (
                "You are a statistical-method selection tool in a meta-analysis workflow. "
                "The study evidence has already been selected; choose one defensible synthesis method from the "
                "allowed options using only the analysis question and independent effect summaries below. "
                "Do not infer a target review's implementation, do not change the evidence set, and do not calculate "
                "a pooled effect. Fixed-effect requires a substantive common-effect justification. For random-effects, "
                "choose an estimator and interval method suitable for the number of studies and heterogeneity uncertainty. "
                "Use not_applicable only with fixed_effect. Return JSON only with pooling_model, "
                "heterogeneity_estimator, confidence_interval, rationale.\n\n"
                f"INPUT={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
            )
        elif self.prompt_style == "neutral_neutral":
            prompt = (
                "You are a statistical-method selection tool in a meta-analysis workflow. "
                "The study evidence has already been selected. Choose one defensible synthesis configuration from "
                "the declared options using only the analysis question and independent effect summaries below. "
                "Assess fixed-effect and random-effects configurations by the same evidentiary standard; do not begin "
                "with a presumption for either model, estimator, or interval method. The option labels and their display "
                "order are not recommendations or rankings. Do not infer a target review's implementation, do not change "
                "the evidence set, and do not calculate a pooled effect. Use not_applicable only with fixed_effect. "
                "Return JSON only with pooling_model, heterogeneity_estimator, confidence_interval, rationale.\n\n"
                f"INPUT={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
            )
        else:
            raise ValueError(f"Unsupported method-selector prompt style: {self.prompt_style}")
        return prompt, payload

    def select(self, analysis: AnalysisDefinition, effects: list[StudyEffect]) -> MethodSelectionDecision:
        from openai import OpenAI

        if not self.api_key:
            raise RuntimeError("LLM statistical method selector needs an API key.")
        if len(effects) < 2:
            raise ValueError("Statistical method selection requires at least two poolable effects.")

        prompt, _ = self._prompt_and_payload(analysis, effects)
        request = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        self.calls_made += 1
        response = create_chat_completion(
            OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0, timeout=120.0),
            request,
            component=f"MethodSelector/{analysis.analysis_id}",
            max_retries=self.max_retries,
        )
        content = (response.choices[0].message.content or "").removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            raw = MethodSelectionDecision.model_validate_json(content)
        except ValueError as exc:
            raise RuntimeError(f"Method selector returned invalid JSON for {analysis.analysis_id}.") from exc
        if raw.pooling_model == "fixed_effect":
            raw.heterogeneity_estimator = "not_applicable"
            raw.confidence_interval = "normal_approximation"
        elif raw.heterogeneity_estimator == "not_applicable":
            raise RuntimeError("Method selector chose not_applicable estimator for random effects.")
        raw.selector_mode = self.mode
        return raw
