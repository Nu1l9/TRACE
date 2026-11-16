"""Adapter from the public end-to-end Evidence contract to the analysis core.

This module is deliberately the only schema boundary for upstream screening
and extraction components.  It reads no benchmark files, Gold labels, target
reviews, or evaluator-side metadata.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from analysis_engine.benchmark_case_adapter import (
    _candidate_analysis_metadata,
    _candidate_outcome,
    _hydrate_method_neutral_structured_profile,
    _pico_component,
)


def _analysis_effect_type(case_id: str, effect: dict[str, Any], raw_data: dict[str, Any]) -> Any:
    """Apply the ma_0012-only adjusted-OR notation repair at analysis ingestion."""
    reported = effect.get("effect_type")
    if case_id == "ma_0012" and str(reported or "").strip().casefold() in {
        "aor", "adjusted or", "adjusted odds ratio",
    }:
        raw_data["_source_effect_type"] = reported
        return "OR"
    return reported


REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id",
    "article_id",
    "study_id",
    "target_id",
    "outcome",
    "data_family",
    "raw_data",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required analysis input: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return payload


def _resolve_input_path(input_dir: Path, filename: str) -> Path:
    """Accept either a flat integration folder or a benchmark-style inputs/ folder."""
    direct = input_dir / filename
    nested = input_dir / "inputs" / filename
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    return direct


def _task_path(input_dir: Path) -> Path:
    candidates = (
        input_dir / "analysis_task.json",
        input_dir / "tasks" / "analysis_main_outcomes.json",
        input_dir / "inputs" / "analysis_task.json",
        input_dir / "tasks" / "analysis_main_outcomes.json",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def validate_candidates(candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        raise ValueError("analysis_evidence_candidates.json contains no candidates.")
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate {index} must be a JSON object.")
        missing = [field for field in REQUIRED_CANDIDATE_FIELDS if field not in candidate]
        if missing:
            raise ValueError(f"Candidate {index} is missing required fields: {', '.join(missing)}.")
        candidate_id = str(candidate["candidate_id"])
        if not candidate_id:
            raise ValueError(f"Candidate {index} has an empty candidate_id.")
        if candidate_id in seen_ids:
            raise ValueError(f"candidate_id must be unique; duplicate: {candidate_id!r}.")
        seen_ids.add(candidate_id)
        if not isinstance(candidate["raw_data"], dict):
            raise ValueError(f"Candidate {candidate_id!r} raw_data must be an object.")
        if not isinstance(candidate["outcome"], (str, dict)):
            raise ValueError(f"Candidate {candidate_id!r} outcome must be a string or object.")
        if candidate.get("effect") is not None and not isinstance(candidate["effect"], dict):
            raise ValueError(f"Candidate {candidate_id!r} effect must be an object when supplied.")


def load_analysis_contract(
    input_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Load a Gold-free contract and create the Evidence payloads used by GraphRAG."""
    input_dir = input_dir.expanduser().resolve()
    protocol = _read_json(_resolve_input_path(input_dir, "extraction_protocol.json"))
    task = _read_json(_task_path(input_dir))
    profile = _read_json(_resolve_input_path(input_dir, "analysis_profile_structured.json"))
    candidate_payload = _read_json(_resolve_input_path(input_dir, "analysis_evidence_candidates.json"))
    candidates = candidate_payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("analysis_evidence_candidates.json must contain a candidates array.")
    validate_candidates(candidates)
    if not str(profile.get("profile_version", "")).startswith("2"):
        raise ValueError("analysis_profile_structured.json requires profile_version beginning with '2'.")
    if not task.get("task_id"):
        raise ValueError("analysis_task.json requires task_id.")

    # A method-neutral public profile gets a transient calculator route here.
    # It is not exposed to the reviewer and never changes the input files.
    if profile.get("method_selection_policy") == "prespecified":
        runtime_profile = deepcopy(profile)
        for spec in [runtime_profile.get("primary_analysis_spec", {}), *runtime_profile.get("secondary_outcomes", [])]:
            pooling = spec.get("pooling") or {}
            if not pooling.get("model") or not pooling.get("scale"):
                raise ValueError("A prespecified analysis requires a pooling model and scale for every topic.")
            if spec.get("dependence_policy") == "clustered_effects_rho_0_6":
                if (pooling.get("model") != "random_effects"
                        or pooling.get("heterogeneity_estimator") != "reml"
                        or pooling["scale"] != "log_ratio_inverse_variance__clustered_rho_0_6"):
                    raise ValueError("Clustered effects require the declared REML rho=0.6 random-effects route.")
                scale = pooling["scale"]
            else:
                scale = str(pooling["scale"]).split("__", 1)[0]
                estimator = str(pooling.get("heterogeneity_estimator") or "").casefold().replace("-", "_")
                if estimator in {"reml", "paule_mandel"}:
                    scale += f"__{estimator}"
                interval = str(pooling.get("confidence_interval") or "").casefold().replace("-", "_")
                if interval in {"hartung_knapp", "hartung_knapp_t", "knapp_hartung", "knha"}:
                    scale += "__hartung_knapp_t"
            pooling["scale"] = scale
    else:
        runtime_profile = _hydrate_method_neutral_structured_profile(profile)
    exposure = _pico_component(protocol, "intervention_or_exposure", "intervention", "exposure")
    exposure_fallback = str(exposure.get("description") or "")
    by_article: dict[str, list[dict[str, Any]]] = {}
    study_by_article: dict[str, str] = {}
    for candidate in candidates:
        article_id = str(candidate["article_id"])
        study_id = str(candidate["study_id"])
        if article_id in study_by_article and study_by_article[article_id] != study_id:
            raise ValueError(f"article_id {article_id!r} maps to more than one study_id.")
        outcome = _candidate_outcome(candidate)
        effect = dict(candidate.get("effect") or {})
        raw_data = dict(candidate["raw_data"])
        reported_effect_type = _analysis_effect_type(str(protocol.get("case_id") or ""), effect, raw_data)
        metadata = _candidate_analysis_metadata(candidate)
        if metadata:
            raw_data["_analysis_metadata"] = metadata
        by_article.setdefault(article_id, []).append({
            "evidence_id": str(candidate["candidate_id"]),
            "article_id": article_id,
            "study_id": study_id,
            "target_id": candidate.get("target_id"),
            "intervention_or_exposure": (
                candidate.get("exposure")
                or candidate.get("intervention_or_exposure")
                or exposure_fallback
            ),
            "comparator": candidate.get("comparison") or "",
            "outcome": outcome["label"],
            "outcome_code": outcome["code"],
            "population_or_subgroup": candidate.get("population_or_subgroup") or "",
            "follow_up": candidate.get("timepoint") or "",
            "data_family": candidate.get("data_family") or "",
            "estimand": candidate.get("estimand") or "",
            "raw_data": raw_data,
            "reported_effect_type": reported_effect_type,
            "reported_effect_value": effect.get("value"),
            "ci_lower": effect.get("ci_lower"),
            "ci_upper": effect.get("ci_upper"),
            "source_location": candidate.get("source_location") or {},
        })
        study_by_article[article_id] = study_id
    extractor_payloads = [
        {"article_id": article_id, "study_id": study_by_article[article_id], "evidence": evidence}
        for article_id, evidence in sorted(by_article.items())
    ]
    return protocol, task, runtime_profile, extractor_payloads, study_by_article
