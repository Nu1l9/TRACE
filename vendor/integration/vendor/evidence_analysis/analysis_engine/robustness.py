"""Deterministic post-pooling robustness diagnostics backed by R metafor."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from analysis_core.models import AnalysisDefinition, MainMetaStatistics, StudyEffect

from .models import RobustnessDiagnostics


class MetaforRobustnessDiagnosticsTool:
    """Assess influence, leave-one-out stability, and model dependence.

    The tool is intentionally observational: it returns diagnostics only. The
    analysis planner or a human can subsequently schedule a prespecified
    sensitivity analysis, but no study is silently excluded here.
    """

    name = "assess_robustness_metafor"

    def __init__(
        self,
        *,
        minimum_studies: int = 3,
        rscript_binary: str = "Rscript",
        script_path: Path | None = None,
    ) -> None:
        self.minimum_studies = minimum_studies
        self.rscript_binary = rscript_binary
        self.script_path = script_path or Path(__file__).with_name("metafor_robustness.R")

    def assess(
        self,
        analysis: AnalysisDefinition,
        statistics: MainMetaStatistics,
        effects: list[StudyEffect],
    ) -> RobustnessDiagnostics:
        k = len(effects)
        independent_study_count = len({effect.study_id for effect in effects})
        if "__clustered" in analysis.pooling_scale or (
            statistics.model == "fixed_effect" and "mantel_haenszel" in analysis.pooling_scale
        ):
            return RobustnessDiagnostics(
                status="not_assessable", k=k, independent_study_count=independent_study_count,
                minimum_studies=self.minimum_studies, effect_type=statistics.effect_type,
                primary_model=statistics.model, stability="unsupported_primary_calculator",
                caveats=["These univariate inverse-variance diagnostics cannot reproduce the declared clustered or Mantel-Haenszel main tool; no substitute model was fitted."],
            )
        if {effect.effect_scale for effect in effects} != {"log_ratio"}:
            return RobustnessDiagnostics(
                status="not_assessable",
                k=k,
                independent_study_count=independent_study_count,
                minimum_studies=self.minimum_studies,
                effect_type=statistics.effect_type,
                primary_model=statistics.model,
                stability="not_assessable_non_ratio_effect_scale",
                caveats=[
                    "This robustness implementation currently reports back-transformed ratio effects only."
                ],
            )
        if independent_study_count != k:
            return RobustnessDiagnostics(
                status="not_assessable",
                k=k,
                independent_study_count=independent_study_count,
                minimum_studies=self.minimum_studies,
                effect_type=statistics.effect_type,
                primary_model=statistics.model,
                stability="not_assessable_nonindependent_effects",
                caveats=[
                    "Leave-one-out and influence diagnostics require independent effect estimates. Resolve within-study dependence with a prespecified selection, RVE, or multilevel model before assessment."
                ],
            )
        if k < self.minimum_studies:
            return RobustnessDiagnostics(
                status="not_assessable",
                k=k,
                independent_study_count=independent_study_count,
                minimum_studies=self.minimum_studies,
                effect_type=statistics.effect_type,
                primary_model=statistics.model,
                stability="not_assessable_insufficient_studies",
                caveats=[
                    f"Leave-one-out and influence diagnostics require at least {self.minimum_studies} independent studies; k={k}."
                ],
            )
        if shutil.which(self.rscript_binary) is None:
            return RobustnessDiagnostics(
                status="unavailable",
                k=k,
                independent_study_count=independent_study_count,
                minimum_studies=self.minimum_studies,
                effect_type=statistics.effect_type,
                primary_model=statistics.model,
                stability="not_run_rscript_unavailable",
                caveats=["Rscript was not found on PATH; metafor diagnostics were not executed."],
            )
        return self._run_metafor(analysis, statistics, effects)

    def _run_metafor(
        self,
        analysis: AnalysisDefinition,
        statistics: MainMetaStatistics,
        effects: list[StudyEffect],
    ) -> RobustnessDiagnostics:
        with tempfile.TemporaryDirectory(prefix="metafor_robustness_") as directory:
            workdir = Path(directory)
            input_path = workdir / "effects.csv"
            output_path = workdir / "result.json"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["study_id", "evidence_id", "yi", "vi"])
                writer.writeheader()
                for effect in effects:
                    writer.writerow({
                        "study_id": effect.study_id,
                        "evidence_id": effect.evidence_id,
                        "yi": effect.log_effect,
                        "vi": effect.standard_error**2,
                    })
            from extensions.r_runtime import run_rscript
            try:
                completed = run_rscript(
                    [
                        self.rscript_binary,
                        str(self.script_path),
                        str(input_path),
                        str(output_path),
                        statistics.model,
                        effects[0].effect_type,
                        analysis.pooling_scale,
                    ],
                    purpose="robustness diagnostics",
                )
            except RuntimeError as exc:
                return RobustnessDiagnostics(
                    status="failed",
                    k=len(effects),
                    independent_study_count=len({effect.study_id for effect in effects}),
                    minimum_studies=self.minimum_studies,
                    effect_type=effects[0].effect_type,
                    primary_model=statistics.model,
                    stability="not_assessable_execution_timeout",
                    caveats=[
                        "Robustness diagnostics timed out and were not used to alter the completed primary pooled result."
                    ],
                    error=str(exc),
                )
            if completed.returncode != 0 or not output_path.exists():
                detail = (completed.stderr or completed.stdout or "metafor did not produce an output file").strip()
                return RobustnessDiagnostics(
                    status="failed",
                    k=len(effects),
                    independent_study_count=len({effect.study_id for effect in effects}),
                    minimum_studies=self.minimum_studies,
                    effect_type=effects[0].effect_type,
                    primary_model=statistics.model,
                    stability="not_assessable_execution_failed",
                    caveats=["Robustness diagnostics were not used to alter the primary pooled result."],
                    error=detail[:1000],
                )
            raw = json.loads(output_path.read_text(encoding="utf-8"))
        result = _diagnostics_from_metafor(raw, statistics, self.minimum_studies)
        result.caveats.append(
            f"Primary and leave-one-out fits use {raw['primary_method']}/{raw['primary_ci']}; "
            "the separate fixed-versus-random comparison uses FE/normal versus DL/normal."
        )
        return result


def _as_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _records(value: object) -> list[dict[str, object]]:
    return [record for record in value if isinstance(record, dict)] if isinstance(value, list) else []


def _diagnostics_from_metafor(
    raw: dict[str, object],
    statistics: MainMetaStatistics,
    minimum_studies: int,
) -> RobustnessDiagnostics:
    leave_one_out = _records(raw.get("leave_one_out"))
    influence = _records(raw.get("influence"))
    baujat = _records(raw.get("baujat"))
    model_fits = raw.get("model_fits") if isinstance(raw.get("model_fits"), dict) else {}
    primary_log_effect = math.log(statistics.pooled_effect)
    primary_ci_excludes_null = statistics.ci_lower > 1 or statistics.ci_upper < 1
    directional_reversal = False
    inferential_status_changes_after_removal = False
    for record in leave_one_out:
        log_effect = _as_float(record.get("pooled_log_effect"))
        lower = _as_float(record.get("ci_lower_log"))
        upper = _as_float(record.get("ci_upper_log"))
        if log_effect is not None and primary_log_effect * log_effect < 0:
            directional_reversal = True
        if lower is not None and upper is not None:
            leave_one_out_ci_excludes_null = lower > 0 or upper < 0
            # A leave-one-out run can either lose or gain conventional
            # statistical significance. Both changes mean inference depends
            # materially on one study, regardless of the primary result.
            if leave_one_out_ci_excludes_null != primary_ci_excludes_null:
                inferential_status_changes_after_removal = True

    cook_threshold = 4 / max(statistics.k, 1)
    influential = []
    for record in influence:
        cook = _as_float(record.get("cook_distance"))
        residual = _as_float(record.get("studentized_residual"))
        if (cook is not None and cook > cook_threshold) or (residual is not None and abs(residual) > 1.96):
            influential.append({
                "study_id": record.get("study_id"),
                "evidence_id": record.get("evidence_id"),
                "cook_distance": cook,
                "studentized_residual": residual,
            })
    influential.sort(key=lambda record: record.get("cook_distance") or 0, reverse=True)

    heterogeneity_drivers = sorted(
        (
            {
                "study_id": record.get("study_id"),
                "evidence_id": record.get("evidence_id"),
                "heterogeneity_contribution": _as_float(record.get("heterogeneity_contribution")),
                "influence_on_pooled_effect": _as_float(record.get("influence_on_pooled_effect")),
            }
            for record in baujat
        ),
        key=lambda record: record.get("heterogeneity_contribution") or 0,
        reverse=True,
    )[:5]

    if directional_reversal or inferential_status_changes_after_removal:
        stability = "potentially_fragile_under_leave_one_out"
    else:
        stability = "no_directional_or_inferential_reversal_observed"
    comparison = {
        "primary_pooled_effect": statistics.pooled_effect,
        "fixed_effect_pooled_effect": _exp_number(model_fits, "fixed_effects", "pooled_log_effect"),
        "random_effects_pooled_effect": _exp_number(model_fits, "random_effects", "pooled_log_effect"),
        "fixed_effects_i_squared": _nested_number(model_fits, "fixed_effects", "i_squared"),
        "random_effects_i_squared": _nested_number(model_fits, "random_effects", "i_squared"),
    }
    return RobustnessDiagnostics(
        status="completed",
        k=int(raw.get("k") or statistics.k),
        independent_study_count=int(raw.get("independent_study_count") or statistics.k),
        minimum_studies=minimum_studies,
        effect_type=str(raw.get("effect_type") or statistics.effect_type),
        primary_model=statistics.model,
        methods_run=["leave1out", "influence", "baujat", "fixed_vs_random"],
        model_comparison=comparison,
        leave_one_out=[_normalise_record(record) for record in leave_one_out],
        influential_studies=influential,
        heterogeneity_drivers=heterogeneity_drivers,
        stability=stability,
        caveats=[
            "Influence thresholds flag studies for review; they are not automatic exclusion rules.",
            "Leave-one-out and fixed-versus-random comparisons are robustness diagnostics, not replacements for a prespecified sensitivity-analysis protocol.",
        ],
    )


def _nested_number(mapping: dict[str, object], key: str, field: str) -> float | None:
    nested = mapping.get(key)
    return _as_float(nested.get(field)) if isinstance(nested, dict) else None


def _exp_number(mapping: dict[str, object], key: str, field: str) -> float | None:
    value = _nested_number(mapping, key, field)
    return math.exp(value) if value is not None else None


def _normalise_record(record: dict[str, object]) -> dict[str, float | str | None]:
    normalised: dict[str, float | str | None] = {}
    for key, value in record.items():
        number = _as_float(value)
        normalised[key] = number if number is not None else str(value) if value is not None else None
    return normalised
