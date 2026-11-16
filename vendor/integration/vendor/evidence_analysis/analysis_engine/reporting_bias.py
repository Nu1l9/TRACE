"""Deterministic reporting-bias diagnostics backed by R metafor.

This module assesses funnel-plot asymmetry and sensitivity to missing studies.
It deliberately does not classify a study as low/high risk of bias; study-level
RoB requires a separate evidence-annotation tool.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from analysis_core.models import AnalysisDefinition, MainMetaStatistics, StudyEffect

from .models import ReportingBiasAssessment


class MetaforReportingBiasTool:
    """Run prespecified small-study-effect diagnostics after primary pooling."""

    name = "assess_reporting_bias_metafor"

    def __init__(
        self,
        *,
        minimum_studies: int = 10,
        minimum_standard_error_ratio: float = 1.25,
        minimum_standard_error_cv: float = 0.10,
        rscript_binary: str = "Rscript",
        script_path: Path | None = None,
    ) -> None:
        self.minimum_studies = minimum_studies
        self.minimum_standard_error_ratio = minimum_standard_error_ratio
        self.minimum_standard_error_cv = minimum_standard_error_cv
        self.rscript_binary = rscript_binary
        self.script_path = script_path or Path(__file__).with_name("metafor_reporting_bias.R")

    def assess(
        self,
        analysis: AnalysisDefinition,
        statistics: MainMetaStatistics,
        effects: list[StudyEffect],
        *,
        artifact_dir: Path | None = None,
        artifact_stem: str | None = None,
    ) -> ReportingBiasAssessment:
        effect_types = {effect.effect_type for effect in effects}
        k = len(effects)
        independent_study_count = len({effect.study_id for effect in effects})
        precision_diagnostics = _precision_diagnostics(effects)
        if {effect.effect_scale for effect in effects} != {"log_ratio"}:
            return ReportingBiasAssessment(
                status="not_assessable",
                k=k,
                independent_study_count=independent_study_count,
                minimum_studies=self.minimum_studies,
                effect_type=statistics.effect_type,
                model=statistics.model,
                precision_diagnostics=precision_diagnostics,
                interpretation="not_assessable_non_ratio_effect_scale",
                caveats=[
                    "This reporting-bias implementation currently reports back-transformed ratio effects only."
                ],
            )
        if independent_study_count != k:
            return ReportingBiasAssessment(
                status="not_assessable",
                k=k,
                independent_study_count=independent_study_count,
                minimum_studies=self.minimum_studies,
                effect_type=statistics.effect_type,
                model=statistics.model,
                precision_diagnostics=precision_diagnostics,
                interpretation="not_assessable_nonindependent_effects",
                caveats=[
                    "Funnel-asymmetry diagnostics require independent effect estimates. Resolve within-study dependence with a prespecified selection, RVE, or multilevel model before assessment."
                ],
            )
        if k < self.minimum_studies:
            return ReportingBiasAssessment(
                status="not_assessable",
                k=k,
                independent_study_count=independent_study_count,
                minimum_studies=self.minimum_studies,
                effect_type=statistics.effect_type,
                model=statistics.model,
                precision_diagnostics=precision_diagnostics,
                interpretation="not_assessable_insufficient_studies",
                caveats=[
                    f"Funnel-asymmetry and selection diagnostics require at least {self.minimum_studies} independent studies; k={k}."
                ],
            )
        if len(effect_types) != 1:
            return ReportingBiasAssessment(
                status="incompatible_effect_set",
                k=k,
                independent_study_count=independent_study_count,
                minimum_studies=self.minimum_studies,
                precision_diagnostics=precision_diagnostics,
                interpretation="not_assessable_mixed_effect_types",
                caveats=[
                    "Publication-bias diagnostics are not run on a mixed HR/RR/OR effect set. Use a prespecified same-effect-type sensitivity analysis."
                ],
            )
        if _has_similar_precision(
            precision_diagnostics,
            minimum_standard_error_ratio=self.minimum_standard_error_ratio,
            minimum_standard_error_cv=self.minimum_standard_error_cv,
        ):
            return ReportingBiasAssessment(
                status="not_assessable",
                k=k,
                independent_study_count=independent_study_count,
                minimum_studies=self.minimum_studies,
                effect_type=next(iter(effect_types)),
                model=statistics.model,
                precision_diagnostics=precision_diagnostics,
                interpretation="not_assessable_similar_precision",
                caveats=[
                    "Funnel-asymmetry tests are not run when study standard errors are too similar. "
                    f"Prespecified gate: SE max/min must be at least {self.minimum_standard_error_ratio:.2f} "
                    f"and SE CV must be at least {self.minimum_standard_error_cv:.2f}.",
                ],
            )
        if shutil.which(self.rscript_binary) is None:
            return ReportingBiasAssessment(
                status="unavailable",
                k=k,
                independent_study_count=independent_study_count,
                minimum_studies=self.minimum_studies,
                effect_type=next(iter(effect_types)),
                model=statistics.model,
                precision_diagnostics=precision_diagnostics,
                interpretation="not_run_rscript_unavailable",
                caveats=["Rscript was not found on PATH; metafor diagnostics were not executed."],
            )
        assessment = self._run_metafor(
            analysis,
            statistics,
            effects,
            artifact_dir=artifact_dir,
            artifact_stem=artifact_stem or analysis.analysis_id,
        )
        assessment.precision_diagnostics = precision_diagnostics
        assessment.funnel_plot_points = _funnel_plot_points(effects)
        return assessment

    def _run_metafor(
        self,
        analysis: AnalysisDefinition,
        statistics: MainMetaStatistics,
        effects: list[StudyEffect],
        *,
        artifact_dir: Path | None,
        artifact_stem: str,
    ) -> ReportingBiasAssessment:
        effect_type = effects[0].effect_type
        with tempfile.TemporaryDirectory(prefix="metafor_reporting_bias_") as directory:
            workdir = Path(directory)
            input_path = workdir / "effects.csv"
            output_path = workdir / "result.json"
            standard_funnel_path = None
            contour_funnel_path = None
            figure_artifacts: dict[str, str] = {}
            if artifact_dir is not None:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                standard_funnel_path = artifact_dir / f"{artifact_stem}_funnel.svg"
                contour_funnel_path = artifact_dir / f"{artifact_stem}_contour_funnel.svg"
                figure_artifacts = {
                    "funnel_plot_svg": str(standard_funnel_path),
                    "contour_enhanced_funnel_plot_svg": str(contour_funnel_path),
                }
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
            command = [
                self.rscript_binary,
                str(self.script_path),
                str(input_path),
                str(output_path),
                statistics.model,
                effect_type,
            ]
            if standard_funnel_path is not None and contour_funnel_path is not None:
                command.extend([str(standard_funnel_path), str(contour_funnel_path)])
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            if completed.returncode != 0 or not output_path.exists():
                detail = (completed.stderr or completed.stdout or "metafor did not produce an output file").strip()
                return ReportingBiasAssessment(
                    status="failed",
                    k=len(effects),
                    independent_study_count=len({effect.study_id for effect in effects}),
                    minimum_studies=self.minimum_studies,
                    effect_type=effect_type,
                    model=statistics.model,
                    interpretation="metafor_execution_failed",
                    caveats=["Reporting-bias diagnostics were not used to alter the primary pooled result."],
                    error=detail[:1000],
                )
            raw = json.loads(output_path.read_text(encoding="utf-8"))
        existing_artifacts = {
            name: (
                str(Path(path).relative_to(artifact_dir.parent))
                if artifact_dir is not None else path
            )
            for name, path in figure_artifacts.items()
            if Path(path).exists()
        }
        return _assessment_from_metafor(
            raw,
            statistics,
            self.minimum_studies,
            figure_artifacts=existing_artifacts,
        )


def _as_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _funnel_plot_points(effects: list[StudyEffect]) -> list[dict[str, float | str]]:
    """Return the exact independent points passed to metafor diagnostics."""
    return [
        {
            "study_id": effect.study_id,
            "evidence_id": effect.evidence_id,
            "log_effect": effect.log_effect,
            "standard_error": effect.standard_error,
            "precision": 1 / effect.standard_error,
        }
        for effect in effects
        if effect.standard_error > 0 and math.isfinite(effect.standard_error)
    ]


def _precision_diagnostics(effects: list[StudyEffect]) -> dict[str, float | int | str | None]:
    standard_errors = sorted(
        effect.standard_error
        for effect in effects
        if effect.standard_error > 0 and math.isfinite(effect.standard_error)
    )
    if not standard_errors:
        return {"valid_standard_error_count": 0, "standard_error_min": None, "standard_error_max": None,
                "standard_error_ratio": None, "standard_error_cv": None}
    mean = sum(standard_errors) / len(standard_errors)
    variance = sum((value - mean) ** 2 for value in standard_errors) / len(standard_errors)
    return {
        "valid_standard_error_count": len(standard_errors),
        "standard_error_min": standard_errors[0],
        "standard_error_max": standard_errors[-1],
        "standard_error_ratio": standard_errors[-1] / standard_errors[0],
        "standard_error_cv": math.sqrt(variance) / mean if mean > 0 else None,
    }


def _has_similar_precision(
    diagnostics: dict[str, float | int | str | None],
    *,
    minimum_standard_error_ratio: float,
    minimum_standard_error_cv: float,
) -> bool:
    ratio = _as_float(diagnostics.get("standard_error_ratio"))
    coefficient_of_variation = _as_float(diagnostics.get("standard_error_cv"))
    if ratio is None or coefficient_of_variation is None:
        return True
    return ratio < minimum_standard_error_ratio or coefficient_of_variation < minimum_standard_error_cv


def _assessment_from_metafor(
    raw: dict[str, object],
    statistics: MainMetaStatistics,
    minimum_studies: int,
    *,
    figure_artifacts: dict[str, str] | None = None,
) -> ReportingBiasAssessment:
    regtest = raw.get("regtest") if isinstance(raw.get("regtest"), dict) else {}
    ranktest = raw.get("ranktest") if isinstance(raw.get("ranktest"), dict) else {}
    trimfill = raw.get("trimfill") if isinstance(raw.get("trimfill"), dict) else {}
    unadjusted = raw.get("unadjusted") if isinstance(raw.get("unadjusted"), dict) else {}
    regtest_p = _as_float(regtest.get("p_value"))
    ranktest_p = _as_float(ranktest.get("p_value"))
    unadjusted_log = _as_float(unadjusted.get("pooled_log_effect"))
    unadjusted_lower_log = _as_float(unadjusted.get("ci_lower_log"))
    unadjusted_upper_log = _as_float(unadjusted.get("ci_upper_log"))
    adjusted_log = _as_float(trimfill.get("adjusted_log_effect"))
    adjusted_lower_log = _as_float(trimfill.get("adjusted_ci_lower_log"))
    adjusted_upper_log = _as_float(trimfill.get("adjusted_ci_upper_log"))
    imputed = _as_float(trimfill.get("imputed_studies"))
    # With k0=0, some metafor versions leave the adjusted coefficient unset.
    # The adjusted fit is then exactly the unadjusted *diagnostic projection*,
    # which can differ from the full primary-analysis effect set.
    if imputed == 0 and adjusted_log is None:
        adjusted_log = unadjusted_log
    if imputed == 0 and adjusted_lower_log is None:
        adjusted_lower_log = unadjusted_lower_log
    if imputed == 0 and adjusted_upper_log is None:
        adjusted_upper_log = unadjusted_upper_log
    possible_small_study_effect = any(
        p_value is not None and p_value < 0.10 for p_value in (regtest_p, ranktest_p)
    )
    interpretation = (
        "possible_small_study_effect" if possible_small_study_effect else "no_detected_funnel_asymmetry"
    )
    return ReportingBiasAssessment(
        status="completed",
        k=int(raw.get("k") or statistics.k),
        independent_study_count=int(raw.get("independent_study_count") or statistics.k),
        minimum_studies=minimum_studies,
        effect_type=str(raw.get("effect_type") or statistics.effect_type),
        model=str(raw.get("model") or statistics.model),
        methods_run=["regtest", "ranktest", "trimfill"],
        figure_artifacts=figure_artifacts or {},
        funnel_asymmetry={
            "egger_p_value": regtest_p,
            "begg_p_value": ranktest_p,
        },
        trim_and_fill={
            "imputed_studies": int(imputed) if imputed is not None else None,
            "side": trimfill.get("side"),
            "primary_analysis_pooled_effect": statistics.pooled_effect,
            "original_pooled_effect": math.exp(unadjusted_log) if unadjusted_log is not None else None,
            "adjusted_pooled_effect": math.exp(adjusted_log) if adjusted_log is not None else None,
            "adjusted_ci_lower": math.exp(adjusted_lower_log) if adjusted_lower_log is not None else None,
            "adjusted_ci_upper": math.exp(adjusted_upper_log) if adjusted_upper_log is not None else None,
        },
        interpretation=interpretation,
        caveats=[
            "Funnel asymmetry and small-study effects are not specific evidence of publication bias.",
            "This diagnostic is a sensitivity analysis and does not replace study-level risk-of-bias assessment.",
        ],
    )
