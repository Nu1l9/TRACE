"""Standalone forest-plot renderer for completed meta-analysis topics.

This presentation-only tool receives effects that have already passed
selection, dependence and poolability checks plus already-computed statistics.
It never changes the selected set or any statistical estimate.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

from analysis_core.models import AnalysisDefinition, MainMetaStatistics, StudyEffect

from .models import ForestPlotArtifact


class ForestPlotRenderer:
    """Render study estimates and a pooled summary diamond."""

    name = "render_forest_plot"

    def render(
        self,
        analysis: AnalysisDefinition,
        statistics: MainMetaStatistics,
        effects: list[StudyEffect],
        *,
        artifact_dir: Path | None,
        artifact_stem: str | None = None,
    ) -> ForestPlotArtifact:
        independent_studies = len({effect.study_id for effect in effects})
        if artifact_dir is None:
            return ForestPlotArtifact(
                status="skipped", k=len(effects), independent_study_count=independent_studies,
                effect_type=statistics.effect_type, reason="artifact_directory_not_configured",
            )
        if not effects:
            return ForestPlotArtifact(
                status="not_assessable", k=0, independent_study_count=0,
                effect_type=statistics.effect_type, reason="no_poolable_effects",
            )
        try:
            return self._render(
                analysis, statistics, effects, artifact_dir=artifact_dir,
                artifact_stem=artifact_stem or analysis.analysis_id,
            )
        except Exception as exc:  # A visual artifact must never interrupt analysis.
            return ForestPlotArtifact(
                status="failed", k=len(effects), independent_study_count=independent_studies,
                effect_type=statistics.effect_type, reason="rendering_failed", error=str(exc)[:1000],
            )

    def _render(
        self,
        analysis: AnalysisDefinition,
        statistics: MainMetaStatistics,
        effects: list[StudyEffect],
        *,
        artifact_dir: Path,
        artifact_stem: str,
    ) -> ForestPlotArtifact:
        cache_dir = Path(tempfile.gettempdir()) / "meta_agent_matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        ratio_scale = {effect.effect_scale for effect in effects} == {"log_ratio"}
        adjusted_points = 0
        if ratio_scale:
            rows = [
                # ``StudyEffect`` retains log_effect for the calculator, but
                # its public value and confidence bounds are ratio-scale
                # values. This is the same scale shown in source papers.
                (effect.study_id, effect.value, effect.ci_lower, effect.ci_upper)
                for effect in effects
            ]
            # StudyEffect stores yi/CI in log space for computation, whereas
            # MainMetaStatistics is the calculator's back-transformed public
            # estimate (RR/OR/HR/IRR). Do not exponentiate it a second time.
            pooled, pooled_lower, pooled_upper = (
                statistics.pooled_effect, statistics.ci_lower, statistics.ci_upper
            )
            reference, x_label = 1.0, f"{statistics.effect_type} (log scale)"
        else:
            rows = []
            for effect in effects:
                point, adjusted = _display_difference_scale_point(effect)
                adjusted_points += adjusted
                rows.append((effect.study_id, point, effect.ci_lower, effect.ci_upper))
            pooled, pooled_lower, pooled_upper = statistics.pooled_effect, statistics.ci_lower, statistics.ci_upper
            reference, x_label = 0.0, statistics.effect_type

        artifact_dir.mkdir(parents=True, exist_ok=True)
        svg_path = artifact_dir / f"{artifact_stem}_forest.svg"
        png_path = artifact_dir / f"{artifact_stem}_forest.png"
        figure, axis = plt.subplots(figsize=(10.5, max(3.2, 1.8 + 0.40 * (len(rows) + 1))), constrained_layout=True)
        try:
            y_positions = list(range(len(rows), 0, -1))
            values = [row[1] for row in rows]
            lower = [row[2] for row in rows]
            upper = [row[3] for row in rows]
            errors = [[value - low for value, low in zip(values, lower)], [high - value for value, high in zip(values, upper)]]
            axis.errorbar(values, y_positions, xerr=errors, fmt="s", color="#1f4e79", ecolor="#4f6474", capsize=3, markersize=5, zorder=3)
            axis.axvline(reference, color="#6b7280", linewidth=1, linestyle="--", zorder=1)
            if ratio_scale:
                axis.set_xscale("log")
            diamond_y, half_height = 0.15, 0.20
            axis.add_patch(Polygon(
                [(pooled_lower, diamond_y), (pooled, diamond_y + half_height), (pooled_upper, diamond_y), (pooled, diamond_y - half_height)],
                closed=True, facecolor="#b5473c", edgecolor="#7f1d1d", zorder=4,
            ))
            axis.set_yticks([*y_positions, diamond_y])
            axis.set_yticklabels([*(row[0] for row in rows), "Pooled"])
            axis.set_ylim(-0.45, len(rows) + 0.8)
            axis.set_xlabel(x_label)
            axis.set_title(analysis.title, loc="left", fontsize=11, fontweight="bold")
            axis.grid(axis="x", color="#d1d5db", linewidth=0.6, zorder=0)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0)
            figure.text(
                0.99, 0.015, f"Pooled: {pooled:.3g} [{pooled_lower:.3g}, {pooled_upper:.3g}], k={statistics.k}",
                ha="right", va="bottom", fontsize=8, color="#374151",
            )
            figure.savefig(svg_path, format="svg", bbox_inches="tight")
            figure.savefig(png_path, dpi=220, bbox_inches="tight")
        finally:
            plt.close(figure)
        return ForestPlotArtifact(
            status="completed", k=len(effects), independent_study_count=len({effect.study_id for effect in effects}),
            effect_type=statistics.effect_type, scale="ratio" if ratio_scale else "difference",
            displayed_pooled_effect=pooled, displayed_ci_lower=pooled_lower, displayed_ci_upper=pooled_upper,
            figure_artifacts={"forest_plot_svg": str(svg_path), "forest_plot_png": str(png_path)},
            reason="rendered_from_completed_poolability_and_statistics",
            notes=(
                [
                    f"{adjusted_points} bounded diagnostic/proportion point estimate(s) were displayed "
                    "on their analysis transform's back-transformed scale because raw source values lay outside their corrected confidence bounds."
                ]
                if adjusted_points else []
            ),
        )


def _display_difference_scale_point(effect: StudyEffect) -> tuple[float, bool]:
    """Return a point compatible with an already back-transformed CI.

    The diagnostic-accuracy tool uses logit estimates and the prevalence tool
    uses a double-arcsine estimate for boundary cells. Their raw source rate
    can legitimately sit outside the continuity-corrected confidence bounds.
    """
    if effect.ci_lower <= effect.value <= effect.ci_upper:
        return effect.value, False
    if effect.effect_type in {"sensitivity", "specificity"}:
        return 1.0 / (1.0 + math.exp(-effect.log_effect)), True
    if effect.effect_type == "proportion":
        return math.sin(effect.log_effect / 2.0) ** 2, True
    return min(max(effect.value, effect.ci_lower), effect.ci_upper), True
