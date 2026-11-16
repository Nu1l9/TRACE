from __future__ import annotations

from .models import PipelineAnalysisAgentResult


def render_graph_report(result: PipelineAnalysisAgentResult) -> str:
    lines = [
        f"# {result.plan.plan_id} Graph-Driven Analysis Report",
        "",
        "This run used graph constraints and embedding retrieval to assemble local candidate sets before scheduling topic review and statistical tools via Planner. No benchmark Gold annotations were read at runtime.",
        "",
        "## Overview",
        "",
        f"- Overall status: `{result.status}`",
        f"- Evidence records: {result.inventory.record_count}; studies: {result.inventory.study_count}",
        f"- Topics: {len(result.analyses)}; completed: {sum(item.status == 'completed' for item in result.analyses)}",
        "",
        "## Retrieval and Memory",
        "",
    ]
    for item in result.retrieval_inventory:
        memory = result.memory.topics[item.analysis_id]
        lines.append(
            f"- `{item.analysis_id}`: candidates {item.candidate_record_count} / "
            f"{item.candidate_context_count} contexts / {item.candidate_study_count} studies; "
            f"final state `{memory.state}`"
        )

    lines.extend(["", "## Analysis Results", ""])
    for item in result.analyses:
        lines.extend([
            f"### {item.analysis.title}",
            "",
            f"- analysis_id: `{item.analysis.analysis_id}`; pooled_result_id: `{item.analysis.pooled_result_id}`",
            f"- Status: `{item.status}`",
            f"- Selected records: {len(item.selected_evidence_ids)}; passed numerical poolability: {len(item.poolability.study_effects)}",
        ])
        if item.statistics:
            lines.extend([
                f"- Pooled effect: {item.statistics.effect_type}={item.statistics.pooled_effect:.6f}",
                f"- 95% CI: {item.statistics.ci_lower:.6f}–{item.statistics.ci_upper:.6f}; k={item.statistics.k}",
            ])
        if item.forest_plot:
            forest = item.forest_plot
            lines.append(f"- Forest plot: `{forest.status}`; {forest.reason}")
            if forest.figure_artifacts:
                lines.append(
                    "- Forest plot files: "
                    f"[SVG]({forest.figure_artifacts.get('forest_plot_svg', '')}); "
                    f"[PNG]({forest.figure_artifacts.get('forest_plot_png', '')})"
                )
            if forest.error:
                lines.append(f"- Forest plot note: {forest.error}")
        if item.reporting_bias:
            assessment = item.reporting_bias
            if item.reporting_bias_projection:
                projection = item.reporting_bias_projection
                lines.extend([
                    f"- Reporting-bias diagnostic projection: `{projection.status}`; {projection.source_effect_count} primary-analysis effects "
                    f"→ {len(projection.selected_effects)} `{projection.effect_type}` effects / "
                    f"{projection.independent_study_count} independent studies",
                    f"- Projection rule: {projection.policy}",
                    f"- Projection selector: `{projection.reviewer_mode}`; {projection.summary}",
                ])
            lines.extend([
                f"- Reporting-bias / small-study diagnostics: `{assessment.status}`; {assessment.interpretation}",
                f"- Diagnostic basis: {assessment.k} effects / {assessment.independent_study_count} independent studies; minimum required={assessment.minimum_studies}",
            ])
            if assessment.status == "completed":
                precision = assessment.precision_diagnostics
                egger = assessment.funnel_asymmetry.get("egger_p_value")
                begg = assessment.funnel_asymmetry.get("begg_p_value")
                lines.append(
                    "- Precision dispersion: "
                    f"SE min={precision.get('standard_error_min')}; "
                    f"SE max={precision.get('standard_error_max')}; "
                    f"max/min={precision.get('standard_error_ratio')}; "
                    f"CV={precision.get('standard_error_cv')}"
                )
                lines.append(f"- Egger p={egger}; Begg p={begg}")
                lines.append(
                    "- Trim-and-fill: "
                    f"imputed={assessment.trim_and_fill.get('imputed_studies')}; "
                    f"original={assessment.trim_and_fill.get('original_pooled_effect')}; "
                    f"adjusted={assessment.trim_and_fill.get('adjusted_pooled_effect')}"
                )
                if assessment.figure_artifacts:
                    lines.append(
                        "- Funnel plots: "
                        f"[Standard funnel]({assessment.figure_artifacts.get('funnel_plot_svg', '')}); "
                        f"[Contour-enhanced funnel]({assessment.figure_artifacts.get('contour_enhanced_funnel_plot_svg', '')})"
                    )
            lines.extend(f"- Bias diagnostic note: {caveat}" for caveat in assessment.caveats)
        if item.robustness:
            robustness = item.robustness
            lines.extend([
                f"- Robustness / influential-study diagnostics: `{robustness.status}`; {robustness.stability}",
                f"- Diagnostic basis: {robustness.k} effects / {robustness.independent_study_count} independent studies; minimum required={robustness.minimum_studies}",
            ])
            if robustness.status == "completed":
                lines.append(
                    "- Model sensitivity: "
                    f"fixed={robustness.model_comparison.get('fixed_effect_pooled_effect')}; "
                    f"random={robustness.model_comparison.get('random_effects_pooled_effect')}"
                )
                if robustness.influential_studies:
                    labels = [
                        f"{record.get('study_id')} (Cook={record.get('cook_distance')})"
                        for record in robustness.influential_studies[:5]
                    ]
                    lines.append(f"- Influential studies requiring review: {'; '.join(labels)}")
                else:
                    lines.append("- Influential studies requiring review: No study exceeded the prespecified alert threshold")
                drivers = [record.get("study_id") for record in robustness.heterogeneity_drivers[:3]]
                if drivers:
                    lines.append(f"- Studies with greatest heterogeneity contribution: {'; '.join(str(driver) for driver in drivers)}")
            lines.extend(f"- Robustness diagnostic note: {caveat}" for caveat in robustness.caveats)
        lines.extend([*(f"- Warning: {warning}" for warning in item.warnings), ""])

    lines.extend(["## Planner Decisions", ""])
    for decision in result.memory.planner_decisions:
        lines.append(f"- `{decision.phase}` / `{decision.planner_mode}`: {decision.summary}")
        for action in decision.actions:
            lines.append(f"  - `{action.action}` → `{action.analysis_id}`: {action.reason}")
    lines.extend(["", "## Tool Trace", ""])
    for entry in result.trace:
        lines.append(f"- `{entry.tool_name}` / `{entry.status}`: {entry.summary}")
    return "\n".join(lines) + "\n"


render_graph_chinese_report = render_graph_report
