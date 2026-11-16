from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .evidence_graph import EvidenceGraph, load_evidence_records
from .global_router import DeterministicGlobalEvidenceRouter, GlobalEvidenceRouter
from .models import (
    BaseAnalysisAgentResult,
    AnalysisDefinition,
    AnalysisExecutionResult,
    EvidenceAssignmentDecision,
    EvidenceRecord,
    PoolabilitySet,
    ToolTraceEntry,
)
from .plan_resolver import resolve_analysis_plan_structured, runtime_profile_for_analysis
from .statistics import qualify_adjusted_ratio_records, qualify_rate_ratio_records
from .tools import CalculateMainMetaTool


def _duration_months(value: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(months?|weeks?|years?)", value.casefold())
    if not match:
        return 0.0
    amount = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("year"):
        return amount * 12
    if unit.startswith("week"):
        return amount / 4.345
    return amount


def _policy_score(record: EvidenceRecord, analysis: AnalysisDefinition) -> tuple[int, float, int, str]:
    labelled_longest = int("longest eligible" in record.follow_up.casefold())
    duration = _duration_months(record.follow_up)
    complete = sum(
        value is not None
        for value in (record.reported_effect_value, record.ci_lower, record.ci_upper)
    )
    return labelled_longest, duration, complete, record.evidence_id


def _apply_dependence_policy(
    analysis: AnalysisDefinition,
    selected_records: list[EvidenceRecord],
    decisions: list[EvidenceAssignmentDecision],
) -> list[EvidenceRecord]:
    if analysis.dependence_policy == "correlated_effects_not_implemented":
        return selected_records
    grouped: dict[tuple[str, ...], list[EvidenceRecord]] = defaultdict(list)
    for record in selected_records:
        if analysis.dependence_policy == "one_effect_per_study":
            # "One effect per study" means one competing model per *source
            # analysis context*, not one row for an entire publication.  A
            # paper can legitimately contribute non-overlapping age/sex
            # strata or distinct calendar-year cohorts.  Collapsing all of
            # those into a single estimate silently discards valid evidence.
            key = (
                record.study_id,
                record.outcome_code or "unspecified_outcome_code",
                record.outcome.casefold(),
                record.comparator.casefold(),
                record.population_or_subgroup.casefold(),
                record.follow_up.casefold(),
                record.estimand.casefold(),
            )
        else:
            key = (
                record.study_id,
                record.outcome_code or "unspecified_outcome_code",
                record.outcome.casefold(),
                record.comparator.casefold(),
                record.population_or_subgroup.casefold(),
                record.follow_up.casefold(),
                record.estimand.casefold(),
            )
        grouped[key].append(record)

    retained: list[EvidenceRecord] = []
    dropped: set[str] = set()
    for records in grouped.values():
        preferred = max(records, key=lambda record: _policy_score(record, analysis))
        retained.append(preferred)
        dropped.update(record.evidence_id for record in records if record.evidence_id != preferred.evidence_id)
    for decision in decisions:
        if decision.analysis_id == analysis.analysis_id and decision.evidence_id in dropped:
            decision.role = "excluded"
            decision.selected_within_context = False
            decision.reason = (
                "Eligible source candidate, but another record was preferred after the analysis-level "
                "timepoint/dependence policy was applied."
            )
    return retained


def _qualify(analysis: AnalysisDefinition, profile, records: list[EvidenceRecord]) -> PoolabilitySet:
    if profile.data_family == "adjusted_ratio_effect":
        return qualify_adjusted_ratio_records(profile, records)
    if profile.data_family == "event_counts_with_person_time" and profile.effect_type == "IRR":
        return qualify_rate_ratio_records(profile, records)
    raise NotImplementedError(
        f"No structured poolability implementation for {profile.data_family}/{profile.effect_type}."
    )


class BaseAnalysisAgent:
    """GraphRAG analysis-plan executor with evidence-to-analysis assignment."""

    def __init__(
        self,
        router: GlobalEvidenceRouter | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.router = router or DeterministicGlobalEvidenceRouter()
        self.calculator = CalculateMainMetaTool()
        self.progress_callback = progress_callback

    def _progress(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def run(
        self,
        *,
        protocol: dict[str, Any],
        task: dict[str, Any],
        analysis_profile: dict[str, Any],
        extractor_payloads: list[dict[str, Any]],
        study_by_article: dict[str, str],
    ) -> BaseAnalysisAgentResult:
        trace: list[ToolTraceEntry] = []
        self._progress("[plan] Resolving primary, secondary, subgroup, and sensitivity analysis specifications.")
        plan = resolve_analysis_plan_structured(protocol, task, analysis_profile)
        self._progress(
            f"[plan] Resolved {len(plan.analyses)} declared analyses; primary analysis is {plan.primary_analysis_id}."
        )
        trace.append(ToolTraceEntry(
            tool_name="resolve_analysis_plan_structured",
            status="completed",
            summary=f"Resolved {len(plan.analyses)} allowed analyses without reading gold labels.",
        ))

        self._progress("[graph] Building Study-Article-Evidence GraphRAG from extracted evidence.")
        graph = EvidenceGraph(load_evidence_records(extractor_payloads, study_by_article))
        inventory = graph.inventory()
        self._progress(
            f"[graph] Completed: {inventory.record_count} evidence records, {inventory.study_count} studies, "
            f"{inventory.graph_node_count} nodes, {inventory.graph_edge_count} edges."
        )
        trace.append(ToolTraceEntry(
            tool_name="build_evidence_graph",
            status="completed",
            summary=f"Indexed {inventory.record_count} evidence records from {inventory.study_count} studies.",
        ))

        self._progress(
            f"[router] Running {self.router.mode} global router across the full graph and {len(plan.analyses)} analyses."
        )
        global_routing = self.router.route(plan, graph)
        routed_count = sum(bool(route.selected_analysis_ids) for route in global_routing.routes)
        self._progress(
            f"[router] Completed: {routed_count}/{len(global_routing.routes)} evidence records assigned to at least one analysis."
        )
        trace.append(ToolTraceEntry(
            tool_name="global_evidence_router",
            status="completed",
            summary=global_routing.summary,
        ))

        results: list[AnalysisExecutionResult] = []
        assignment_index: dict[str, list[str]] = defaultdict(list)
        routes_by_id = {route.evidence_id: route for route in global_routing.routes}
        for analysis_index, analysis in enumerate(plan.analyses, start=1):
            self._progress(
                f"[analysis {analysis_index}/{len(plan.analyses)}] {analysis.analysis_id} "
                f"({analysis.kind}): reading global assignment matrix."
            )
            runtime_profile = runtime_profile_for_analysis(analysis, plan)
            candidates = graph.retrieve(runtime_profile)
            candidate_records = sum(len(group.records) for group in candidates.groups)
            self._progress(
                f"[analysis {analysis_index}/{len(plan.analyses)}] Retrieved {candidate_records} candidate evidence records "
                f"across {len(candidates.groups)} graph contexts."
            )
            assignments: list[EvidenceAssignmentDecision] = []
            selected_by_id: dict[str, EvidenceRecord] = {}
            for record in graph.records:
                route = routes_by_id[record.evidence_id]
                is_selected = analysis.analysis_id in route.selected_analysis_ids
                role = "included" if is_selected else (
                    "needs_review" if route.status == "needs_review" else "excluded"
                )
                assignments.append(EvidenceAssignmentDecision(
                    analysis_id=analysis.analysis_id,
                    evidence_id=record.evidence_id,
                    study_id=record.study_id,
                    role=role,
                    reason=route.reason,
                    reviewer_mode=global_routing.reviewer_mode,
                    selected_within_context=is_selected,
                ))
                if is_selected:
                    selected_by_id[record.evidence_id] = record

            selected_records = _apply_dependence_policy(
                analysis, list(selected_by_id.values()), assignments
            )
            retained_ids = {record.evidence_id for record in selected_records}
            for evidence_id in set(selected_by_id) - retained_ids:
                route = routes_by_id[evidence_id]
                route.selected_analysis_ids = [
                    value for value in route.selected_analysis_ids if value != analysis.analysis_id
                ]
                if not route.selected_analysis_ids and route.status == "routed":
                    route.status = "excluded"
                    route.reason = (
                        "Eligible globally, but another record was preferred after the analysis-level "
                        "timepoint/dependence policy was applied."
                    )
            self._progress(
                f"[analysis {analysis_index}/{len(plan.analyses)}] Evidence assignment completed: "
                f"included {len(selected_records)}, excluded "
                f"{sum(item.role == 'excluded' for item in assignments)}, "
                f"needs_review {sum(item.role == 'needs_review' for item in assignments)}."
            )
            for record in selected_records:
                assignment_index[record.evidence_id].append(analysis.analysis_id)

            warnings: list[str] = []
            poolability = _qualify(analysis, runtime_profile, selected_records)
            unique_studies = {effect.study_id for effect in poolability.study_effects}
            self._progress(
                f"[analysis {analysis_index}/{len(plan.analyses)}] Numerical poolability: "
                f"{len(poolability.study_effects)} poolable effects across {len(unique_studies)} independent studies."
            )
            statistics = None
            status = "insufficient_data"
            if analysis.dependence_policy == "correlated_effects_not_implemented":
                status = "blocked"
                warnings.append("Correlated-effect synthesis is requested but is not implemented in structured.")
            elif len(unique_studies) >= analysis.minimum_studies:
                statistics = self.calculator.invoke(runtime_profile, poolability.study_effects)
                status = "completed"
                self._progress(
                    f"[analysis {analysis_index}/{len(plan.analyses)}] Pooling completed: "
                    f"{statistics.effect_type}={statistics.pooled_effect:.4f}, "
                    f"95% CI {statistics.ci_lower:.4f}-{statistics.ci_upper:.4f}, k={statistics.k}."
                )
                if len(poolability.study_effects) != len(unique_studies):
                    warnings.append(
                        "Multiple outcomes from the same study were retained. The current calculator treats "
                        "study-outcome units as separate effects; use RVE or a multilevel model for strict "
                        "within-study dependence handling."
                    )
            else:
                warnings.append(
                    f"Only {len(unique_studies)} independent studies were poolable; "
                    f"minimum is {analysis.minimum_studies}."
                )
                self._progress(
                    f"[analysis {analysis_index}/{len(plan.analyses)}] Not pooled: insufficient independent studies "
                    f"({len(unique_studies)}/{analysis.minimum_studies})."
                )

            results.append(AnalysisExecutionResult(
                analysis=analysis,
                status=status,
                candidate_set=candidates,
                assignments=assignments,
                selected_records=selected_records,
                poolability=poolability,
                statistics=statistics,
                warnings=warnings,
            ))

        primary = next(result for result in results if result.analysis.analysis_id == plan.primary_analysis_id)
        if primary.status == "completed":
            overall_status = "completed"
        elif any(result.status == "completed" for result in results):
            overall_status = "partial"
        else:
            overall_status = "blocked"
        trace.append(ToolTraceEntry(
            tool_name="execute_analysis_plan_structured",
            status="completed" if overall_status != "blocked" else "blocked",
            summary=(
                f"Completed {sum(result.status == 'completed' for result in results)} of "
                f"{len(results)} planned primary/secondary/subgroup/sensitivity analyses."
            ),
        ))
        self._progress(
            f"[done] Overall status={overall_status}; completed "
            f"{sum(result.status == 'completed' for result in results)}/{len(results)} analyses."
        )
        return BaseAnalysisAgentResult(
            status=overall_status,
            plan=plan,
            inventory=inventory,
            global_routing=global_routing,
            analyses=results,
            evidence_assignment_index=dict(assignment_index),
            trace=trace,
        )
