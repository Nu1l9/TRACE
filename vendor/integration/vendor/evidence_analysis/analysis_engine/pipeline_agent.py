from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .evidence_graph import EvidenceGraph, load_evidence_records
from .eligibility import is_categorical_high_low
from .forest_plot import ForestPlotRenderer
from analysis_core.models import EvidenceRecord, PoolabilitySet, ToolTraceEntry
from analysis_core.plan_resolver import resolve_analysis_plan_structured, runtime_profile_for_analysis
from analysis_core.base_agent import _apply_dependence_policy

from .models import (
    PipelineAnalysisAgentResult,
    AnalysisMemory,
    ForestPlotArtifact,
    PlannerAction,
    ReportingBiasAssessment,
    TopicAnalysisResult,
    TopicMemory,
)
from .planner import AnalysisPlanner, DeterministicAnalysisPlanner
from .pairwise_tools import MetaforPairwiseMetaTool, PairwiseEffectTool
from .retrieval import EmbeddingBackend, HashEmbeddingBackend, HybridGraphRetriever
from .reporting_bias import MetaforReportingBiasTool
from .reporting_bias_selector import DeterministicReportingBiasSelector, ReportingBiasSelector
from .robustness import MetaforRobustnessDiagnosticsTool
from .runtime_facets import RuntimeFacetAnnotator, RuntimeFacetResult
from .topic_reviewer import DeterministicTopicReviewer, TopicReviewer
from .method_selector import MethodSelectionDecision, StatisticalMethodSelector


def _apply_categorical_high_low_rule(definition, selected, decisions):
    """Apply the opt-in contrast gate after review and record every exclusion."""
    if "Analysis eligibility: categorical_highest_vs_lowest" not in definition.selection_rules:
        return selected
    accepted = [record for record in selected if is_categorical_high_low(record)]
    rejected = {record.evidence_id for record in selected if not is_categorical_high_low(record)}
    for decision in decisions:
        if decision.analysis_id == definition.analysis_id and decision.evidence_id in rejected:
            decision.role = "excluded"
            decision.selected_within_context = False
            decision.reason = "Excluded by prespecified categorical highest-versus-lowest contrast rule."
    return accepted


def _dependent_study_ids(definition: Any, effects: list[Any]) -> list[str]:
    """Identify repeated studies when the protocol requires one independent effect."""
    if definition.dependence_policy == "clustered_effects_rho_0_6":
        if definition.pooling_model != "random_effects" or not definition.pooling_scale.endswith("__clustered_rho_0_6"):
            raise ValueError("Clustered effects require random_effects with the clustered_rho_0_6 calculator.")
        return []
    rules = " ".join(str(rule).casefold() for rule in definition.selection_rules)
    requires_independence = (
        definition.dependence_policy == "one_effect_per_study"
        or "one eligible effect per independent study" in rules
        or "one calculator-complete all-cause mortality record per independent randomized trial" in rules
    )
    if not requires_independence:
        return []
    counts = Counter(effect.study_id for effect in effects)
    return sorted(study_id for study_id, count in counts.items() if count > 1)


class PipelineAnalysisAgent:
    """Graph-first controller: retrieve -> plan -> local review -> memory -> calculate."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingBackend | None = None,
        planner: AnalysisPlanner | None = None,
        reviewer: TopicReviewer | None = None,
        reporting_bias_tool: MetaforReportingBiasTool | None = None,
        reporting_bias_selector: ReportingBiasSelector | None = None,
        robustness_tool: MetaforRobustnessDiagnosticsTool | None = None,
        forest_plot_renderer: ForestPlotRenderer | None = None,
        method_selector: StatisticalMethodSelector | None = None,
        runtime_facet_annotator: RuntimeFacetAnnotator | None = None,
        pre_review_filter_mode: str = "strict_protocol",
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.embeddings = embeddings or HashEmbeddingBackend()
        self.planner = planner or DeterministicAnalysisPlanner()
        self.reviewer = reviewer or DeterministicTopicReviewer()
        self.effect_tool = PairwiseEffectTool()
        self.calculator = MetaforPairwiseMetaTool()
        self.reporting_bias_tool = reporting_bias_tool or MetaforReportingBiasTool()
        self.reporting_bias_selector = reporting_bias_selector or DeterministicReportingBiasSelector(
            minimum_studies=self.reporting_bias_tool.minimum_studies
        )
        self.robustness_tool = robustness_tool or MetaforRobustnessDiagnosticsTool()
        self.forest_plot_renderer = forest_plot_renderer or ForestPlotRenderer()
        self.method_selector = method_selector
        self.runtime_facet_annotator = runtime_facet_annotator
        self.pre_review_filter_mode = pre_review_filter_mode
        self.progress_callback = progress_callback
        # Retained only as runtime artefacts for inspection/export by a runner.
        # They are not model inputs and do not alter benchmark scoring.
        self.last_graph = None
        self.last_candidate_sets = None
        self.last_runtime_facets: RuntimeFacetResult | None = None

    def _progress(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def _build_graph(self, records, plan, runtime_facets):
        return EvidenceGraph(records, runtime_facets=runtime_facets)

    def _make_retriever(self, graph):
        return HybridGraphRetriever(
            graph, self.embeddings, pre_review_filter_mode=self.pre_review_filter_mode
        )

    def _review_candidates(self, definition, candidates, plan, retriever):
        return self.reviewer.review(definition, candidates), candidates

    @staticmethod
    def _profile_with_selected_method(profile, decision: MethodSelectionDecision):
        """Translate a method-tool decision into metafor's internal dispatch key."""
        scale = profile.pooling_scale.split("__", 1)[0]
        if decision.heterogeneity_estimator == "reml":
            scale += "__reml"
        elif decision.heterogeneity_estimator == "paule_mandel":
            scale += "__paule_mandel"
        if decision.confidence_interval == "hartung_knapp_t":
            scale += "__hartung_knapp_t"
        return profile.model_copy(update={
            "pooling_model": decision.pooling_model,
            "pooling_scale": scale,
        })

    def run(
        self,
        *,
        protocol: dict[str, Any],
        task: dict[str, Any],
        analysis_profile: dict[str, Any],
        extractor_payloads: list[dict[str, Any]],
        study_by_article: dict[str, str],
        reporting_bias_artifact_dir: Path | None = None,
        forest_plot_artifact_dir: Path | None = None,
    ) -> PipelineAnalysisAgentResult:
        trace: list[ToolTraceEntry] = []
        self.last_graph = None
        self.last_candidate_sets = None
        self.last_runtime_facets = None
        self._progress("[plan] Resolving declared primary, secondary, subgroup, and sensitivity topics.")
        plan = resolve_analysis_plan_structured(protocol, task, analysis_profile)
        self._progress(f"[plan] Resolved {len(plan.analyses)} analysis topics.")

        records = load_evidence_records(extractor_payloads, study_by_article)
        runtime_facets: dict[str, dict[str, Any]] | None = None
        if self.runtime_facet_annotator is not None:
            self._progress(
                "[semantic-graph] Generating runtime facets from source evidence; no evidence selection or statistics computed."
            )
            self.last_runtime_facets = self.runtime_facet_annotator.annotate(plan, records)
            runtime_facets = self.last_runtime_facets.by_evidence_id()
            explicit = sum(
                1
                for annotation in self.last_runtime_facets.annotations
                for dimension in annotation.dimensions
                if dimension.status in {"explicit_level", "explicit_other"}
            )
            self._progress(
                f"[semantic-graph] Completed source facet audit for {len(self.last_runtime_facets.annotations)} evidence records "
                f"({explicit} explicit level or explicit non-target annotations)."
            )
            trace.append(ToolTraceEntry(
                tool_name="annotate_runtime_source_facets",
                status="completed",
                summary=(
                    f"{self.last_runtime_facets.mode}: generated runtime source facets for "
                    f"{len(self.last_runtime_facets.annotations)} Evidence records; "
                    "no evidence-selection or Gold inputs were used."
                ),
            ))

        self._progress("[graph] Building Study-Article-Evidence GraphRAG.")
        graph = self._build_graph(records, plan, runtime_facets)
        self.last_graph = graph
        inventory = graph.inventory()
        self._progress(
            f"[graph] Completed: {inventory.record_count} evidence records, {inventory.study_count} studies, "
            f"{inventory.graph_node_count} nodes, {inventory.graph_edge_count} edges."
        )
        trace.append(ToolTraceEntry(
            tool_name="build_evidence_graph",
            status="completed",
            summary=f"Built GraphRAG from {inventory.record_count} Evidence records.",
        ))

        retriever = self._make_retriever(graph)
        self._progress(f"[inventory] Running graph-constrained candidate inventory across {len(plan.analyses)} topics.")
        retrieval_inventory = retriever.inventory(plan)
        total_candidates = sum(item.candidate_record_count for item in retrieval_inventory)
        self._progress(
            f"[inventory] Completed: {total_candidates} graph-constrained candidate records across {len(plan.analyses)} topics."
        )
        trace.append(ToolTraceEntry(
            tool_name="inventory_topic_candidates",
            status="completed",
            summary=f"Built graph-only inventory for {len(retrieval_inventory)} declared topics.",
        ))

        memory = AnalysisMemory(topics={
            definition.analysis_id: TopicMemory(
                analysis_id=definition.analysis_id,
                title=definition.title,
                kind=definition.kind,
                minimum_studies=definition.minimum_studies,
                state="retrieved",
                candidate_record_count=item.candidate_record_count,
                candidate_context_count=item.candidate_context_count,
                candidate_study_count=item.candidate_study_count,
                next_allowed_actions=["review_topic"] if item.candidate_record_count else ["block_topic"],
            )
            for definition, item in zip(plan.analyses, retrieval_inventory)
        })

        self._progress(
            f"[Planner] Reviewing candidate counts across {len(plan.analyses)} topics to schedule review_topic calls."
        )
        retrieval_decision = self.planner.plan_retrieval(plan, retrieval_inventory)
        memory.planner_decisions.append(retrieval_decision)
        trace.append(ToolTraceEntry(
            tool_name="plan_topic_review_tools",
            status="completed",
            summary=retrieval_decision.summary,
        ))
        self._progress(
            f"[Planner] Scheduled {sum(action.action == 'review_topic' for action in retrieval_decision.actions)} topics for local review."
        )

        definition_by_id = {definition.analysis_id: definition for definition in plan.analyses}
        review_ids = {
            action.analysis_id
            for action in retrieval_decision.actions
            if action.action == "review_topic" and action.analysis_id in definition_by_id
        }
        if review_ids:
            review_plan = plan.model_copy(update={
                "analyses": [item for item in plan.analyses if item.analysis_id in review_ids],
            })
            self._progress(
                f"[retrieval] Running {self.embeddings.mode} embedding retrieval for {len(review_ids)} approved topics."
            )
            candidate_sets, _ = retriever.retrieve_all(review_plan)
        else:
            candidate_sets = {}
        self.last_candidate_sets = candidate_sets
        trace.append(ToolTraceEntry(
            tool_name="retrieve_selected_topic_candidates",
            status="completed",
            summary=(
                f"Embedding-ranked local Evidence for {len(review_ids)} Planner-approved topics "
                f"using {self.embeddings.mode}."
            ),
        ))
        result_by_id: dict[str, TopicAnalysisResult] = {}
        selected_records_by_analysis: dict[str, dict[str, EvidenceRecord]] = {}
        reviewed_ids: set[str] = set()
        for action in retrieval_decision.actions:
            if action.action != "review_topic" or action.analysis_id in reviewed_ids:
                continue
            definition = definition_by_id.get(action.analysis_id)
            if not definition:
                continue
            candidates = candidate_sets[definition.analysis_id]
            if not candidates.groups:
                continue
            contested_contexts = sum(len(group.records) > 1 for group in candidates.groups)
            self._progress(
                f"[review_topic] {definition.analysis_id}: passing {sum(len(group.records) for group in candidates.groups)} candidates "
                f"to {self.reviewer.mode} Reviewer ({contested_contexts} contested contexts)."
            )
            review, candidates = self._review_candidates(definition, candidates, plan, retriever)
            candidate_sets[definition.analysis_id] = candidates
            reviewed_ids.add(definition.analysis_id)
            selected_by_id = {
                record.evidence_id: record
                for group in candidates.groups
                for record in group.records
                if record.evidence_id in review.selected_evidence_ids
            }
            selected = _apply_dependence_policy(definition, list(selected_by_id.values()), review.assignments)
            selected = _apply_categorical_high_low_rule(definition, selected, review.assignments)
            selected_records_by_analysis[definition.analysis_id] = {
                record.evidence_id: record for record in selected
            }
            profile = runtime_profile_for_analysis(definition, plan)
            poolability = self.effect_tool.qualify(profile, selected)
            poolable_studies = {effect.study_id for effect in poolability.study_effects}
            topic_memory = memory.topics[definition.analysis_id]
            topic_memory.state = "poolability_checked"
            topic_memory.selected_record_count = len(selected)
            topic_memory.selected_study_count = len({record.study_id for record in selected})
            topic_memory.poolable_effect_count = len(poolability.study_effects)
            topic_memory.poolable_study_count = len(poolable_studies)
            topic_memory.next_allowed_actions = ["calculate_meta", "block_topic"]
            topic_memory.notes.append(review.summary)
            result_by_id[definition.analysis_id] = TopicAnalysisResult(
                analysis=definition,
                status="insufficient_data",
                candidate_inventory=next(item for item in retrieval_inventory if item.analysis_id == definition.analysis_id),
                assignments=review.assignments,
                source_resolutions=review.source_resolutions,
                source_resolution_issues=review.source_resolution_issues,
                selected_evidence_ids=[record.evidence_id for record in selected],
                poolability=poolability,
                warnings=[],
            )
            self._progress(
                f"[assess_poolability] {definition.analysis_id}: selected {len(selected)} records; "
                f"poolable {len(poolability.study_effects)} effects across {len(poolable_studies)} independent studies."
            )
            trace.append(ToolTraceEntry(
                tool_name="derive_pairwise_effects",
                status="completed",
                summary=(
                    f"{definition.analysis_id}: derived {len(poolability.study_effects)} poolable effects "
                    f"from data_family={profile.data_family}."
                ),
            ))

        # Any topic the planner did not send to review stays explicitly blocked in memory.
        for definition in plan.analyses:
            if definition.analysis_id in result_by_id:
                continue
            topic_memory = memory.topics[definition.analysis_id]
            topic_memory.state = "blocked"
            topic_memory.notes.append("Planner did not schedule a topic-review tool call.")
            result_by_id[definition.analysis_id] = TopicAnalysisResult(
                analysis=definition,
                status="blocked",
                candidate_inventory=next(item for item in retrieval_inventory if item.analysis_id == definition.analysis_id),
                assignments=[],
                selected_evidence_ids=[],
                poolability=PoolabilitySet(decisions=[], study_effects=[]),
                warnings=["No topic-review tool call was scheduled."],
            )
        trace.append(ToolTraceEntry(
            tool_name="review_topic_candidates",
            status="completed",
            summary=f"Reviewed {len(reviewed_ids)} topic-local graph subgraphs with {self.reviewer.mode} reviewer.",
        ))

        self._progress("[Planner] Reviewing updated memory to schedule calculate_meta calls.")
        feasibility_decision = self.planner.plan_feasibility(memory)
        memory.planner_decisions.append(feasibility_decision)
        trace.append(ToolTraceEntry(
            tool_name="plan_synthesis_tools",
            status="completed",
            summary=feasibility_decision.summary,
        ))

        calculation_ids = {
            action.analysis_id
            for action in feasibility_decision.actions
            if action.action == "calculate_meta"
        }
        for definition in plan.analyses:
            result = result_by_id[definition.analysis_id]
            topic_memory = memory.topics[definition.analysis_id]
            enough_studies = topic_memory.poolable_study_count >= definition.minimum_studies
            if definition.analysis_id in calculation_ids and enough_studies:
                dependent_studies = _dependent_study_ids(definition, result.poolability.study_effects)
                if dependent_studies:
                    warning = (
                        "Independent-study rule violated: multiple effects from "
                        f"{', '.join(dependent_studies)} remain in one meta-analysis. "
                        "Resolve or combine the source arms before calculation."
                    )
                    result.status = "blocked"
                    result.warnings.append(warning)
                    topic_memory.state = "blocked"
                    topic_memory.notes.append(warning)
                    topic_memory.next_allowed_actions = []
                    trace.append(ToolTraceEntry(
                        tool_name="validate_independent_studies",
                        status="blocked",
                        summary=f"{definition.analysis_id}: {warning}",
                    ))
                    self._progress(f"[calculate_meta] {definition.analysis_id}: {warning}")
                    continue
                profile = runtime_profile_for_analysis(definition, plan)
                if definition.dependence_policy == "clustered_effects_rho_0_6":
                    result.warnings.append(
                        "Multiple outcomes from one study were pooled with a multilevel random-effects model, "
                        "assumed within-study sampling correlation rho=0.6, and study-cluster robust confidence interval. "
                        "Sensitivity to the unknown correlation should be checked before publication."
                    )
                if self.method_selector is not None:
                    self._progress(
                        f"[select_statistical_method] {definition.analysis_id}: selecting pooling model and interval estimator."
                    )
                    try:
                        if "__clustered" in profile.pooling_scale:
                            # This separate tool has a declared multilevel/robust contract;
                            # the univariate method menu must not erase its route.
                            decision = None
                            result.warnings.append("Declared clustered calculator retained; univariate method selection does not apply.")
                        elif hasattr(self.method_selector, "select_with_context"):
                            decision = self.method_selector.select_with_context(
                                definition, result.poolability.study_effects,
                                protocol=protocol, task=task,
                                records=selected_records_by_analysis[definition.analysis_id],
                                selection_summary=topic_memory.notes, assignments=result.assignments,
                                poolability=result.poolability,
                            )
                        else:
                            decision = self.method_selector.select(definition, result.poolability.study_effects)
                        if decision is not None:
                            profile = self._profile_with_selected_method(profile, decision)
                            result.method_selection = decision
                    except (ValueError, RuntimeError) as exc:
                        result.status = "blocked"
                        topic_memory.state = "blocked"
                        topic_memory.next_allowed_actions = []
                        result.warnings.append(f"Method selection failed ({type(exc).__name__}); no default method substituted. See method_selection audit.")
                        trace.append(ToolTraceEntry(tool_name="select_statistical_method", status="blocked",
                                                    summary=f"{definition.analysis_id}: unsupported or failed method selection."))
                        continue
                    trace.append(ToolTraceEntry(
                        tool_name="select_statistical_method",
                        status="completed",
                        summary=(
                            f"{definition.analysis_id}: {decision.pooling_model}; "
                            f"estimator={decision.heterogeneity_estimator}; "
                            f"ci={decision.confidence_interval}."
                        ) if decision is not None else f"{definition.analysis_id}: declared clustered tool contract.",
                    ))
                try:
                    result.statistics = self.calculator.invoke(profile, result.poolability.study_effects)
                except (ValueError, RuntimeError) as exc:
                    result.status = "blocked"
                    topic_memory.state = "blocked"
                    topic_memory.next_allowed_actions = []
                    result.warnings.append(f"Calculation failed: {exc}. The evidence and chosen method were not replaced.")
                    trace.append(ToolTraceEntry(tool_name="calculate_pairwise_meta_metafor", status="blocked",
                                                summary=f"{definition.analysis_id}: {type(exc).__name__}."))
                    continue
                result.status = "completed"
                topic_memory.state = "completed"
                topic_memory.next_allowed_actions = []
                self._progress(
                    f"[calculate_meta] {definition.analysis_id}: completed, "
                    f"{result.statistics.effect_type}={result.statistics.pooled_effect:.4f}, k={result.statistics.k}."
                )
                result.forest_plot = self.forest_plot_renderer.render(
                    definition,
                    result.statistics,
                    result.poolability.study_effects,
                    artifact_dir=forest_plot_artifact_dir,
                    artifact_stem=definition.analysis_id,
                )
                trace.append(ToolTraceEntry(
                    tool_name="render_forest_plot",
                    status="completed" if result.forest_plot.status == "completed" else "skipped",
                    summary=(
                        f"{definition.analysis_id}: forest plot {result.forest_plot.status} "
                        f"from {result.forest_plot.k} poolable effects; "
                        "presentation only, not used for selection or calculation."
                    ),
                ))
                if result.forest_plot.status == "completed":
                    self._progress(f"[forest_plot] {definition.analysis_id}: rendered forest plot.")
                if definition.kind == "primary":
                    self._progress(
                        f"[robustness] {definition.analysis_id}: running leave-one-out, influence, heterogeneity-driver, and model-sensitivity diagnostics."
                    )
                    result.robustness = self.robustness_tool.assess(
                        definition.model_copy(update={"pooling_model": profile.pooling_model,
                                                      "pooling_scale": profile.pooling_scale}),
                        result.statistics,
                        result.poolability.study_effects,
                    )
                    trace.append(ToolTraceEntry(
                        tool_name="assess_robustness_metafor",
                        status="completed" if result.robustness.status == "completed" else "skipped",
                        summary=(
                            f"{definition.analysis_id}: {result.robustness.stability} "
                            f"(status={result.robustness.status}, k={result.robustness.k})."
                        ),
                    ))
                    self._progress(
                        f"[robustness] {definition.analysis_id}: {result.robustness.stability}."
                    )
                    self._progress(
                        f"[reporting_bias_selector] {definition.analysis_id}: projecting primary evidence to one effect per independent study."
                    )
                    projection = self.reporting_bias_selector.project(
                        definition,
                        result.poolability.study_effects,
                        selected_records_by_analysis.get(definition.analysis_id, {}),
                    )
                    result.reporting_bias_projection = projection
                    self._progress(
                        f"[reporting_bias_selector] {definition.analysis_id}: {projection.summary}"
                    )
                    trace.append(ToolTraceEntry(
                        tool_name="project_reporting_bias_evidence",
                        status="completed" if projection.status == "completed" else "skipped",
                        summary=(
                            f"{definition.analysis_id}: {projection.status}; "
                            f"{len(projection.selected_effects)} selected diagnostic effects "
                            f"from {projection.independent_study_count} independent studies."
                        ),
                    ))
                    if projection.status == "completed":
                        self._progress(
                            f"[reporting_bias] {definition.analysis_id}: running Egger, Begg, and trim-and-fill across "
                            f"{len(projection.selected_effects)} independent {projection.effect_type} effects."
                        )
                        result.reporting_bias = self.reporting_bias_tool.assess(
                            definition,
                            result.statistics,
                            projection.selected_effects,
                            artifact_dir=reporting_bias_artifact_dir,
                            artifact_stem=definition.analysis_id,
                        )
                    else:
                        insufficient_studies = (
                            projection.independent_study_count < projection.minimum_studies
                        )
                        result.reporting_bias = ReportingBiasAssessment(
                            status="not_assessable",
                            k=projection.source_effect_count,
                            independent_study_count=projection.independent_study_count,
                            minimum_studies=projection.minimum_studies,
                            interpretation=(
                                "not_assessable_insufficient_studies"
                                if insufficient_studies
                                else "not_assessable_no_effect_type_with_sufficient_independent_studies"
                            ),
                            caveats=[projection.summary],
                        )
                    trace.append(ToolTraceEntry(
                        tool_name="assess_reporting_bias_metafor",
                        status="completed" if result.reporting_bias.status == "completed" else "skipped",
                        summary=(
                            f"{definition.analysis_id}: {result.reporting_bias.interpretation} "
                            f"(status={result.reporting_bias.status}, k={result.reporting_bias.k})."
                        ),
                    ))
                    self._progress(
                        f"[reporting_bias] {definition.analysis_id}: {result.reporting_bias.interpretation}."
                    )
            elif result.status != "blocked":
                result.status = "insufficient_data"
                topic_memory.state = "blocked"
                topic_memory.next_allowed_actions = []
                result.warnings.append(
                    f"Poolable independent studies: {topic_memory.poolable_study_count}/{definition.minimum_studies}."
                )
                self._progress(
                    f"[block_topic] {definition.analysis_id}: insufficient poolable independent studies "
                    f"({topic_memory.poolable_study_count}/{definition.minimum_studies})."
                )

        analyses = [result_by_id[definition.analysis_id] for definition in plan.analyses]
        completed = sum(item.status == "completed" for item in analyses)
        status = "completed" if completed == len(analyses) else "partial" if completed else "blocked"
        trace.append(ToolTraceEntry(
            tool_name="execute_memory_driven_analysis",
            status="completed" if completed else "blocked",
            summary=f"Completed {completed} of {len(analyses)} topics through planner-scheduled tools.",
        ))
        self._progress(f"[done] Overall status={status}; completed {completed}/{len(analyses)} topics.")
        return PipelineAnalysisAgentResult(
            status=status,
            plan=plan,
            inventory=inventory,
            retrieval_inventory=retrieval_inventory,
            memory=memory,
            analyses=analyses,
            trace=trace,
        )
