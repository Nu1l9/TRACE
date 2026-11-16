from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Protocol

from .assignment_reviewer import eligibility_reasons, selection_score
from .evidence_graph import EvidenceGraph
from .models import GlobalEvidenceRoute, GlobalRoutingResult, ResolvedAnalysisPlan


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _canonical_timepoint(value: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(months?|weeks?|years?)\b", value.casefold())
    return f"{match.group(1)} {match.group(2)}" if match else _normalise(value)


def _context_key(record) -> tuple[str, str, str, str, str, str, str]:
    return (
        record.study_id,
        _normalise(record.intervention_or_exposure),
        _normalise(record.outcome),
        _normalise(record.comparator),
        _normalise(record.population_or_subgroup),
        _canonical_timepoint(record.follow_up),
        _normalise(record.estimand),
    )


def _compact_evidence(record) -> dict:
    source = record.source_location
    return {
        "evidence_id": record.evidence_id,
        "article_id": record.article_id,
        "study_id": record.study_id,
        "target_id": record.target_id,
        "exposure": record.intervention_or_exposure,
        "outcome": record.outcome,
        "outcome_code": record.outcome_code,
        "comparison": record.comparator,
        "population_or_subgroup": record.population_or_subgroup,
        "follow_up": record.follow_up,
        "estimand": record.estimand,
        "adjustment_model": record.adjustment_model,
        "data_family": record.data_family,
        "effect": {
            "type": record.reported_effect_type,
            "value": record.reported_effect_value,
            "ci_lower": record.ci_lower,
            "ci_upper": record.ci_upper,
        },
        "quantitative_usability": record.quantitative_usability,
        "source_location": {
            "document_id": source.get("document_id"),
            "page": source.get("page"),
            "source_type": source.get("source_type"),
            "source_id": source.get("source_id"),
            "source_quote": source.get("source_quote") or source.get("source_snippet"),
        },
    }


def _graph_payload(graph: EvidenceGraph) -> dict:
    study_to_records: dict[str, list[str]] = defaultdict(list)
    contexts: dict[tuple[str, str, str, str, str, str, str], list[str]] = defaultdict(list)
    for record in graph.records:
        study_to_records[record.study_id].append(record.evidence_id)
        contexts[_context_key(record)].append(record.evidence_id)
    return {
        "inventory": graph.inventory().model_dump(),
        "study_to_evidence": [
            {"study_id": study_id, "evidence_ids": sorted(evidence_ids)}
            for study_id, evidence_ids in sorted(study_to_records.items())
        ],
        "selection_contexts": [
            {
                "study_id": context[0],
                "exposure": context[1],
                "outcome": context[2],
                "comparison": context[3],
                "population": context[4],
                "timepoint": context[5],
                "estimand": context[6],
                "evidence_ids": sorted(evidence_ids),
            }
            for context, evidence_ids in sorted(contexts.items())
        ],
        "evidence": [_compact_evidence(record) for record in graph.records],
    }


def _compact_router_plan(plan: ResolvedAnalysisPlan) -> dict:
    """Compile the plan into the fields that affect an Evidence-to-analysis route.

    The resolved plan is intentionally verbose because it is also written to the
    audit artifact. Sending that full artifact to a reasoning model repeats the
    same exposure, comparison, and selection rules for every analysis.
    """
    return {
        "primary_analysis_id": plan.primary_analysis_id,
        "routing_rule": "For each analysis and evidence context, retain at most one competing model.",
        "analyses": [
            {
                "id": analysis.analysis_id,
                "kind": analysis.kind,
                "target": analysis.target_id,
                "outcomes": analysis.eligible_outcome_codes,
                "data_families": analysis.eligible_data_families,
                "effect_types": analysis.accepted_effect_types,
                "estimands": analysis.estimands,
                "exposures": analysis.exposure_terms,
                "comparisons": analysis.comparison_terms,
                "population_policy": analysis.population_policy,
                "population_terms": analysis.population_terms,
                "timepoint_policy": analysis.timepoint_policy,
                "model_policy": analysis.model_policy,
                "dependence_policy": analysis.dependence_policy,
                "parent": analysis.parent_analysis_id,
                "dimension": {
                    "field": analysis.dimension_field,
                    "value_id": analysis.dimension_value_id,
                    "terms": analysis.dimension_terms,
                },
            }
            for analysis in plan.analyses
        ],
    }


def _compact_router_graph(graph: EvidenceGraph) -> dict:
    """Serialize the complete graph without retrieval-only text or repeated edges."""
    contexts: dict[tuple[str, str, str, str, str, str, str], list[str]] = defaultdict(list)
    evidence: list[dict] = []
    for record in graph.records:
        contexts[_context_key(record)].append(record.evidence_id)
        evidence.append({
            "id": record.evidence_id,
            "study": record.study_id,
            "article": record.article_id,
            "target": record.target_id,
            "exposure": record.intervention_or_exposure,
            "outcome": record.outcome,
            "outcome_code": record.outcome_code,
            "comparison": record.comparator,
            "population": record.population_or_subgroup,
            "timepoint": record.follow_up,
            "estimand": record.estimand,
            "model": record.adjustment_model,
            "data_family": record.data_family,
            "effect_type": record.reported_effect_type,
            "usability": record.quantitative_usability,
        })
    return {
        "evidence": evidence,
        "competing_model_contexts": [
            {"evidence_ids": sorted(evidence_ids)}
            for _, evidence_ids in sorted(contexts.items())
            if len(evidence_ids) > 1
        ],
    }


class GlobalEvidenceRouter(Protocol):
    mode: str

    def route(self, plan: ResolvedAnalysisPlan, graph: EvidenceGraph) -> GlobalRoutingResult: ...


class DeterministicGlobalEvidenceRouter:
    """Offline global router used for repeatable tests and ablations."""

    mode = "deterministic"

    def route(self, plan: ResolvedAnalysisPlan, graph: EvidenceGraph) -> GlobalRoutingResult:
        selected_by_evidence: dict[str, list[str]] = defaultdict(list)
        eligible_by_evidence: set[str] = set()
        for analysis in plan.analyses:
            contexts: dict[tuple[str, str, str, str, str, str, str], list] = defaultdict(list)
            for record in graph.records:
                if not eligibility_reasons(record, analysis):
                    eligible_by_evidence.add(record.evidence_id)
                    contexts[_context_key(record)].append(record)
            for records in contexts.values():
                selected = max(records, key=lambda record: selection_score(record, analysis))
                selected_by_evidence[selected.evidence_id].append(analysis.analysis_id)

        routes: list[GlobalEvidenceRoute] = []
        for record in graph.records:
            analysis_ids = sorted(selected_by_evidence.get(record.evidence_id, []))
            if analysis_ids:
                routes.append(GlobalEvidenceRoute(
                    evidence_id=record.evidence_id,
                    study_id=record.study_id,
                    selected_analysis_ids=analysis_ids,
                    status="routed",
                    reason="Globally matched to the analysis plan and retained within its GraphRAG context.",
                ))
            elif record.evidence_id in eligible_by_evidence:
                routes.append(GlobalEvidenceRoute(
                    evidence_id=record.evidence_id,
                    study_id=record.study_id,
                    status="excluded",
                    reason="Eligible for at least one analysis but a competing record was preferred in the same context.",
                ))
            else:
                routes.append(GlobalEvidenceRoute(
                    evidence_id=record.evidence_id,
                    study_id=record.study_id,
                    status="excluded",
                    reason="Does not match any permitted primary, secondary, subgroup, or sensitivity analysis.",
                ))
        return GlobalRoutingResult(
            reviewer_mode=self.mode,
            routes=routes,
            summary=f"Deterministically routed {sum(bool(route.selected_analysis_ids) for route in routes)} of {len(routes)} Evidence nodes in one global pass.",
        )


class LLMGlobalEvidenceRouter:
    """One official OpenAI Responses request that routes the whole Evidence graph to the plan."""

    mode = "llm"

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        reasoning_effort: str = "xhigh",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def route(self, plan: ResolvedAnalysisPlan, graph: EvidenceGraph) -> GlobalRoutingResult:
        try:
            from openai import APIStatusError, OpenAI
            from pydantic import BaseModel, Field
        except ImportError as exc:
            raise RuntimeError("LLM global router requires the openai package.") from exc
        if not self.api_key:
            raise RuntimeError("LLM global router needs OPENAI_API_KEY.")

        class LLMRoute(BaseModel):
            i: str
            a: list[str] = Field(default_factory=list)
            s: str

        class LLMRoutingResponse(BaseModel):
            routes: list[LLMRoute]
            summary: str

        plan_payload = _compact_router_plan(plan)
        graph_payload = _compact_router_graph(graph)
        prompt = (
            "You are the single global evidence router for a meta-analysis system. You receive the entire "
            "prespecified routing plan and the complete compact Study-Article-Evidence GraphRAG payload. Review "
            "the graph jointly, not one analysis at a time. For every supplied evidence ID, decide the analyses "
            "this exact record should feed. A record can feed multiple analyses, such as a primary analysis and an "
            "HR-only sensitivity analysis. Do not route subgroup-only records into an overall-only primary analysis. "
            "Within every competing_model_context, choose at most one competing model per analysis. Respect the "
            "declared data family, estimand, comparison, population and effect-measure constraints. Do not calculate "
            "pooled effects. Do not invent IDs.\n\n"
            "Return exactly one compact JSON object: {\"routes\":[...],\"summary\":\"...\"}. routes must have "
            "exactly one item for every evidence ID. Each item is {\"i\":evidence_id,\"a\":[analysis_id,...],"
            "\"s\":\"routed|excluded|needs_review\"}. a may be empty. Do not include natural-language reasons "
            "per route, markdown, or any other keys.\n\n"
            f"ROUTING_PLAN:\n{json.dumps(plan_payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"COMPLETE_COMPACT_GRAPH:\n{json.dumps(graph_payload, ensure_ascii=False, separators=(',', ':'))}"
        )
        self.last_request_char_count = len(prompt)
        try:
            response = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=0,
                timeout=240.0,
            ).responses.create(
                model=self.model,
                input=[{"role": "user", "content": prompt}],
                reasoning={"effort": self.reasoning_effort},
                store=False,
            )
        except APIStatusError as exc:
            if exc.status_code == 524:
                raise RuntimeError(
                    "The configured OpenAI API endpoint timed out before returning global routing results (HTTP 524). "
                    f"Compact request size was ~{self.last_request_char_count:,} chars; wait at least 120s before retrying. "
                    "If timeouts persist, set LLM_REASONING_EFFORT to 'high'."
                ) from exc
            raise
        content = response.output_text or ""
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = LLMRoutingResponse.model_validate_json(content)
        except ValueError as exc:
            raise RuntimeError("LLM global router did not return a valid routing JSON object.") from exc
        return _validate_llm_routes(parsed.routes, parsed.summary, plan, graph)


def _validate_llm_routes(routes, summary: str, plan: ResolvedAnalysisPlan, graph: EvidenceGraph) -> GlobalRoutingResult:
    known_evidence = {record.evidence_id: record for record in graph.records}
    known_analysis = {analysis.analysis_id for analysis in plan.analyses}
    by_id = {}
    for route in routes:
        if route.i not in known_evidence or route.i in by_id:
            continue
        invalid_ids = set(route.a) - known_analysis
        status = route.s if route.s in {"routed", "excluded", "needs_review"} else "needs_review"
        if invalid_ids:
            status = "needs_review"
        selected_analysis_ids = sorted(set(route.a) & known_analysis)
        if selected_analysis_ids and status == "excluded":
            status = "routed"
        by_id[route.i] = GlobalEvidenceRoute(
            evidence_id=route.i,
            study_id=known_evidence[route.i].study_id,
            selected_analysis_ids=selected_analysis_ids,
            status=status,
            reason=(
                "The global LLM routed this evidence to its listed analysis IDs."
                if selected_analysis_ids
                else "The global LLM did not route this evidence to a permitted analysis."
            ),
        )
    for evidence_id, record in known_evidence.items():
        if evidence_id not in by_id:
            by_id[evidence_id] = GlobalEvidenceRoute(
                evidence_id=evidence_id,
                study_id=record.study_id,
                status="needs_review",
                reason="The global LLM response omitted this evidence node.",
            )

    # The LLM is asked to choose one model inside each GraphRAG context. This
    # guard keeps a malformed response from silently double-counting a model.
    for analysis in plan.analyses:
        selected_contexts: dict[tuple[str, str, str, str, str, str, str], list] = defaultdict(list)
        for evidence_id, route in by_id.items():
            if analysis.analysis_id in route.selected_analysis_ids:
                selected_contexts[_context_key(known_evidence[evidence_id])].append(known_evidence[evidence_id])
        for records in selected_contexts.values():
            if len(records) < 2:
                continue
            winner = max(records, key=lambda record: selection_score(record, analysis))
            for record in records:
                if record.evidence_id == winner.evidence_id:
                    continue
                route = by_id[record.evidence_id]
                route.selected_analysis_ids = [
                    value for value in route.selected_analysis_ids if value != analysis.analysis_id
                ]
                if not route.selected_analysis_ids:
                    route.status = "excluded"
                    route.reason = (
                        "The global LLM selected multiple competing models in one GraphRAG context; "
                        "the deterministic context safeguard retained another record."
                    )
    return GlobalRoutingResult(
        reviewer_mode="llm",
        routes=[by_id[record.evidence_id] for record in graph.records],
        summary=summary,
    )
