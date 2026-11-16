from __future__ import annotations

import json
import os
from typing import Protocol

from pydantic import BaseModel, Field

from analysis_core.models import ResolvedAnalysisPlan

from .llm_transport import create_chat_completion
from .models import AnalysisMemory, PlannerAction, PlannerDecision, TopicCandidateInventory


class AnalysisPlanner(Protocol):
    mode: str

    def plan_retrieval(
        self,
        plan: ResolvedAnalysisPlan,
        inventory: list[TopicCandidateInventory],
    ) -> PlannerDecision: ...

    def plan_feasibility(self, memory: AnalysisMemory) -> PlannerDecision: ...


class DeterministicAnalysisPlanner:
    """Offline baseline with the same tool protocol as the LLM planner."""

    mode = "deterministic"

    def plan_retrieval(self, plan, inventory) -> PlannerDecision:
        definitions = {item.analysis_id: item for item in plan.analyses}
        actions = [
            PlannerAction(
                action=(
                    "review_topic"
                    if item.candidate_study_count >= definitions[item.analysis_id].minimum_studies
                    else "block_topic"
                ),
                analysis_id=item.analysis_id,
                reason=(
                    "Graph inventory has enough plausible independent studies for detailed review."
                    if item.candidate_study_count >= definitions[item.analysis_id].minimum_studies
                    else "Graph inventory is below the minimum independent-study threshold."
                ),
            )
            for item in inventory
        ]
        return PlannerDecision(
            phase="retrieval",
            actions=actions,
            summary="Scheduled local review only for topics whose graph inventory meets the independent-study threshold.",
            planner_mode=self.mode,
        )

    def plan_feasibility(self, memory: AnalysisMemory) -> PlannerDecision:
        actions: list[PlannerAction] = []
        for topic in memory.topics.values():
            if topic.poolable_study_count >= topic.minimum_studies:
                actions.append(PlannerAction(
                    action="calculate_meta",
                    analysis_id=topic.analysis_id,
                    reason="The required number of independent poolable studies is available.",
                ))
            else:
                actions.append(PlannerAction(
                    action="block_topic",
                    analysis_id=topic.analysis_id,
                    reason="The required number of independent poolable studies is not available.",
                ))
        return PlannerDecision(
            phase="feasibility",
            actions=actions,
            summary="Converted memory-based feasibility counts into calculation or block actions.",
            planner_mode=self.mode,
        )


class LLMAnalysisPlanner:
    """LLM planner that schedules tools without receiving raw Evidence rows."""

    mode = "llm"

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        reasoning_effort: str = "xhigh",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_retries = max_retries
        self.calls_made = 0

    def plan_retrieval(self, plan, inventory) -> PlannerDecision:
        definitions = {item.analysis_id: item for item in plan.analyses}
        inventory_by_id = {item.analysis_id: item for item in inventory}
        payload = {
            "primary_analysis_id": plan.primary_analysis_id,
            "topics": [
                {
                    "analysis_id": definition.analysis_id,
                    "kind": definition.kind,
                    "title": definition.title,
                    "outcome": definition.outcome_group,
                    "data_family": definition.eligible_data_families,
                    "effect_representations": definition.accepted_effect_types,
                    "minimum_independent_studies": definition.minimum_studies,
                    "prespecified": definition.prespecified,
                    "candidate_records": inventory_by_id[definition.analysis_id].candidate_record_count,
                    "candidate_contexts": inventory_by_id[definition.analysis_id].candidate_context_count,
                    "candidate_studies": inventory_by_id[definition.analysis_id].candidate_study_count,
                }
                for definition in plan.analyses
            ],
        }
        decision = self._request(
            phase="retrieval",
            payload=payload,
            allowed_actions={"review_topic", "block_topic"},
            instruction=(
                "This is a topic-triage decision, not a requirement to run every declared topic. "
                "For every listed analysis, return exactly one action: review_topic only when the PICO-aligned "
                "graph inventory plausibly supports a useful quantitative answer at or above its stated independent-study threshold; "
                "otherwise return block_topic with a concise reason. Do not select Evidence and do not calculate statistics."
            ),
        )
        actions_by_id = {action.analysis_id: action for action in decision.actions}
        completed_actions: list[PlannerAction] = []
        for definition in plan.analyses:
            item = inventory_by_id[definition.analysis_id]
            action = actions_by_id.get(definition.analysis_id)
            if action is None:
                completed_actions.append(PlannerAction(
                    action="block_topic",
                    analysis_id=definition.analysis_id,
                    reason="Planner did not prioritize this declared topic for detailed review.",
                ))
            elif action.action == "review_topic" and item.candidate_study_count < definition.minimum_studies:
                completed_actions.append(PlannerAction(
                    action="block_topic",
                    analysis_id=definition.analysis_id,
                    reason=(
                        "Graph inventory is below the declared minimum independent-study threshold; "
                        "detailed review was not run."
                    ),
                ))
            else:
                completed_actions.append(action)
        return decision.model_copy(update={"actions": completed_actions})

    def plan_feasibility(self, memory) -> PlannerDecision:
        payload = memory.model_dump()
        return self._request(
            phase="feasibility",
            payload=payload,
            allowed_actions={"calculate_meta", "block_topic"},
            instruction=(
                "Use the memory counts to decide whether a deterministic meta-analysis calculation can run. "
                "Do not select Evidence. A topic below its minimum poolable-study count must be blocked."
            ),
        )

    def _request(self, *, phase: str, payload: dict, allowed_actions: set[str], instruction: str) -> PlannerDecision:
        from openai import OpenAI

        if not self.api_key:
            raise RuntimeError("LLM analysis planner needs OPENAI_API_KEY.")

        class RawAction(BaseModel):
            action: str
            analysis_id: str
            reason: str

        class RawDecision(BaseModel):
            actions: list[RawAction] = Field(default_factory=list)
            summary: str

        prompt = (
            "You are a meta-analysis tool planner. You receive a compact graph-derived inventory or execution "
            "memory, never raw source text. Your only role is to schedule the next tools. "
            f"{instruction}\n\n"
            "Return JSON only: {\"actions\":[{\"action\":...,\"analysis_id\":...,\"reason\":...}],"
            "\"summary\":...}. Use only supplied analysis IDs and allowed actions.\n\n"
            f"PHASE={phase}\nALLOWED_ACTIONS={sorted(allowed_actions)}\nPAYLOAD={json.dumps(payload, ensure_ascii=False)}"
        )
        # The analysis agents use an OpenAI-compatible JSON contract.  Chat
        # Completions is supported by a wider range of providers than the
        # Responses API, including the configured 5.6-luna endpoint.
        self.calls_made += 1
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=120.0,
        )
        response = create_chat_completion(
            client,
            request,
            component="Planner",
            max_retries=self.max_retries,
        )
        content = (response.choices[0].message.content or "").removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            raw = RawDecision.model_validate_json(content)
        except ValueError as exc:
            raise RuntimeError("LLM planner did not return valid planning JSON.") from exc
        topics = payload.get("topics", [])
        if isinstance(topics, list):
            known_ids = {item.get("analysis_id") for item in topics if isinstance(item, dict)}
        else:
            known_ids = set(topics.keys())
        actions = [
            PlannerAction(action=item.action, analysis_id=item.analysis_id, reason=item.reason)
            for item in raw.actions
            if item.action in allowed_actions and item.analysis_id in known_ids
        ]
        return PlannerDecision(
            phase=phase,
            actions=actions,
            summary=raw.summary,
            planner_mode=self.mode,
        )
