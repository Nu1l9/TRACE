"""Semantic reviewer can request a broader source graph before final adjudication."""

import json
from pydantic import BaseModel, Field

from .eligibility import selection_score
from .llm_transport import create_chat_completion
from .models import TopicReviewResult
from .topic_reviewer import LLMTopicReviewer, _make_result, build_topic_reviewer_prompt


class ExpandableTopicReview(TopicReviewResult):
    request_expansion: bool = False
    expansion_reason: str = ""


class GraphTopicReviewer(LLMTopicReviewer):
    def set_tool_feedback(self, analysis_id, feedback):
        self.tool_feedback = (analysis_id, feedback)

    def review(self, analysis, candidates):
        from openai import OpenAI

        class RawReview(BaseModel):
            selected_evidence_ids: list[str] = Field(default_factory=list)
            summary: str
            request_expansion: bool = False
            expansion_reason: str = ""

        if not self.api_key:
            raise RuntimeError("Semantic Reviewer needs an API key.")
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0, timeout=180.0)
        prompt = build_topic_reviewer_prompt(
            analysis, candidates.groups, hide_calculation_details=self.hide_calculation_details,
            allow_expansion=True,
        )
        feedback = getattr(self, "tool_feedback", None)
        if feedback and feedback[0] == analysis.analysis_id:
            prompt += (
                "\nQUANTITATIVE_TOOL_FEEDBACK=" + json.dumps(feedback[1], ensure_ascii=False)
                + "\nThese are actual per-record validation results from derive_pairwise_effects. "
                "Poolable means numerically computable, not clinically eligible or independent. "
                "Decide eligibility and competing records from the declared question as before. "
                "Do not infer numerical impossibility merely from a missing reported CI or a zero cell "
                "when the tool successfully derived the effect. The tool applies its supported "
                "transformations; their appropriateness still depends on the declared analysis. "
                "A failed validation is not proof of clinical ineligibility; state unresolved limitations."
            )
        request = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        print(f"[GraphReviewer] topic={analysis.analysis_id}: groups={len(candidates.groups)} "
              f"prompt_chars={len(prompt)}", flush=True)
        self.calls_made += 1
        response = create_chat_completion(
            client, request, component=f"GraphReviewer/{analysis.analysis_id}",
            max_retries=self.max_retries,
        )
        content = (response.choices[0].message.content or "").strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        raw = RawReview.model_validate_json(content)
        selected_ids = set(raw.selected_evidence_ids)

        selected = set()
        for group in candidates.groups:
            choices = [record for record in group.records if record.evidence_id in selected_ids]
            if choices:
                selected.add(max(choices, key=lambda record: selection_score(record, analysis)).evidence_id)
        result = _make_result(
            analysis, candidates, selected, summary=raw.summary, reviewer_mode="llm",
            enforce_source_resolutions=False,
        )
        return ExpandableTopicReview(
            **result.model_dump(), request_expansion=raw.request_expansion,
            expansion_reason=raw.expansion_reason,
        )
