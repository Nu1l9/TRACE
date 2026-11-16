from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from dotenv import load_dotenv

from .llm_schemas import (
    AbstractScreeningResponse,
    FullTextScreeningResponse,
    ValidationError,
    parse_llm_json,
)


def build_llm_callback(
    model: str = "gpt-4o-mini",
    api_key_env: str = "OPENAI_API_KEY",
    base_url: Optional[str] = None,
    system_message: Optional[str] = None,
) -> Callable[[str], str]:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Screening requires the openai package. "
            "Install the MINERS package dependencies."
        ) from exc

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} before running with --screen.")

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    def callback(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_message
                    or "You are a careful systematic-review abstract screener. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    return callback


class AbstractScreener:
    """Protocol-aware abstract screener used by MINERS."""

    def __init__(
        self,
        llm_callback: Callable[[str], str],
        topic: str = "",
        screening_criteria: Optional[Dict[str, List[str]]] = None,
        decision_mode: str = "abstract",
    ):
        if decision_mode not in {"abstract", "fulltext_binary"}:
            raise ValueError("decision_mode must be 'abstract' or 'fulltext_binary'.")
        self.ask_llm = llm_callback
        self.topic = topic
        self.screening_criteria = screening_criteria or {"include": [], "exclude": []}
        self.decision_mode = decision_mode

    def screen_papers(self, papers: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        papers = list(papers)
        results: List[Dict[str, Any]] = []
        labels = {str(paper.get("content_label") or "Abstract") for paper in papers}
        label = labels.pop() if len(labels) == 1 else "mixed-content"
        print(f"[AI Screener] Screening {len(papers)} {label.lower()} record(s)", flush=True)
        for index, paper in enumerate(papers, start=1):
            if not paper.get("abstract"):
                results.append(
                    {
                        **paper,
                        "status": "Unsure",
                        "reason": f"No {paper.get('content_label', 'abstract').lower()} is available.",
                    }
                )
                continue
            print(
                f"[AI Screener] {index}/{len(papers)} corpus_id={paper.get('corpus_id')}",
                flush=True,
            )
            raw = self.ask_llm(self._build_prompt(paper))
            decision = self._parse_response(raw)
            results.append({**paper, **decision})
        return results

    def _build_prompt(self, paper: Dict[str, Any]) -> str:
        required = list(self.screening_criteria.get("include", []))
        include = "\n".join(f"- {value}" for value in required)
        exclude = "\n".join(
            f"- {value}" for value in self.screening_criteria.get("exclude", [])
        )
        decision_policy = (
            """Decision policy:
- Decide eligibility from the supplied title and abstract using PICO as the review scope.
- Return \"Unsure\" only when the abstract does not provide enough information to
  decide, for example because key details are missing, ambiguous, or only partly match."""
            if self.decision_mode == "abstract"
            else """Full-text adjudication policy:
- This article reached this stage because its title/abstract decision was \"Unsure\".
- You are now reading supplied full text. Return only \"Include\" or \"Exclude\";
  never return \"Unsure\".
- Make the final eligibility decision yourself from the supplied full text,
  using the PICO criteria as the review scope."""
        )
        statuses = '"Include" | "Exclude" | "Unsure"' if self.decision_mode == "abstract" else '"Include" | "Exclude"'
        return f"""
Screen this paper for a systematic review.

Research topic:
{self.topic or "Not specified"}

Inclusion criteria:
{include or "- Use the research topic and PICO fields provided by the review task."}

Exclusion criteria:
{exclude or "- Exclude papers that clearly do not match the inclusion criteria."}

Paper:
Title: {paper.get("title", "")}
{paper.get("content_label", "Abstract")}: {paper.get("abstract", "")}

Return exactly one JSON object with this schema:
{{
  "status": {statuses},
  "reason": "short explanation of the final decision"
}}

{decision_policy}

Do not include markdown fences.
""".strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            schema = (
                AbstractScreeningResponse
                if self.decision_mode == "abstract"
                else FullTextScreeningResponse
            )
            data = parse_llm_json(response, schema)
        except (TypeError, ValueError, ValidationError):
            return {
                "status": "Unsure",
                "reason": "The model response did not match the screening response schema.",
                "response_parse_error": True,
            }
        return {
            "status": data.status,
            "reason": data.reason,
            "response_parse_error": False,
        }
