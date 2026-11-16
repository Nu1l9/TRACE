from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .llm_schemas import RiskOfBiasEvidence, RiskOfBiasRoutingResponse, ValidationError, parse_llm_json


TOOL_METADATA: Dict[str, Dict[str, str]] = {
    "RoB 2": {
        "version": "2019-08-22",
        "scope": "Randomized trials; select the appropriate individual, cluster, or crossover variant.",
    },
    "ROBINS-I": {
        "version": "2016",
        "scope": "Non-randomized studies of interventions.",
    },
    "ROBINS-E": {
        "version": "2024-03-24",
        "scope": "Non-randomized exposure studies; the released tool is designed for follow-up/cohort studies.",
    },
    "QUADAS-3": {
        "version": "1.2 (2026)",
        "scope": "Diagnostic accuracy, screening, or disease-staging studies.",
    },
    "QUIPS": {
        "version": "2013",
        "scope": "Prognostic factor studies.",
    },
    "PROBAST+AI": {
        "version": "2025",
        "scope": "Prediction model development or validation studies.",
    },
}

@dataclass(frozen=True)
class RoutingRule:
    tool: str
    variant: str = ""


@dataclass(frozen=True)
class FullTextPage:
    """One page or source section sent verbatim to the routing model."""

    section: str
    text: str


def deterministic_tool_for(
    question_type: str, study_design: str, allocation: str
) -> Optional[RoutingRule]:
    """Map evidence-grounded study attributes to one official RoB tool.

    The mapping is intentionally conservative: a tool is returned only when
    both the review question and the study design support that tool. In
    particular, diagnostic, prognostic, and prediction-model labels alone are
    insufficient, and ROBINS-E is restricted to follow-up/cohort exposure
    studies because the supplied 2024 version has that scope.
    """

    if question_type == "prediction_model" and study_design == "prediction_model_study":
        return RoutingRule("PROBAST+AI")
    if question_type == "diagnostic_accuracy" and study_design == "diagnostic_accuracy":
        return RoutingRule("QUADAS-3")
    if question_type == "prognostic_factor" and study_design == "prognostic_factor_study":
        return RoutingRule("QUIPS")
    if question_type == "intervention_effect":
        if allocation == "randomized" and study_design in {
            "individual_rct",
            "cluster_rct",
            "crossover_rct",
        }:
            variant = {
                "individual_rct": "individual randomized parallel-group variant",
                "cluster_rct": "cluster-randomized variant",
                "crossover_rct": "crossover-trial variant",
            }[study_design]
            return RoutingRule("RoB 2", variant)
        if allocation == "nonrandomized" or study_design == "nonrandomized_intervention":
            return RoutingRule("ROBINS-I")
    if question_type == "exposure_effect" and study_design == "cohort":
        return RoutingRule("ROBINS-E")
    return None


def _normalise_text(value: str) -> str:
    return " ".join(value.lower().split())


def _page_ordered_fulltext(
    sections: Mapping[str, str], max_total_chars: int
) -> tuple[List[FullTextPage], bool]:
    """Return the supplied body in page order without relevance filtering."""

    def page_key(item: tuple[str, str]) -> tuple[int, str]:
        match = re.fullmatch(r"page_(\d+)", item[0])
        return (int(match.group(1)), item[0]) if match else (10**9, item[0])

    remaining = max(1, int(max_total_chars))
    pages: List[FullTextPage] = []
    truncated = False
    for section, raw_text in sorted(sections.items(), key=page_key):
        text = str(raw_text or "").strip()
        if not text:
            continue
        if len(text) > remaining:
            pages.append(FullTextPage(section, text[:remaining]))
            truncated = True
            break
        pages.append(FullTextPage(section, text))
        remaining -= len(text)
        if remaining == 0:
            truncated = True
            break
    return pages, truncated


class RiskOfBiasRecommendationAgent:
    """Recommend, but never silently apply, a RoB tool for an included study."""

    def __init__(self, callback: Callable[[str], str]):
        self.callback = callback

    @staticmethod
    def system_message() -> str:
        return (
            "You are a systematic-review methods assistant. Classify the study only "
            "from the supplied full text. Return valid JSON and never invent evidence."
        )

    def recommend(
        self,
        study: Mapping[str, Any],
        sections: Mapping[str, str],
        review_question: str = "",
        review_type: str = "auto",
        expected_designs: Iterable[str] = (),
        top_k: int = 8,
        max_total_chars: int = 120_000,
    ) -> Dict[str, Any]:
        expected_designs = list(dict.fromkeys(str(value) for value in expected_designs))
        study_id = str(study.get("study_id") or study.get("corpus_id") or study.get("title") or "unknown")
        base = {
            "study_id": study_id,
            "pmid": str(study.get("pmid") or ""),
            "title": str(study.get("title") or ""),
            "review_question": review_question,
            "review_type_constraint": review_type,
            "expected_designs": expected_designs,
        }
        if not sections:
            return {
                **base,
                "fulltext_status": "unavailable",
                "recommended_tool": None,
                "agent_recommended_tool": None,
                "tool_version": None,
                "tool_variant": "",
                "study_design": "unclear",
                "question_type": "other_or_unclear",
                "allocation": "unclear",
                "confidence": "low",
                "recommendation_state": "needs_fulltext",
                "reason": "Risk-of-bias tool routing requires full text; no local or supplied full text was available.",
                "evidence": [],
                "validation_flags": ["fulltext_unavailable"],
                "fulltext_pages_supplied": 0,
                "fulltext_chars_supplied": 0,
                "fulltext_truncated": False,
            }

        # ``top_k`` is retained for backward CLI/API compatibility. Routing
        # deliberately reads page-ordered full text rather than retrieved excerpts.
        _ = top_k
        fulltext_pages, truncated = _page_ordered_fulltext(sections, max_total_chars)
        if not fulltext_pages:
            return {
                **base,
                "fulltext_status": "available_but_empty",
                "recommended_tool": None,
                "agent_recommended_tool": None,
                "tool_version": None,
                "tool_variant": "",
                "study_design": "unclear",
                "question_type": "other_or_unclear",
                "allocation": "unclear",
                "confidence": "low",
                "recommendation_state": "needs_human_review",
                "reason": "The supplied full text did not contain extractable text.",
                "evidence": [],
                "validation_flags": ["empty_fulltext"],
                "fulltext_pages_supplied": 0,
                "fulltext_chars_supplied": 0,
                "fulltext_truncated": False,
            }

        raw = self.callback(
            self._build_prompt(study, fulltext_pages, review_question, review_type)
        )
        try:
            parsed = parse_llm_json(raw, RiskOfBiasRoutingResponse)
        except (TypeError, ValueError, ValidationError):
            return self._invalid_response(
                base, fulltext_pages, "invalid_routing_response_schema", truncated
            )
        return self._validate_response(base, parsed, fulltext_pages, truncated)

    @staticmethod
    def _build_prompt(
        study: Mapping[str, Any],
        fulltext_pages: List[FullTextPage],
        review_question: str,
        review_type: str,
    ) -> str:
        blocks = "\n\n".join(
            f"[{index}] SECTION: {item.section}\n{item.text}"
            for index, item in enumerate(fulltext_pages, start=1)
        )
        return f"""
Recommend a risk-of-bias tool for one already-included study. This is routing only,
not a risk-of-bias judgement. Use only the supplied full text. Do not infer an RCT,
cohort, exposure, diagnostic, prognostic, or prediction-model design without explicit
evidence. The study may be outside the current review scope.

Review question: {review_question or 'Not specified'}
Review type constraint: {review_type}
Study title: {study.get('title', '')}
Abstract: {study.get('abstract', '')}

Tool mapping:
- RoB 2: randomized intervention trials; identify individual, cluster, or crossover variant.
- ROBINS-I: non-randomized intervention studies.
- ROBINS-E: non-randomized exposure studies only when the design is a follow-up/cohort study.
- QUADAS-3: primary diagnostic-accuracy studies with an explicit index test and reference standard; this can include diagnosis, screening, or disease-staging studies.
- QUIPS: prognostic factor studies.
- PROBAST+AI: prediction model development or validation studies.
- null: no recommendation can be defended from the supplied evidence.

Full text, in original page order:
{blocks}

Return exactly one JSON object with this schema:
{{
  "question_type": "intervention_effect" | "exposure_effect" | "diagnostic_accuracy" | "prognostic_factor" | "prediction_model" | "other_or_unclear",
  "study_design": "individual_rct" | "cluster_rct" | "crossover_rct" | "nonrandomized_intervention" | "cohort" | "case_control" | "cross_sectional" | "diagnostic_accuracy" | "prognostic_factor_study" | "prediction_model_study" | "other" | "unclear",
  "allocation": "randomized" | "nonrandomized" | "not_applicable" | "unclear",
  "recommended_tool": "RoB 2" | "ROBINS-I" | "ROBINS-E" | "QUADAS-3" | "QUIPS" | "PROBAST+AI" | null,
  "confidence": "high" | "medium" | "low",
  "reason": "brief evidence-grounded explanation",
  "evidence": [
    {{"evidence_id": 1, "claim": "study_design" | "question_type" | "allocation" | "tool_rationale", "quote": "exact text copied from excerpt"}}
  ],
  "missing_information": ["facts required to recommend a tool"]
}}

Include at least one evidence item for study_design and question_type. Quotes must be
verbatim from the indicated excerpt. Return no Markdown fences.
""".strip()

    @staticmethod
    def _invalid_response(
        base: Dict[str, Any],
        fulltext_pages: List[FullTextPage],
        flag: str,
        truncated: bool,
    ) -> Dict[str, Any]:
        return {
            **base,
            "fulltext_status": "available",
            "recommended_tool": None,
            "agent_recommended_tool": None,
            "tool_version": None,
            "tool_variant": "",
            "study_design": "unclear",
            "question_type": "other_or_unclear",
            "allocation": "unclear",
            "confidence": "low",
            "recommendation_state": "needs_human_review",
            "reason": "The routing agent did not return a valid structured recommendation.",
            "evidence": [],
            "validation_flags": [flag],
            "evidence_chunks_considered": len(fulltext_pages),
            "fulltext_pages_supplied": len(fulltext_pages),
            "fulltext_chars_supplied": sum(len(item.text) for item in fulltext_pages),
            "fulltext_truncated": truncated,
        }

    def _validate_response(
        self,
        base: Dict[str, Any],
        parsed: RiskOfBiasRoutingResponse,
        fulltext_pages: List[FullTextPage],
        truncated: bool,
    ) -> Dict[str, Any]:
        question_type = parsed.question_type
        study_design = parsed.study_design
        allocation = parsed.allocation
        confidence = parsed.confidence
        expected = deterministic_tool_for(question_type, study_design, allocation)
        agent_tool = parsed.recommended_tool
        evidence, flags = self._verified_evidence(parsed.evidence, fulltext_pages)

        claims = {item["claim"] for item in evidence}
        if "study_design" not in claims:
            flags.append("missing_verified_design_evidence")
        if "question_type" not in claims:
            flags.append("missing_verified_question_type_evidence")
        if expected is None:
            flags.append("no_defensible_tool_mapping")
        elif agent_tool != expected.tool:
            flags.append("agent_tool_disagrees_with_rules")
        if confidence == "low":
            flags.append("low_confidence")

        review_type_constraint = base["review_type_constraint"]
        if review_type_constraint != "auto" and question_type != review_type_constraint:
            flags.append("question_type_conflicts_with_protocol")

        expected_designs = base["expected_designs"]
        if expected_designs and study_design not in expected_designs:
            flags.append("outside_predefined_design_scope")
            state = "outside_predefined_scope"
        elif expected is None or flags:
            state = "needs_human_review"
        else:
            state = "ready_for_confirmation"

        return {
            **base,
            "fulltext_status": "available",
            "recommended_tool": expected.tool if expected else None,
            "agent_recommended_tool": agent_tool,
            "tool_version": TOOL_METADATA[expected.tool]["version"] if expected else None,
            "tool_variant": expected.variant if expected else "",
            "study_design": study_design,
            "question_type": question_type,
            "allocation": allocation,
            "confidence": confidence,
            "recommendation_state": state,
            "reason": parsed.reason,
            "missing_information": list(dict.fromkeys(parsed.missing_information)),
            "evidence": evidence,
            "validation_flags": list(dict.fromkeys(flags)),
            "evidence_chunks_considered": len(fulltext_pages),
            "fulltext_pages_supplied": len(fulltext_pages),
            "fulltext_chars_supplied": sum(len(item.text) for item in fulltext_pages),
            "fulltext_truncated": truncated,
        }

    @staticmethod
    def _verified_evidence(
        raw_evidence: List[RiskOfBiasEvidence], chunks: List[FullTextPage]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        verified: List[Dict[str, Any]] = []
        flags: List[str] = []
        for item in raw_evidence:
            index = item.evidence_id - 1
            if not 0 <= index < len(chunks):
                flags.append("invalid_evidence_reference")
                continue
            claim = item.claim
            quote = item.quote
            if len(_normalise_text(quote)) < 12:
                flags.append("evidence_quote_too_short")
                continue
            source = chunks[index]
            if _normalise_text(quote) not in _normalise_text(source.text):
                flags.append("evidence_quote_not_verified")
                continue
            verified.append(
                {
                    "claim": claim,
                    "section": source.section,
                    "quote": quote,
                    "evidence_id": index + 1,
                    "source_page": index + 1,
                }
            )
        return verified, flags
