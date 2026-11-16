from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from .dataset import DocumentStore
from .paper_node_adapter import AbstractScreener, build_llm_callback

if TYPE_CHECKING:
    from .retrieval import RetrievalCandidate


def build_screening_criteria(task: Dict[str, Any], protocol_mode: str = "strict") -> Dict[str, List[str]]:
    include = []
    for name, value in (
        ("Population", task.get("Population")),
        ("Intervention", task.get("Intervention")),
        ("Exposure", task.get("Exposure")),
        ("Comparison", task.get("Comparison")),
        ("Outcome", task.get("Outcome")),
        ("Eligible study designs", task.get("Study_designs")),
    ):
        if value:
            include.append(f"{name}: {value}")
    exclude: List[str] = []
    if task.get("Study_designs"):
        include.append("Judge eligibility from the clinical question and study design. Missing extractable numbers alone do not establish a PICO mismatch.")
    if protocol_mode == "oracle":
        raw_include = task.get("inclusion_criteria")
        raw_exclude = task.get("exclusion_criteria")
        if raw_include:
            include.extend(value.strip() for value in str(raw_include).split(";") if value.strip())
        if raw_exclude:
            exclude.extend(value.strip() for value in str(raw_exclude).split(";") if value.strip())
    return {"include": include, "exclude": exclude}


class ProtocolScreener:
    """PICO screener with title/abstract first-pass and full-text resolution."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        protocol_mode: str = "strict",
    ):
        self.callback = build_llm_callback(model=model, api_key_env=api_key_env, base_url=base_url)
        self.protocol_mode = protocol_mode

    def _screener(self, task: Dict[str, Any]):
        return AbstractScreener(
            llm_callback=self.callback,
            topic=str(task.get("Research_Question") or task.get("Title") or ""),
            screening_criteria=build_screening_criteria(task, self.protocol_mode),
        )

    def screen_abstracts(
        self, task: Dict[str, Any], candidates: Iterable[RetrievalCandidate]
    ) -> List[Dict[str, Any]]:
        papers = [
            {
                "corpus_id": candidate.corpus_id,
                "pmid": candidate.pmid,
                "title": candidate.title,
                "abstract": candidate.abstract,
                "retrieval_rank": candidate.rank,
                "has_sections": candidate.has_sections,
            }
            for candidate in candidates
        ]
        return self._screener(task).screen_papers(papers)

    def resolve_unsure_with_fulltext(
        self,
        task: Dict[str, Any],
        decisions: List[Dict[str, Any]],
        store: DocumentStore,
        max_fulltext_chars: int = 12_000,
        rescreen_all_available: bool = False,
    ) -> List[Dict[str, Any]]:
        from .evidence import FullTextEvidenceRetriever

        screener = self._screener(task)
        evidence_retriever = FullTextEvidenceRetriever()
        output: List[Dict[str, Any]] = []
        for decision in decisions:
            abstract_status = decision.get("status", "Unsure")
            if abstract_status != "Unsure" and not rescreen_all_available:
                output.append(decision)
                continue
            corpus_id = int(decision["corpus_id"])
            sections = store.get_fulltext_sections(corpus_id)
            if not sections:
                if abstract_status != "Unsure":
                    output.append(decision)
                    continue
                output.append(
                    {
                        **decision,
                        "fulltext_status": "unavailable",
                        "final_status": "Unresolved",
                    }
                )
                continue
            evidence_query = "\n".join(
                [
                    str(task.get("Research_Question") or ""),
                    " ".join(build_screening_criteria(task, self.protocol_mode)["include"]),
                    f"Abstract screening decision ({abstract_status}): {decision.get('reason', '')}",
                    "study design population intervention exposure comparator outcome sample size follow-up",
                ]
            )
            evidence = evidence_retriever.retrieve(
                sections,
                evidence_query,
                top_k=6,
                max_total_chars=max_fulltext_chars,
            )
            pieces = [f"[{item.section.upper()} | score={item.score:.4f}]\n{item.text}" for item in evidence]
            evidence_text = "\n\n".join(pieces)
            fulltext_paper = {
                **decision,
                "abstract": (
                    f"ABSTRACT:\n{decision.get('abstract', '')}\n\n"
                    f"RETRIEVED FULL-TEXT EVIDENCE:\n{evidence_text}"
                ),
            }
            rescored = screener.screen_papers([fulltext_paper])[0]
            final_status = rescored.get("status", "Unsure")
            if final_status == "Unsure":
                final_status = "Unresolved"
            output.append(
                {
                    **rescored,
                    "abstract_status": abstract_status,
                    "fulltext_status": "screened",
                    "fulltext_evidence": [
                        {"section": item.section, "score": item.score, "text": item.text}
                        for item in evidence
                    ],
                    "final_status": final_status,
                }
            )
        for item in output:
            status = item.get("final_status") or item.get("status")
            if status not in ("Include", "Exclude", "Unresolved"):
                status = "Unresolved"
            item["final_status"] = status
        return output


def included_and_unresolved(
    decisions: Iterable[Dict[str, Any]], unsure_policy: str = "exclude"
) -> Tuple[List[int], List[int]]:
    included: List[int] = []
    unresolved: List[int] = []
    for item in decisions:
        status = item.get("final_status") or item.get("status")
        corpus_id = int(item["corpus_id"])
        if status == "Include":
            included.append(corpus_id)
        elif status != "Exclude":
            unresolved.append(corpus_id)
            if unsure_policy == "include":
                included.append(corpus_id)
    return included, unresolved
