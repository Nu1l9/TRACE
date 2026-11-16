from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .retrieval import RetrievalCandidate


def generate_synthesis_report(
    callback,
    task: Dict[str, Any],
    decisions: List[Dict[str, Any]],
    max_chars_per_study: int = 3500,
) -> str:
    included = [
        item
        for item in decisions
        if (item.get("final_status") or item.get("status")) == "Include"
    ]
    evidence_blocks = []
    for item in included:
        evidence = item.get("fulltext_evidence") or []
        evidence_text = "\n".join(str(value.get("text") or "") for value in evidence)
        source = evidence_text or str(item.get("abstract") or "")
        evidence_blocks.append(
            f"Corpus ID: {item['corpus_id']}\nTitle: {item.get('title', '')}\nEvidence: {source[:max_chars_per_study]}"
        )
    prompt = f"""
You are writing an auditable systematic-review report from studies already screened against a PI/ECO protocol.

Research question: {task.get('Research_Question', '')}
Population: {task.get('Population') or 'NR'}
Intervention/Exposure: {task.get('Intervention') or task.get('Exposure') or 'NR'}
Comparison: {task.get('Comparison') or 'NR'}
Outcome: {task.get('Outcome') or 'NR'}

Included study evidence:
{'\n\n---\n\n'.join(evidence_blocks) if evidence_blocks else 'No studies were included.'}

Write Markdown with exactly these sections:
1. Eligibility criteria
2. Included studies (cite every study using its Corpus ID)
3. Conclusion (2-3 sentences; do not invent pooled statistics)
4. Key insights (bullet points)
5. Limitations

Do not cite excluded or unresolved studies. Explicitly state when evidence is insufficient.
"""
    return str(callback(prompt))


def write_run_output(
    output_dir: str | Path,
    task: Dict[str, Any],
    candidates: Iterable[RetrievalCandidate],
    decisions: List[Dict[str, Any]],
    included_ids: List[int],
    unresolved_ids: List[int],
    run_config: Dict[str, Any],
    generated_report: str | None = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_list = list(candidates)
    result = {
        "paper_id": int(task["ID"]),
        "included_paper_ids": included_ids,
        "unresolved_paper_ids": unresolved_ids,
        "retrieved_papers": [candidate.to_dict() for candidate in candidate_list],
        "screening_decisions": decisions,
        "run_config": run_config,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    by_id = {int(item["corpus_id"]): item for item in decisions}
    lines = [
        f"# MINERS review task {task['ID']}",
        "",
        f"## Research question\n\n{task.get('Research_Question', '')}",
        "",
        "## Protocol",
        "",
        f"- Population: {task.get('Population') or 'NR'}",
        f"- Intervention/Exposure: {task.get('Intervention') or task.get('Exposure') or 'NR'}",
        f"- Comparison: {task.get('Comparison') or 'NR'}",
        f"- Outcome: {task.get('Outcome') or 'NR'}",
        "",
        "## Included studies",
        "",
    ]
    for corpus_id in included_ids:
        item = by_id.get(corpus_id, {})
        lines.append(
            f"- [{corpus_id}] {item.get('title', '')} — {item.get('reason', '')}"
        )
    lines.extend(["", "## Unresolved studies", ""])
    for corpus_id in unresolved_ids:
        item = by_id.get(corpus_id, {})
        lines.append(
            f"- [{corpus_id}] {item.get('title', '')} — {item.get('reason', '')}"
        )
    lines.extend(
        [
            "",
            "## Run notes",
            "",
            "This is an auditable retrieval/screening report. Synthesis generation is intentionally separate.",
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text(generated_report or "\n".join(lines), encoding="utf-8")
    manual_review_path = output_dir / "manual_review.csv"
    manual_rows = []
    for corpus_id in unresolved_ids:
        item = by_id.get(corpus_id, {})
        manual_rows.append(
            {
                "paper_id": int(task["ID"]),
                "corpus_id": corpus_id,
                "pmid": item.get("pmid", ""),
                "title": item.get("title", ""),
                "retrieval_rank": item.get("retrieval_rank", ""),
                "has_sections": item.get("has_sections", False),
                "abstract": item.get("abstract", ""),
                "screening_reason": item.get("reason", ""),
                "manual_final_status": "",
                "manual_evidence": "",
                "manual_source_url": "",
            }
        )
    with manual_review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        columns = [
            "paper_id",
            "corpus_id",
            "pmid",
            "title",
            "retrieval_rank",
            "has_sections",
            "abstract",
            "screening_reason",
            "manual_final_status",
            "manual_evidence",
            "manual_source_url",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manual_rows)
    return {"results": results_path, "report": report_path, "manual_review": manual_review_path}
