"""Case-package risk-of-bias tool routing.

This module deliberately works on a supplied ``Case_XXX`` directory rather
than on a retrieval benchmark corpus. It separates already included studies
from optional negative controls, extracts page-labelled PDF text, asks the
router for a recommendation, and writes both machine-readable and
human-readable audit records.
"""

from __future__ import annotations

import csv
import json
import re
import tempfile
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .rob_routing import RiskOfBiasRecommendationAgent


@dataclass(frozen=True)
class CaseStudy:
    """One primary-study PDF supplied with a case package.

    ``source_role`` is either ``included_study`` for a root-level PDF or
    ``negative_control`` for a PDF under ``benchmark_excluded_articles``.
    """

    case_id: str
    study_id: str
    path: Path
    source_role: str


@dataclass(frozen=True)
class CasePackage:
    """The review context and its supplied primary-study PDFs."""

    case_id: str
    root: Path
    case_pdf: Path
    included_studies: List[CaseStudy]
    negative_controls: List[CaseStudy]


# These labels are only for terminal and Markdown output.  The raw state keys
# remain unchanged in JSON and CSV so downstream scripts keep working.
STATE_EXPLANATIONS = {
    "ready_for_confirmation": "evidence and rules agree; reviewer can confirm the tool",
    "needs_human_review": "evidence, confidence, or the tool mapping needs reviewer attention",
    "outside_predefined_scope": "study design is outside the protocol design scope supplied to this run",
    "needs_fulltext": "no usable full text was supplied; obtain the PDF before routing",
}

CSV_COLUMNS = [
    "case_id",
    "study_id",
    "source_file",
    "source_role",
    "title",
    "fulltext_status",
    "recommended_tool",
    "agent_recommended_tool",
    "tool_version",
    "tool_variant",
    "study_design",
    "question_type",
    "allocation",
    "confidence",
    "recommendation_state",
    "reason",
    "evidence_chunks_considered",
    "missing_information",
    "validation_flags",
    "evidence",
    "reviewer_confirmed_tool",
    "reviewer_notes",
]


def _find_case_pdf(case_dir: Path) -> Path:
    """Return the single root-level review-context PDF for ``case_dir``."""

    candidates = [
        path
        for path in case_dir.glob("*.pdf")
        if re.fullmatch(r"case(?:\s+\d+)?", path.stem, flags=re.IGNORECASE)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{case_dir} must contain exactly one root-level case.pdf or Case <number>.pdf; found {len(candidates)}."
        )
    return candidates[0]


def _normalise_case_id(value: str) -> str:
    """Accept Case_001 as well as harmless shell/markdown separator variants."""

    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def discover_case_packages(path: str | Path, case_ids: Iterable[str] = ()) -> List[CasePackage]:
    """Discover selected case packages without treating controls as studies.

    ``path`` may be a single ``Case_XXX`` directory or its parent directory.
    Selection accepts shell-escaped variants such as ``Case\\_001`` because
    copied PowerShell commands sometimes preserve that backslash.
    """

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Case directory does not exist: {root}")
    is_single_case = root.is_dir() and root.name.lower().startswith("case_")
    candidate_dirs = (
        [root]
        if is_single_case
        else sorted(
            item
            for item in root.iterdir()
            if item.is_dir() and item.name.lower().startswith("case_")
        )
    )
    available_case_names = [item.name for item in candidate_dirs]
    wanted = {_normalise_case_id(value) for value in case_ids if str(value).strip()}
    if wanted:
        candidate_dirs = [
            item for item in candidate_dirs if _normalise_case_id(item.name) in wanted
        ]
    packages: List[CasePackage] = []
    for case_dir in candidate_dirs:
        case_pdf = _find_case_pdf(case_dir)
        included = [
            CaseStudy(case_dir.name, path.stem, path, "included_study")
            for path in sorted(case_dir.glob("*.pdf"))
            if path != case_pdf
        ]
        negative_dir = case_dir / "benchmark_excluded_articles"
        negative = (
            [
                CaseStudy(case_dir.name, path.stem, path, "negative_control")
                for path in sorted(negative_dir.glob("*.pdf"))
            ]
            if negative_dir.is_dir()
            else []
        )
        packages.append(
            CasePackage(
                case_id=case_dir.name,
                root=case_dir,
                case_pdf=case_pdf,
                included_studies=included,
                negative_controls=negative,
            )
        )
    if not packages:
        available = ", ".join(available_case_names) or "none"
        raise FileNotFoundError(
            "No matching Case_XXX directories were found. "
            f"Requested: {', '.join(str(value) for value in case_ids) or 'all'}; "
            f"available after matching: {available}."
        )
    return packages


def pdf_sections(
    path: str | Path,
    max_pages: int = 50,
    max_chars_per_page: int = 12_000,
) -> Dict[str, str]:
    """Read a PDF as page-labelled text sections without altering the source PDF."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Case-based RoB routing requires pypdf. Install MINERS dependencies."
        ) from exc
    reader = PdfReader(str(path))
    sections: Dict[str, str] = {}
    for index, page in enumerate(reader.pages[:max_pages], start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections[f"page_{index}"] = text[:max_chars_per_page]
    return sections


def _case_context(case_pdf: Path, max_pages: int, max_chars: int) -> str:
    sections = pdf_sections(case_pdf, max_pages=max_pages)
    text = "\n\n".join(f"[{section}]\n{value}" for section, value in sections.items())
    return text[:max_chars]


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _review_csv_row(item: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one recommendation while retaining list fields losslessly."""

    list_columns = {"missing_information", "validation_flags", "evidence"}
    reviewer_columns = {"reviewer_confirmed_tool", "reviewer_notes"}
    row = {
        column: item.get(column, "")
        for column in CSV_COLUMNS
        if column not in list_columns | reviewer_columns
    }
    row["missing_information"] = "; ".join(
        str(value) for value in item.get("missing_information") or []
    )
    row["validation_flags"] = "; ".join(
        str(value) for value in item.get("validation_flags") or []
    )
    row["evidence"] = json.dumps(item.get("evidence") or [], ensure_ascii=False)
    row["reviewer_confirmed_tool"] = ""
    row["reviewer_notes"] = ""
    return row


def _write_review_csv(path: Path, recommendations: List[Dict[str, Any]]) -> None:
    """Write an Excel-friendly reviewer worksheet with empty reviewer fields."""

    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for item in recommendations:
            writer.writerow(_review_csv_row(item))


def _markdown_cell(value: Any) -> str:
    return str(value or "-").replace("|", "\\|").replace("\n", " ").strip()


def _short_path(path: str | Path, max_chars: int = 72) -> str:
    value = Path(path).name
    return value if len(value) <= max_chars else f"{value[:max_chars - 3]}..."


def _count_field(
    recommendations: List[Dict[str, Any]], field: str, fallback: str
) -> Counter[str]:
    """Count a recommendation field while giving missing values a clear label."""

    return Counter(str(item.get(field) or fallback) for item in recommendations)


def _count_summary(counts: Counter[str]) -> str:
    """Format a compact deterministic count summary for console and Markdown."""

    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items())) or "none"


def _write_readable_summary(
    path: Path,
    package: CasePackage,
    recommendations: List[Dict[str, Any]],
    args,
) -> None:
    """Write the primary audit report intended for an evidence reviewer.

    JSON remains the canonical machine-readable record.  This report groups
    its important information in the order a reviewer normally needs it:
    overview, proposed tools, items requiring action, then evidence.
    """

    state_counts = _count_field(recommendations, "recommendation_state", "unknown")
    tool_counts = _count_field(recommendations, "recommended_tool", "no recommendation")
    expected_designs = ", ".join(args.expected_design or []) or "not specified"
    model = getattr(args, "model", "not recorded")
    base_url = getattr(args, "base_url", None) or "provider default"

    lines = [
        f"# {package.case_id}: Risk-of-bias tool routing",
        "",
        "This report recommends a risk-of-bias assessment tool for studies that were "
        "already supplied as included. It does not make a risk-of-bias judgement and "
        "does not change eligibility.",
        "",
        "## At a glance",
        "",
        f"- Case context: `{package.case_pdf.name}`",
        f"- Already included primary studies: {len(recommendations)}",
        f"- Ready for confirmation: {state_counts.get('ready_for_confirmation', 0)}",
        f"- Requires review or full text: {len(recommendations) - state_counts.get('ready_for_confirmation', 0)}",
        f"- Tool totals: {_count_summary(tool_counts)}",
        "",
        "## Run configuration",
        "",
        f"- Model: `{model}`",
        f"- API base URL: `{base_url}`",
        f"- Review-type constraint: `{args.review_type}`",
        f"- Expected designs: {expected_designs}",
        "",
        "## Recommendations",
        "",
        "| Study | Recommended tool | Study design | Confidence | State |",
        "|---|---|---|---|---|",
    ]
    if recommendations:
        for item in recommendations:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(_short_path(item.get("source_file") or item.get("title"))),
                        _markdown_cell(item.get("recommended_tool")),
                        _markdown_cell(item.get("study_design")),
                        _markdown_cell(item.get("confidence")),
                        _markdown_cell(item.get("recommendation_state")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| No root-level included-study PDFs were found | - | - | - | - |")

    lines.extend(["", "## Status guide", ""])
    for state, count in sorted(state_counts.items()):
        explanation = STATE_EXPLANATIONS.get(state, "state recorded for audit")
        lines.append(f"- `{state}` ({count}): {explanation}.")

    lines.extend(["", "## Human review queue", ""])
    pending = [
        item
        for item in recommendations
        if item.get("recommendation_state") != "ready_for_confirmation"
    ]
    if pending:
        lines.append(
            "Confirm the evidence below, then record the final tool and any rationale in "
            "`risk_of_bias_manual_review.csv`."
        )
        lines.append("")
        for item in pending:
            flags = ", ".join(item.get("validation_flags") or []) or "no specific flag"
            missing = ", ".join(item.get("missing_information") or [])
            next_step = f" Missing information: {missing}." if missing else ""
            lines.append(
                f"- `{_short_path(item.get('source_file') or item.get('title'))}`: "
                f"{flags}.{next_step}"
            )
    else:
        lines.append("- All supplied included studies are ready for reviewer confirmation.")

    lines.extend(["", "## Evidence audit", ""])
    for item in recommendations:
        lines.extend(
            [
                f"### {_short_path(item.get('source_file') or item.get('title'))}",
                "",
                f"- Recommendation: `{item.get('recommended_tool') or 'No recommendation'}`",
                f"- Agent recommendation: `{item.get('agent_recommended_tool') or 'No recommendation'}`",
                f"- Classification: `{item.get('question_type', 'other_or_unclear')}` / "
                f"`{item.get('study_design', 'unclear')}` / "
                f"`{item.get('allocation', 'unclear')}`",
                f"- Evidence chunks considered: {item.get('evidence_chunks_considered', 0)}",
                f"- Rationale: {item.get('reason') or 'Not provided.'}",
                "- Evidence:",
            ]
        )
        evidence = item.get("evidence") or []
        if evidence:
            for quote in evidence:
                lines.append(
                    f"  - `{quote.get('claim', 'claim')}`, {quote.get('section', 'unknown section')}: "
                    f"{quote.get('quote', '')}"
                )
        else:
            lines.append("  - No verified full-text evidence was retained.")
        missing = item.get("missing_information") or []
        if missing:
            lines.append(f"- Missing information: {', '.join(str(value) for value in missing)}")
        flags = item.get("validation_flags") or []
        if flags:
            lines.append(f"- Validation flags: {', '.join(str(value) for value in flags)}")
        lines.append("")

    lines.extend(
        [
            "## Summary",
            "",
            f"- Total included studies routed: {len(recommendations)}",
            f"- States: {_count_summary(state_counts)}",
            "- Machine-readable evidence: `risk_of_bias_recommendations.json`.",
            "- Reviewer worksheet: `risk_of_bias_manual_review.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommend_one(
    agent: RiskOfBiasRecommendationAgent,
    study: CaseStudy,
    case_context: str,
    args,
) -> Dict[str, Any]:
    try:
        sections = pdf_sections(
            study.path,
            max_pages=args.max_study_pages,
            max_chars_per_page=args.max_chars_per_page,
        )
    except Exception as exc:
        return {
            "case_id": study.case_id,
            "study_id": study.study_id,
            "source_file": str(study.path.resolve()),
            "source_role": study.source_role,
            "title": study.path.stem,
            "fulltext_status": "unreadable",
            "recommended_tool": None,
            "agent_recommended_tool": None,
            "tool_version": None,
            "tool_variant": "",
            "study_design": "unclear",
            "question_type": "other_or_unclear",
            "allocation": "unclear",
            "confidence": "low",
            "recommendation_state": "needs_human_review",
            "reason": f"Could not read the supplied PDF: {type(exc).__name__}: {exc}",
            "evidence": [],
            "validation_flags": ["pdf_read_error"],
        }
    recommendation = agent.recommend(
        {
            "study_id": study.study_id,
            "title": study.path.stem,
            "abstract": "",
        },
        sections,
        review_question=case_context,
        review_type=args.review_type,
        expected_designs=args.expected_design or [],
        top_k=args.top_evidence_chunks,
        max_total_chars=args.max_fulltext_chars,
    )
    return {
        **recommendation,
        "case_id": study.case_id,
        "source_file": str(study.path.resolve()),
        "source_role": study.source_role,
    }


def _recommend_studies(
    agent: RiskOfBiasRecommendationAgent,
    studies: List[CaseStudy],
    case_context: str,
    args,
    label: str,
) -> List[Dict[str, Any]]:
    """Route studies sequentially and emit progress useful for a long local run."""

    recommendations: List[Dict[str, Any]] = []
    total = len(studies)
    for index, study in enumerate(studies, start=1):
        print(
            f"[{study.case_id}] {label} {index:>3}/{total:<3} | reading {_short_path(study.path)}",
            flush=True,
        )
        recommendation = _recommend_one(agent, study, case_context, args)
        print(
            f"[{study.case_id}] {'':18} | tool={recommendation.get('recommended_tool') or 'none'} "
            f"| confidence={recommendation.get('confidence', 'low')} "
            f"| next={recommendation.get('recommendation_state', 'unknown')}",
            flush=True,
        )
        recommendations.append(recommendation)
    return recommendations


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    """Extract a user-supplied archive after rejecting path-traversal entries."""

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Archive contains an unsafe path: {member.filename}")
        archive.extractall(destination)


def _serialise_studies(studies: List[CaseStudy]) -> List[Dict[str, Any]]:
    """Convert dataclasses to JSON-safe manifest records with absolute paths."""

    return [asdict(study) | {"path": str(study.path.resolve())} for study in studies]


def _case_result_counts(recommendations: List[Dict[str, Any]]) -> Dict[str, str]:
    """Return concise stable summaries for the terminal completion message."""

    return {
        "tools": _count_summary(_count_field(recommendations, "recommended_tool", "none")),
        "states": _count_summary(
            _count_field(recommendations, "recommendation_state", "unknown")
        ),
    }


def _run_case_rob_from_dir(
    args, agent: RiskOfBiasRecommendationAgent, cases_dir: Path
) -> Dict[str, Any]:
    """Run the complete routing workflow for one or more discovered packages."""

    packages = discover_case_packages(cases_dir, args.case_id or [])
    output_root = Path(args.output_dir)
    summary: Dict[str, Any] = {
        "cases": 0,
        "included_studies": 0,
        "negative_controls": 0,
        "by_state": {},
    }
    for package in packages:
        print("", flush=True)
        print(f"=== {package.case_id}: risk-of-bias tool routing ===", flush=True)
        print(
            f"[{package.case_id}] Input: {len(package.included_studies)} included studies; "
            f"{len(package.negative_controls)} negative controls "
            f"(controls={'on' if args.include_negative_controls else 'off'}).",
            flush=True,
        )
        context = _case_context(package.case_pdf, args.max_case_pages, args.max_case_chars)
        print(
            f"[{package.case_id}] Review context: {len(context):,} characters extracted from "
            f"{package.case_pdf.name}.",
            flush=True,
        )
        included = _recommend_studies(agent, package.included_studies, context, args, "Included")
        controls = (
            _recommend_studies(
                agent, package.negative_controls, context, args, "Negative control"
            )
            if args.include_negative_controls
            else []
        )
        case_output = output_root / package.case_id
        case_output.mkdir(parents=True, exist_ok=True)
        _write_json(
            case_output / "case_manifest.json",
            {
                "case_id": package.case_id,
                "case_pdf": str(package.case_pdf.resolve()),
                "included_studies": _serialise_studies(package.included_studies),
                "negative_controls": _serialise_studies(package.negative_controls),
                "case_context_chars": len(context),
            },
        )
        _write_json(
            case_output / "risk_of_bias_recommendations.json",
            {
                "case_id": package.case_id,
                "case_pdf": str(package.case_pdf.resolve()),
                "review_type_constraint": args.review_type,
                "expected_designs": args.expected_design or [],
                "recommendations": included,
            },
        )
        _write_review_csv(case_output / "risk_of_bias_manual_review.csv", included)
        _write_readable_summary(case_output / "risk_of_bias_summary.md", package, included, args)
        if args.include_negative_controls:
            _write_json(
                case_output / "negative_control_recommendations.json",
                {"case_id": package.case_id, "recommendations": controls},
            )
        for item in included + controls:
            state = item["recommendation_state"]
            summary["by_state"][state] = summary["by_state"].get(state, 0) + 1
        summary["cases"] += 1
        summary["included_studies"] += len(included)
        summary["negative_controls"] += len(controls)
        counts = _case_result_counts(included)
        print(
            f"[{package.case_id}] Completed: tools ({counts['tools']}); "
            f"states ({counts['states']}).",
            flush=True,
        )
        print(f"[{package.case_id}] Outputs: {case_output.resolve()}", flush=True)
    return summary


def run_case_rob(args, agent: RiskOfBiasRecommendationAgent) -> Dict[str, Any]:
    """Accept either an unpacked directory or a ZIP case bundle for one run."""

    if args.cases_zip:
        archive_path = Path(args.cases_zip)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Case ZIP file does not exist: {archive_path}")
        with tempfile.TemporaryDirectory(prefix="miners_case_rob_") as temp_dir:
            extracted = Path(temp_dir)
            _safe_extract_zip(archive_path, extracted)
            return _run_case_rob_from_dir(args, agent, extracted)
    return _run_case_rob_from_dir(args, agent, Path(args.cases_dir))
