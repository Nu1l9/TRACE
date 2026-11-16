"""PICO screening followed by risk-of-bias tool routing.

Unlike :mod:`case_rob`, this module starts with a mixed set of candidate
articles. Every supplied primary-study PDF first receives title/abstract
screening against PICO, without exposing its evaluation label to the language
model. Only abstract-``Unsure`` articles receive full-text adjudication; only
final ``Include`` articles are sent to the risk-of-bias tool router.

The adapter targets a portable case-archive layout plus a separate,
topic-matched negative-PDF pack. Its archive reader intentionally
uses only the positive-document manifest and the explicit negative PDFs; it
never reads the target review, supplement, or ``gold/`` entries at runtime.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .case_rob import pdf_sections
from .paper_node_adapter import AbstractScreener
from .rob_routing import FullTextPage, RiskOfBiasRecommendationAgent
from .screening import build_screening_criteria


SCREENING_STATUSES = {"Include", "Exclude", "Unsure"}


@dataclass(frozen=True)
class CandidateDocument:
    """One supplied primary-study PDF and its private evaluation membership."""

    case_id: str
    candidate_id: str
    title: str
    source_role: str
    expected_label: str
    source_archive: str
    source_member: str
    pdf_path: Path
    abstract_source_path: Optional[Path] = None

    def public_manifest(self) -> Dict[str, Any]:
        """Return audit metadata; this data is never passed into an LLM prompt."""

        return {
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "title": self.title,
            "source_role": self.source_role,
            "source_archive": self.source_archive,
            "source_member": self.source_member,
            "abstract_source": str(self.abstract_source_path) if self.abstract_source_path else None,
        }


def _json_from_archive(archive_path: Path, case_id: str, relative_path: str) -> tuple[dict, str]:
    """Read exactly one case-relative JSON entry from a portable archive."""

    suffix = f"/cases/{case_id}/{relative_path.replace('\\', '/').lstrip('/')}"
    with zipfile.ZipFile(archive_path) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.replace("\\", "/").endswith(suffix)
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one {relative_path} for {case_id} in {archive_path}; found {len(matches)}."
            )
        with archive.open(matches[0]) as stream:
            try:
                return json.loads(stream.read().decode("utf-8")), matches[0]
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {archive_path}!{matches[0]}") from exc


def _first_case_json_from_archive(
    archive_path: Path,
    case_id: str,
    relative_paths: Iterable[str],
) -> tuple[dict, str]:
    """Load the first available, non-gold document manifest for a case."""

    errors: List[str] = []
    for relative_path in relative_paths:
        try:
            return _json_from_archive(archive_path, case_id, relative_path)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    searched = ", ".join(relative_paths)
    raise FileNotFoundError(
        f"No usable positive-document manifest for {case_id}; searched {searched}. "
        f"Details: {'; '.join(errors)}"
    )


def load_case_protocol(specs_zip: str | Path, case_id: str) -> Dict[str, Any]:
    """Load only the protocol fields that are safe and necessary for screening."""

    raw, _ = _json_from_archive(Path(specs_zip), case_id, "case.json")
    criteria = raw.get("criteria") or {}
    if not isinstance(criteria, Mapping):
        raise ValueError(f"{case_id} case.json has no usable criteria object.")
    pico = {
        "population": str(criteria.get("population") or ""),
        "intervention_or_exposure": str(criteria.get("intervention_or_exposure") or ""),
        "comparator": str(criteria.get("comparator") or ""),
        "outcomes": [str(value) for value in criteria.get("outcomes") or [] if str(value).strip()],
        "study_designs": [str(value) for value in criteria.get("study_designs") or [] if str(value).strip()],
    }
    if not any(
        [
            pico["population"],
            pico["intervention_or_exposure"],
            pico["comparator"],
            *pico["outcomes"],
        ]
    ):
        raise ValueError(f"{case_id} case.json has no usable PICO fields.")
    return {
        "case_id": str(raw.get("case_id") or case_id),
        "title": str(raw.get("title") or case_id),
        "review_question": str(raw.get("review_question") or ""),
        "review_type": str(raw.get("review_type") or ""),
        "pico": pico,
    }


def _case_prefix(member: str, case_id: str) -> str:
    marker = f"/cases/{case_id}/"
    normalized = member.replace("\\", "/")
    index = normalized.find(marker)
    if index < 0:
        raise ValueError(f"Archive member does not identify case {case_id}: {member}")
    return normalized[: index + len(marker) - 1]


def _safe_identifier(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "document"


def _copy_member(archive: zipfile.ZipFile, member: str, destination: Path) -> Path:
    """Copy a declared file member without extracting arbitrary archive content."""

    normalized = member.replace("\\", "/")
    if normalized.startswith("/") or ".." in Path(normalized).parts:
        raise ValueError(f"Unsafe archive member path: {member}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)
    return destination


def _repair_zip_member_name(name: str) -> str:
    """Recover UTF-8 names that a legacy ZIP producer marked as CP437."""

    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


def _archive_member_index(names: Iterable[str]) -> Dict[str, str]:
    """Map declared member names to their exact ZIP entry names."""

    index: Dict[str, str] = {}
    for name in names:
        index.setdefault(name, name)
        index.setdefault(_repair_zip_member_name(name), name)
    return index


def _is_declared_primary_report(entry: Mapping[str, Any], pdf_relative: str) -> bool:
    """Accept only manifest-declared study reports, never a target review or gold item."""

    if entry.get("gold_only"):
        return False
    role = str(entry.get("role") or "")
    document_id = str(entry.get("document_id") or entry.get("article_id") or "")
    if role:
        return role == "candidate_article" and pdf_relative.startswith(
            "source_documents/candidate_article/"
        )
    if document_id.startswith("review_"):
        return False
    return pdf_relative.startswith((
        "documents/primary_reports/",
        "source_documents/primary_reports/",
    ))


def discover_case_candidates(
    specs_zip: str | Path,
    positives_zip: str | Path,
    negatives_zip: str | Path,
    case_id: str,
    destination: str | Path,
) -> tuple[Dict[str, Any], List[CandidateDocument]]:
    """Materialise manifest-listed positive PDFs and explicit negative PDFs.

    ``expected_label`` is retained only in the returned objects and output
    records for offline evaluation.  It is deliberately absent from all
    screening and routing prompts.
    """

    protocol = load_case_protocol(specs_zip, case_id)
    positive_archive_path = Path(positives_zip).resolve()
    negative_archive_path = Path(negatives_zip).resolve()
    destination = Path(destination)
    manifest, manifest_member = _first_case_json_from_archive(
        positive_archive_path,
        case_id,
        (
            "inputs/extraction_document_paths.json",
            "inputs/end_to_end_document_paths.json",
            "documents/manifest.json",
        ),
    )
    root = _case_prefix(manifest_member, case_id)
    positive_docs = manifest.get("documents") or []
    if not isinstance(positive_docs, list):
        raise ValueError(f"{case_id} positive document manifest has no documents list.")

    candidates: List[CandidateDocument] = []
    positive_dir = destination / "positive_candidates"
    with zipfile.ZipFile(positive_archive_path) as archive:
        member_index = _archive_member_index(archive.namelist())
        for entry in sorted(positive_docs, key=lambda item: str(item.get("document_id") or "")):
            if not isinstance(entry, Mapping):
                continue
            paths = entry.get("paths") or {}
            pdf_relative = str(paths.get("pdf") or "").replace("\\", "/").lstrip("/")
            markdown_relative = str(paths.get("markdown") or "").replace("\\", "/").lstrip("/")
            document_id = str(entry.get("document_id") or entry.get("article_id") or "")
            if not document_id or not pdf_relative:
                raise ValueError(f"Invalid positive document entry in {manifest_member}.")
            if not _is_declared_primary_report(entry, pdf_relative):
                continue
            declared_member = f"{root}/{pdf_relative}"
            member = member_index.get(declared_member)
            if member is None:
                raise FileNotFoundError(f"Positive PDF listed by manifest is absent: {declared_member}")
            path = _copy_member(archive, member, positive_dir / f"{_safe_identifier(document_id)}.pdf")
            markdown_path = None
            if markdown_relative:
                markdown_member = member_index.get(f"{root}/{markdown_relative}")
                if markdown_member:
                    markdown_path = _copy_member(
                        archive,
                        markdown_member,
                        positive_dir / f"{_safe_identifier(document_id)}.md",
                    )
            candidates.append(
                CandidateDocument(
                    case_id=case_id,
                    candidate_id=document_id,
                    title=document_id,
                    source_role="positive_candidate",
                    expected_label="Include",
                    source_archive=str(positive_archive_path),
                    source_member=member,
                    pdf_path=path,
                    abstract_source_path=markdown_path,
                )
            )

    if not candidates:
        raise FileNotFoundError(f"No manifest-listed candidate PDFs found for {case_id}.")

    negative_dir = destination / "negative_candidates"
    with zipfile.ZipFile(negative_archive_path) as archive:
        negative_members = sorted(
            name
            for name in archive.namelist()
            if re.search(r"(?:^|/)excluded_articles_\d+/[^/]+/original\.pdf$", name, re.IGNORECASE)
        )
        if not negative_members:
            case_suffix = case_id.rsplit("_", 1)[-1]
            try:
                case_number = int(case_suffix)
            except ValueError as exc:
                raise ValueError(f"Case identifier must end in a number: {case_id}") from exc
            legacy_pattern = re.compile(
                rf"(?:^|/)Case_0*{case_number}/benchmark_excluded_articles/[^/]+\.pdf$",
                re.IGNORECASE,
            )
            negative_members = sorted(
                name for name in archive.namelist() if legacy_pattern.search(name)
            )
        if not negative_members:
            raise FileNotFoundError(
                "No negative PDFs were found. Expected either "
                "excluded_articles_N/*/original.pdf or "
                f"Case_{case_suffix}/benchmark_excluded_articles/*.pdf in {negative_archive_path}."
            )
        for index, member in enumerate(negative_members, start=1):
            normalized_member = Path(member.replace("\\", "/"))
            folder = normalized_member.parent.name
            label = folder if normalized_member.name.lower() == "original.pdf" else normalized_member.stem
            candidate_id = f"negative_{_safe_identifier(label)}"
            path = _copy_member(archive, member, negative_dir / f"{index:02d}_{_safe_identifier(label)}.pdf")
            candidates.append(
                CandidateDocument(
                    case_id=case_id,
                    candidate_id=candidate_id,
                    title=label,
                    source_role="negative_candidate",
                    expected_label="Exclude",
                    source_archive=str(negative_archive_path),
                    source_member=member,
                    pdf_path=path,
                )
            )
    return protocol, candidates


class FullTextProtocolScreener:
    """PICO title/abstract screen, then binary full-text adjudication for Unsure."""

    def __init__(
        self,
        callback: Callable[[str], str],
    ):
        self.callback = callback

    def screen(
        self,
        candidate_id: str,
        title: str,
        sections: Mapping[str, str],
        protocol: Mapping[str, Any],
        max_total_chars: int = 120_000,
        abstract_source_text: Optional[str] = None,
        max_fulltext_attempts: int = 3,
    ) -> Dict[str, Any]:
        base = {"candidate_id": candidate_id, "title": title}
        task = self._protocol_task(protocol)
        if not sections:
            return {
                **base,
                "abstract_status": "Unsure",
                "fulltext_status": "unavailable",
                "final_status": "Unsure",
                "model_status": None,
                "reason": "No readable full-text sections were available for protocol screening.",
                "validation_flags": ["fulltext_unavailable"],
                "evidence_chunks_considered": 0,
            }
        extracted_title, abstract_text, abstract_pages = self._title_and_abstract(
            sections,
            title,
            abstract_source_text=abstract_source_text,
        )
        abstract_screener = self._screener(task, decision_mode="abstract")
        abstract_decision = abstract_screener.screen_papers(
            [
                {
                    "corpus_id": candidate_id,
                    "title": extracted_title,
                    "abstract": abstract_text,
                    "content_label": "Abstract",
                }
            ]
        )[0]
        abstract_status = str(abstract_decision.get("status") or "Unsure")
        if abstract_status not in SCREENING_STATUSES:
            abstract_status = "Unsure"
        common = {
            **base,
            "title": extracted_title,
            "abstract_status": abstract_status,
            "abstract_status_reason": str(abstract_decision.get("reason") or ""),
            "abstract_text": abstract_text,
            "abstract_chars_supplied": len(abstract_text),
            "abstract_pages_examined": abstract_pages,
        }
        if abstract_status in {"Include", "Exclude"}:
            return {
                **common,
                "screening_stage": "abstract",
                "fulltext_status": "not_needed",
                "final_status": abstract_status,
                "model_status": abstract_status,
                "reason": str(abstract_decision.get("reason") or ""),
                "validation_flags": [],
                "evidence_chunks_considered": 0,
                "fulltext_chars_supplied": 0,
                "fulltext_pages_supplied": 0,
                "fulltext_truncated": False,
            }

        fulltext, truncated = self._fulltext_pages(sections, max_total_chars)
        if not fulltext:
            return {
                **common,
                "screening_stage": "fulltext_unavailable",
                "fulltext_status": "available_but_empty",
                "final_status": "Unsure",
                "model_status": None,
                "reason": "The supplied PDF did not contain extractable text.",
                "validation_flags": ["empty_fulltext"],
                "evidence_chunks_considered": 0,
                "fulltext_chars_supplied": 0,
                "fulltext_pages_supplied": 0,
                "fulltext_truncated": False,
            }
        fulltext_text = "\n\n".join(
            f"[{item.section.upper()}]\n{item.text}" for item in fulltext
        )
        fulltext_screener = self._screener(task, decision_mode="fulltext_binary")
        max_fulltext_attempts = max(1, int(max_fulltext_attempts))
        decision: Dict[str, Any] = {}
        last_exception: Optional[Exception] = None
        attempts = 0
        for attempts in range(1, max_fulltext_attempts + 1):
            try:
                decision = fulltext_screener.screen_papers(
                    [
                        {
                            "corpus_id": candidate_id,
                            "title": extracted_title,
                            "abstract": fulltext_text,
                            "content_label": "Full text",
                        }
                    ]
                )[0]
            except Exception as exc:
                last_exception = exc
                if attempts < max_fulltext_attempts:
                    continue
                break
            if not decision.get("response_parse_error"):
                break

        if last_exception is not None and not decision:
            return {
                **common,
                "screening_stage": "fulltext_error",
                "fulltext_status": "available",
                "final_status": "Unsure",
                "model_status": None,
                "reason": f"The screening model call failed after {attempts} attempts: "
                f"{type(last_exception).__name__}: {last_exception}",
                "validation_flags": ["screening_model_error"],
                "evidence_chunks_considered": len(fulltext),
                "fulltext_chars_supplied": sum(len(item.text) for item in fulltext),
                "fulltext_pages_supplied": len(fulltext),
                "fulltext_truncated": truncated,
                "fulltext_attempts": attempts,
            }
        status = str(decision.get("status") or "Unsure")
        if status not in {"Include", "Exclude"}:
            status = "Unsure"
        parse_failure = bool(decision.get("response_parse_error"))
        return {
            **common,
            "screening_stage": "fulltext_adjudication",
            "fulltext_status": "screened",
            "final_status": status,
            "model_status": status,
            "reason": str(decision.get("reason") or ""),
            "validation_flags": (
                [f"fulltext_response_not_parseable_after_{attempts}_attempts"]
                if parse_failure
                else ([f"fulltext_response_retried_{attempts}_times"] if attempts > 1 else [])
            ),
            "evidence_chunks_considered": len(fulltext),
            "fulltext_chars_supplied": len(fulltext_text),
            "fulltext_pages_supplied": len(fulltext),
            "fulltext_truncated": truncated,
            "fulltext_attempts": attempts,
        }

    @staticmethod
    def _protocol_task(protocol: Mapping[str, Any]) -> Dict[str, Any]:
        pico = protocol.get("pico") or {}
        pico = pico if isinstance(pico, Mapping) else {}
        return {
            # Screening uses PICO only. The review question is retained in the
            # protocol for downstream RoB routing, but is not sent here.
            "Research_Question": "",
            "Population": str(pico.get("population") or ""),
            "Exposure": str(pico.get("intervention_or_exposure") or ""),
            "Comparison": str(pico.get("comparator") or ""),
            "Outcome": "; ".join(str(value) for value in pico.get("outcomes") or []),
            "Study_designs": "; ".join(str(value) for value in pico.get("study_designs") or []),
        }

    def _screener(self, task: Mapping[str, Any], decision_mode: str) -> AbstractScreener:
        return AbstractScreener(
            llm_callback=self.callback,
            topic="",
            screening_criteria=build_screening_criteria(dict(task), protocol_mode="strict"),
            decision_mode=decision_mode,
        )

    @staticmethod
    def _title_and_abstract(
        sections: Mapping[str, str],
        fallback_title: str,
        max_abstract_chars: int = 6_000,
        abstract_source_text: Optional[str] = None,
    ) -> tuple[str, str, List[str]]:
        """Extract title/abstract locally; never send body text in first-pass screening."""

        def page_key(item: tuple[str, str]) -> tuple[int, str]:
            match = re.fullmatch(r"page_(\d+)", item[0])
            return (int(match.group(1)), item[0]) if match else (10**9, item[0])

        pages = [
            (section, str(text or "").strip())
            for section, text in sorted(sections.items(), key=page_key)
            if str(text or "").strip()
        ][:3]
        def extract(source: str, labels: List[str]) -> tuple[str, str, List[str]]:
            marker = re.search(
                r"(?im)^\s*(?:#+\s*)?(?:abstract|summary)\s*[:\-\u2013]?\s*(?=\S)",
                source,
            )
            if marker:
                preamble = source[: marker.start()]
                tail = source[marker.end() :]
            else:
                introduction = re.search(
                    r"(?im)^\s*(?:#+\s*)?(?:\d+(?:\.\d+)?\s*)?introduction\b", source
                )
                if introduction:
                    preamble = source[: introduction.start()]
                    tail = preamble
                else:
                    # Do not send arbitrary PDF front matter or body text as
                    # an "abstract". It enters the full-text branch instead.
                    preamble = source[:max_abstract_chars]
                    tail = ""
            ending = re.search(
                r"(?im)^\s*(?:#+\s*)?(?:keywords?|key\s*words?|(?:\d+(?:\.\d+)?\s*)?introduction)\s*[:.]?\s*",
                tail,
            )
            if ending:
                tail = tail[: ending.start()]
            abstract = re.sub(r"\s+", " ", tail).strip()[:max_abstract_chars]
            title = FullTextProtocolScreener._title_from_preamble(preamble, fallback_title)
            return title, abstract, labels

        if abstract_source_text and abstract_source_text.strip():
            markdown_result = extract(str(abstract_source_text), ["portable_markdown"])
            if markdown_result[1]:
                return markdown_result
            # Some publishers render an unlabelled abstract immediately after
            # title/authors. A Markdown paragraph boundary is reliable here:
            # use only the first substantial paragraph, never subsequent body
            # paragraphs, tables, or figures.
            blocks = [
                re.sub(r"\s+", " ", block).strip()
                for block in re.split(r"\n\s*\n", str(abstract_source_text))
            ]
            abstract_block = next(
                (
                    block
                    for block in blocks
                    if len(block) >= 240
                    and not block.startswith("#")
                    and not block.startswith("!")
                    and not block.startswith("<table")
                ),
                "",
            )
            if abstract_block:
                markdown_title = FullTextProtocolScreener._title_from_preamble(
                    str(abstract_source_text)[: str(abstract_source_text).find(abstract_block)],
                    fallback_title,
                )
                return markdown_title, abstract_block[:max_abstract_chars], ["portable_markdown_unlabelled"]
        source = "\n\n".join(text for _, text in pages)
        return extract(source, [section for section, _ in pages]) if source else (fallback_title, "", [])

    @staticmethod
    def _title_from_preamble(preamble: str, fallback_title: str) -> str:
        """Best-effort PDF title recovery without placing body text in stage one."""

        ignored = re.compile(
            r"^(?:received|accepted|published|copyright|open access|www\.|https?://|doi:|"
            r"correspondence|author information|scientific reports|journal of|volume\b|issue\b)",
            re.IGNORECASE,
        )
        candidates = []
        for raw_line in preamble.splitlines():
            line = re.sub(r"\s+", " ", raw_line).lstrip("#").strip(" -:;,")
            words = line.split()
            if (
                ignored.search(line)
                or len(words) < 4
                or len(line) > 360
                or "@" in line
                or re.search(r"\b(?:department|university|hospital|institute|school)\b", line, re.IGNORECASE)
                or re.search(r"\b(?:et al\.?|and)\b", line, re.IGNORECASE) and "," in line
            ):
                continue
            score = min(len(words), 24)
            if re.search(r"[.!?]$", line):
                score -= 4
            if re.search(r"\d{4}", line):
                score -= 3
            candidates.append((score, line))
        return max(candidates, default=(0, fallback_title), key=lambda item: item[0])[1]

    @staticmethod
    def _fulltext_pages(
        sections: Mapping[str, str], max_total_chars: int
    ) -> tuple[List[FullTextPage], bool]:
        """Preserve PDF page order rather than retrieving a subset of evidence chunks."""

        def page_key(item: tuple[str, str]) -> tuple[int, str]:
            match = re.fullmatch(r"page_(\d+)", item[0])
            return (int(match.group(1)), item[0]) if match else (10**9, item[0])

        remaining = max(1, max_total_chars)
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

def _screening_error(candidate: CandidateDocument, exc: Exception) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "title": candidate.title,
        "fulltext_status": "unreadable",
        "final_status": "Unsure",
        "model_status": None,
        "reason": f"Could not read supplied PDF: {type(exc).__name__}: {exc}",
        "validation_flags": ["pdf_read_error"],
        "evidence_chunks_considered": 0,
    }


def _route_one(
    agent: RiskOfBiasRecommendationAgent,
    candidate: CandidateDocument,
    sections: Mapping[str, str],
    protocol: Mapping[str, Any],
    args,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        recommendation = agent.recommend(
            {
                "study_id": candidate.candidate_id,
                "title": title or candidate.title,
                "abstract": "",
            },
            sections,
            review_question=str(protocol.get("review_question") or ""),
            review_type=args.review_type,
            expected_designs=args.expected_design or [],
            top_k=args.top_rob_evidence_chunks,
            max_total_chars=args.max_rob_fulltext_chars,
        )
    except Exception as exc:
        recommendation = {
            "study_id": candidate.candidate_id,
            "title": title or candidate.title,
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
            "reason": f"The routing model call failed: {type(exc).__name__}: {exc}",
            "missing_information": ["risk-of-bias routing model response"],
            "evidence": [],
            "validation_flags": ["routing_model_error"],
            "evidence_chunks_considered": 0,
        }
    return {
        **recommendation,
        "candidate_id": candidate.candidate_id,
        "source_role": candidate.source_role,
        "source_archive": candidate.source_archive,
        "source_member": candidate.source_member,
    }


def _public_screening_record(candidate: CandidateDocument, decision: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        **dict(decision),
        "case_id": candidate.case_id,
        "source_role": candidate.source_role,
        "source_archive": candidate.source_archive,
        "source_member": candidate.source_member,
        "expected_label": candidate.expected_label,
    }


def _read_optional_text(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _screening_evaluation(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Evaluate source-set membership only after the blinded model decisions exist."""

    rows = list(records)
    counts = Counter()
    for row in rows:
        expected_include = row.get("expected_label") == "Include"
        predicted_include = row.get("final_status") == "Include"
        if expected_include and predicted_include:
            counts["tp"] += 1
        elif expected_include:
            counts["fn"] += 1
        elif predicted_include:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "evaluation_label_source": "manifest-listed positive candidates and separately supplied negative candidate PDFs; labels were not shown to either LLM stage",
        "candidate_count": len(rows),
        "expected_include_count": sum(row.get("expected_label") == "Include" for row in rows),
        "expected_exclude_count": sum(row.get("expected_label") == "Exclude" for row in rows),
        "model_status_counts": dict(sorted(Counter(str(row.get("final_status") or "Unsure") for row in rows).items())),
        "confusion_matrix_include_vs_not_include": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "include_precision": precision,
        "include_recall": recall,
        "include_f1": (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None,
        "accuracy_with_unsure_counted_as_not_include": (tp + tn) / len(rows) if rows else None,
        "unresolved_count": sum(row.get("final_status") == "Unsure" for row in rows),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_manual_review_csv(
    path: Path,
    screening: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> None:
    """Write only work that still needs a person: Unsure or RoB confirmation."""

    routes = {item["candidate_id"]: item for item in recommendations}
    fields = [
        "case_id", "candidate_id", "title", "source_role", "source_archive", "source_member",
        "queue_type", "model_screening_status", "screening_reason",
        "recommended_rob_tool", "rob_confidence", "rob_state", "rob_reason",
        "reviewer_screening_status", "reviewer_confirmed_rob_tool", "reviewer_notes",
    ]
    rows: List[Dict[str, Any]] = []
    for item in screening:
        route = routes.get(item["candidate_id"])
        if item.get("final_status") == "Exclude":
            continue
        queue_type = "screening_unsure" if item.get("final_status") == "Unsure" else "rob_confirmation"
        rows.append(
            {
                "case_id": item["case_id"],
                "candidate_id": item["candidate_id"],
                "title": item.get("title", ""),
                "source_role": item.get("source_role", ""),
                "source_archive": item.get("source_archive", ""),
                "source_member": item.get("source_member", ""),
                "queue_type": queue_type,
                "model_screening_status": item.get("final_status", ""),
                "screening_reason": item.get("reason", ""),
                "recommended_rob_tool": route.get("recommended_tool", "") if route else "",
                "rob_confidence": route.get("confidence", "") if route else "",
                "rob_state": route.get("recommendation_state", "") if route else "",
                "rob_reason": route.get("reason", "") if route else "",
                "reviewer_screening_status": "",
                "reviewer_confirmed_rob_tool": "",
                "reviewer_notes": "",
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    protocol: Mapping[str, Any],
    screening: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    evaluation: Mapping[str, Any],
    args,
) -> None:
    status_counts = Counter(str(item.get("final_status") or "Unsure") for item in screening)
    tool_counts = Counter(str(item.get("recommended_tool") or "No recommendation") for item in recommendations)
    lines = [
        f"# {protocol.get('case_id')}: PICO screening and risk-of-bias routing",
        "",
        "Every candidate first underwent title/abstract PICO screening. Only abstract-Unsure "
        "candidates received binary full-text PICO adjudication. "
        "Only model-included studies were sent to risk-of-bias tool routing. "
        "Source-set labels were withheld from both LLM prompts and used only for the final local evaluation.",
        "",
        "## Protocol",
        "",
        f"- Review question: {protocol.get('review_question') or '-'}",
        f"- Model: `{getattr(args, 'model', 'not recorded')}`",
        f"- Base URL: `{getattr(args, 'base_url', None) or 'provider default'}`",
        "",
        "## Screening summary",
        "",
        f"- Candidates: {len(screening)}",
        f"- Include: {status_counts.get('Include', 0)}",
        f"- Exclude: {status_counts.get('Exclude', 0)}",
        f"- Unsure: {status_counts.get('Unsure', 0)}",
        f"- Include precision: {evaluation.get('include_precision')}",
        f"- Include recall: {evaluation.get('include_recall')}",
        "",
        "| Candidate | Screening result | RoB tool (included only) |",
        "|---|---|---|",
    ]
    routes = {item["candidate_id"]: item for item in recommendations}
    for item in screening:
        route = routes.get(item["candidate_id"])
        title = str(item.get("title") or item["candidate_id"]).replace("|", "\\|")
        lines.append(
            f"| {title} | {item.get('final_status')} | {route.get('recommended_tool') if route else '-'} |"
        )
    lines.extend(["", "## Routing summary", ""])
    lines.extend(f"- {tool}: {count}" for tool, count in sorted(tool_counts.items()))
    if not recommendations:
        lines.append("- No candidate met the strict inclusion rule, so no study was routed.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `input_manifest.json`: declared source documents and private local evaluation membership.",
            "- `screening_results.json`: abstract-stage and, where needed, full-text-adjudication decisions.",
            "- `risk_of_bias_recommendations.json`: recommendations only for screened-in studies.",
            "- `manual_review.csv`: Unsure screening decisions and RoB confirmation queue.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    output_dir: Path,
    protocol: Mapping[str, Any],
    candidates: List[CandidateDocument],
    screening: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    args,
) -> Dict[str, Any]:
    manifest = {
        "protocol": protocol,
        "candidates": [candidate.public_manifest() for candidate in candidates],
        "run_config": {
            "model": getattr(args, "model", None),
            "base_url": getattr(args, "base_url", None),
            "review_type": getattr(args, "review_type", None),
            "expected_design": getattr(args, "expected_design", None) or [],
            "screening_policy": "PICO title/abstract first; binary full-text adjudication only for abstract-Unsure records",
        },
    }
    evaluation = _screening_evaluation(screening) if screening else {}
    _write_json(output_dir / "input_manifest.json", manifest)
    _write_json(output_dir / "screening_results.json", {"protocol": protocol, "decisions": screening, "evaluation": evaluation})
    _write_json(output_dir / "risk_of_bias_recommendations.json", {"case_id": protocol["case_id"], "recommendations": recommendations})
    combined = {
        "protocol": protocol,
        "screening": screening,
        "risk_of_bias_recommendations": recommendations,
        "evaluation": evaluation,
    }
    _write_json(output_dir / "results.json", combined)
    _write_manual_review_csv(output_dir / "manual_review.csv", screening, recommendations)
    _write_report(output_dir / "report.md", protocol, screening, recommendations, evaluation, args)
    return evaluation


def _load_resume_records(output_dir: Path, case_id: str) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load completed records for a retry without exposing labels to an LLM."""

    path = output_dir / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resume: previous results are missing at {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot resume: invalid previous results at {path}.") from exc
    if str((payload.get("protocol") or {}).get("case_id") or "") != case_id:
        raise ValueError(f"Cannot resume: results at {path} belong to a different case.")
    screening = {
        str(item.get("candidate_id")): dict(item)
        for item in payload.get("screening") or []
        if isinstance(item, Mapping) and item.get("candidate_id")
    }
    recommendations = {
        str(item.get("candidate_id")): dict(item)
        for item in payload.get("risk_of_bias_recommendations") or []
        if isinstance(item, Mapping) and item.get("candidate_id")
    }
    return screening, recommendations


def run_case_screening_rob(
    args,
    screener: Optional[FullTextProtocolScreener],
    router: Optional[RiskOfBiasRecommendationAgent],
) -> Dict[str, Any]:
    """Run or validate a two-stage PICO screening + RoB routing experiment."""

    output_dir = Path(args.output_dir) / args.case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_screening: Dict[str, Dict[str, Any]] = {}
    resume_recommendations: Dict[str, Dict[str, Any]] = {}
    if getattr(args, "resume_unsure", False):
        resume_screening, resume_recommendations = _load_resume_records(output_dir, args.case_id)
    with tempfile.TemporaryDirectory(prefix=f"{args.case_id}_candidates_") as temporary:
        protocol, candidates = discover_case_candidates(
            args.specs_zip,
            args.positives_zip,
            args.negatives_zip,
            args.case_id,
            temporary,
        )
        if args.dry_run:
            _write_outputs(output_dir, protocol, candidates, [], [], args)
            return {
                "case_id": args.case_id,
                "dry_run": True,
                "positive_candidates": sum(item.expected_label == "Include" for item in candidates),
                "negative_candidates": sum(item.expected_label == "Exclude" for item in candidates),
                "output_dir": str(output_dir.resolve()),
            }
        if screener is None or router is None:
            raise ValueError("A screener and router are required unless --dry-run is used.")

        screening: List[Dict[str, Any]] = []
        recommendations: List[Dict[str, Any]] = []
        total = len(candidates)
        for index, candidate in enumerate(candidates, start=1):
            previous = resume_screening.get(candidate.candidate_id)
            if previous and previous.get("final_status") in {"Include", "Exclude"}:
                screening.append(previous)
                recommendation = resume_recommendations.get(candidate.candidate_id)
                if recommendation:
                    recommendations.append(recommendation)
                print(
                    f"[{args.case_id}] resume {index:>4}/{total:<2} | {candidate.candidate_id} "
                    f"| final={previous['final_status']}",
                    flush=True,
                )
                continue
            print(f"[{args.case_id}] abstract {index:>2}/{total:<2} | {candidate.candidate_id}", flush=True)
            try:
                sections = pdf_sections(
                    candidate.pdf_path,
                    max_pages=args.max_study_pages,
                    max_chars_per_page=args.max_chars_per_page,
                )
            except Exception as exc:
                decision = _screening_error(candidate, exc)
                sections = {}
            else:
                decision = screener.screen(
                    candidate.candidate_id,
                    candidate.title,
                    sections,
                    protocol,
                    max_total_chars=args.max_screening_fulltext_chars,
                    abstract_source_text=_read_optional_text(candidate.abstract_source_path),
                    max_fulltext_attempts=getattr(args, "max_fulltext_adjudication_attempts", 3),
                )
            public_decision = _public_screening_record(candidate, decision)
            screening.append(public_decision)
            print(
                f"[{args.case_id}] {'':13} | abstract={public_decision.get('abstract_status', 'not_run')} "
                f"| final={public_decision['final_status']} "
                f"| fulltext_pages={public_decision.get('fulltext_pages_supplied', 0)} "
                f"| attempts={public_decision.get('fulltext_attempts', 0)}",
                flush=True,
            )
            if public_decision["final_status"] == "Include":
                print(f"[{args.case_id}] route          | {candidate.candidate_id}", flush=True)
                recommendation = _route_one(
                    router,
                    candidate,
                    sections,
                    protocol,
                    args,
                    title=str(public_decision.get("title") or candidate.title),
                )
                recommendations.append(recommendation)
                print(
                    f"[{args.case_id}] {'':13} | tool={recommendation.get('recommended_tool') or 'none'} "
                    f"| state={recommendation.get('recommendation_state', 'unknown')}",
                    flush=True,
                )
            _write_outputs(output_dir, protocol, candidates, screening, recommendations, args)
        evaluation = _write_outputs(output_dir, protocol, candidates, screening, recommendations, args)
    return {
        "case_id": args.case_id,
        "candidate_count": len(candidates),
        "screening_status_counts": dict(sorted(Counter(item["final_status"] for item in screening).items())),
        "routed_count": len(recommendations),
        "evaluation": evaluation,
        "output_dir": str(output_dir.resolve()),
    }
