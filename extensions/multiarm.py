"""Collapse reviewed, explicitly disjoint trial arms into one analysis record.

The extractor may emit the Cartesian product of intervention and control arms.
Those pairs share participants and must never be treated as independent studies.
Aggregation is opt-in through the analysis selection rule and leaves an audit
of every original row and arm.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


COUNT_FIELDS = ("intervention_events", "intervention_total", "control_events", "control_total")


class _HTMLRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def _source_table_arms(request: dict[str, Any], rows: list[dict[str, Any]], intervention, controls):
    """Verify every source arm against integer Death and Alive cells in one table."""
    source = rows[0]["source_location"]
    index = source.get("content_list_item_index")
    if type(index) is not int or index < 0:
        return None
    documents = {item.get("article_id"): item for item in request.get("documents", [])}
    path = (documents.get(rows[0]["article_id"], {}).get("paths") or {}).get("content_list")
    if not path:
        return None
    try:
        items = json.loads(Path(path).read_text(encoding="utf-8"))
        item = items[index]
    except (OSError, ValueError, IndexError, TypeError, KeyError):
        return None
    if item.get("type") != "table":
        return None
    body = str(item.get("table_body") or "")
    parser = _HTMLRows()
    parser.feed(body)
    if not parser.rows:
        return None
    header = parser.rows[0]
    death = next((row for row in parser.rows if row and re.fullmatch(r"deaths?|mortality", row[0], re.I)), None)
    alive = next((row for row in parser.rows if row and re.fullmatch(r"alive|survivors?", row[0], re.I)), None)
    if not death or not alive or len(header) < 4:
        return None
    source_intervention, source_controls = [], []
    for column, label in enumerate(header[1:], start=1):
        if column >= len(death) or column >= len(alive):
            continue
        d = re.match(r"^\s*(\d+)\s*(?:\(|$)", death[column])
        a = re.match(r"^\s*(\d+)\s*(?:\(|$)", alive[column])
        if not d or not a:
            continue
        value = int(d.group(1)), int(d.group(1)) + int(a.group(1))
        if re.search(r"\barm\s*\d+\b", label, re.I):
            source_intervention.append(value)
        else:
            source_controls.append(value)
    if (Counter(source_intervention) != Counter(intervention.values())
            or Counter(source_controls) != Counter(controls.values())):
        return None
    return "\n".join(str(part) for key in ("table_caption", "table_body", "table_footnote")
                     for part in (item.get(key) if isinstance(item.get(key), list) else [item.get(key)]) if part)


def _label(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("label", "")
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _aggregation_allowed(request: dict[str, Any], target_id: str) -> bool:
    specs = [request.get("analysis_profile", {}).get("primary_analysis_spec", {}),
             *request.get("analysis_profile", {}).get("secondary_outcomes", [])]
    for spec in specs:
        if spec.get("target_id") != target_id:
            continue
        rules = spec.get("selection_rules", [])
        if isinstance(rules, dict):
            rules = rules.get("rules", [])
        text = " ".join(str(rule).casefold() for rule in rules)
        if "disjoint" in text and ("merg" in text or "combin" in text or "aggregat" in text):
            return True
    return False


def _counts(raw: dict[str, Any]) -> tuple[int, int, int, int] | None:
    result = []
    for field in COUNT_FIELDS:
        value = raw.get(field)
        if type(value) is not int or value < 0:
            return None
        result.append(value)
    ai, ni, ac, nc = result
    return (ai, ni, ac, nc) if 0 < ni and 0 < nc and ai <= ni and ac <= nc else None


def aggregate_disjoint_binary_arms(request: dict[str, Any], ledger: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate a complete, consistently reviewed arm grid; otherwise leave it intact."""
    output = copy.deepcopy(ledger)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in output.get("candidates", []):
        source = row.get("source_location") or {}
        if row.get("data_family") != "binary_arm_counts" or not _aggregation_allowed(request, row.get("target_id", "")):
            continue
        if source.get("review_context", {}).get("evidence_role") not in {"main_analysis_candidate", "uncertain"}:
            continue
        if _counts(row.get("raw_data") or {}) is None or not source.get("source_quote"):
            continue
        key = (str(row.get("article_id")), str(row.get("study_id")), str(row.get("target_id")),
               _label(row.get("outcome")), str(row.get("estimand")), str(row.get("timepoint")),
               str(source.get("source_id")), str(source.get("content_list_item_index")) if source.get("content_list_item_index") is not None
               else hashlib.sha256(source["source_quote"].encode()).hexdigest())
        groups[key].append(row)

    remove: set[str] = set()
    append: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for rows in groups.values():
        intervention: dict[str, tuple[int, int]] = {}
        controls: dict[str, tuple[int, int]] = {}
        pairs: set[tuple[str, str]] = set()
        valid = True
        for row in rows:
            exposure, comparator = _label(row.get("exposure")), _label(row.get("comparison"))
            ai, ni, ac, nc = _counts(row["raw_data"])
            if not exposure or not comparator or exposure == comparator or (exposure, comparator) in pairs:
                valid = False
                break
            if (exposure in intervention and intervention[exposure] != (ai, ni)) or (comparator in controls and controls[comparator] != (ac, nc)):
                valid = False
                break
            intervention[exposure] = ai, ni
            controls[comparator] = ac, nc
            pairs.add((exposure, comparator))
        if not valid or len(rows) < 3 or len(intervention) + len(controls) < 3:
            continue
        if len(pairs) != len(intervention) * len(controls) or len(rows) != len(pairs):
            continue
        # Verify the complete arm count table. Older full-table quotations are
        # accepted only when every pair was positively reviewed; short quotes
        # and uncertain reviews require the original source table itself.
        source_table = _source_table_arms(request, rows, intervention, controls)
        quote = rows[0]["source_location"]["source_quote"].casefold()
        if source_table is None and (any(row["source_location"].get("review_context", {}).get("evidence_role") != "main_analysis_candidate" for row in rows)
                                     or "<tr" not in quote or "</table>" not in quote
                                     or not re.search(r"\b(?:arms?|groups?)\b", quote)):
            continue
        ids = sorted(row["candidate_id"] for row in rows)
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
        combined = copy.deepcopy(rows[0])
        combined["candidate_id"] = f"aggregated_binary_arms_{digest}"
        combined["raw_data"] = {
            "intervention_events": sum(events for events, _ in intervention.values()),
            "intervention_total": sum(total for _, total in intervention.values()),
            "control_events": sum(events for events, _ in controls.values()),
            "control_total": sum(total for _, total in controls.values()),
            "outcome_unit": rows[0]["raw_data"].get("outcome_unit", "events"),
        }
        combined["exposure"] = "Combined disjoint intervention arms"
        combined["comparison"] = "Combined disjoint comparator arms"
        combined["population_or_subgroup"] = f"{len(intervention)} intervention arms versus {len(controls)} comparator arms in one study"
        combined["source_location"]["arm_aggregation"] = {
            "method": "sum_consistent_disjoint_source_arms_once",
            "component_candidate_ids": ids,
            "intervention_arms": [{"label": label, "events": values[0], "total": values[1]} for label, values in sorted(intervention.items())],
            "comparator_arms": [{"label": label, "events": values[0], "total": values[1]} for label, values in sorted(controls.items())],
        }
        if source_table:
            combined["source_location"]["source_quote"] = source_table
        combined["source_location"]["review_context"] = {
            "evidence_role": "uncertain" if any(row["source_location"].get("review_context", {}).get("evidence_role") == "uncertain" for row in rows) else "main_analysis_candidate",
            "disposition": "needs_further_review" if any(row["source_location"].get("review_context", {}).get("evidence_role") == "uncertain" for row in rows) else "keep_as_valid_candidate",
            "reason": "Integer Death and Alive counts were verified against the original arm table; original timepoint concerns remain for final review." if source_table else "Derived from individually reviewed, source-linked disjoint arms; component rows remain in the aggregation audit.",
        }
        append.append(combined)
        remove.update(ids)
        audit.append({"derived_candidate_id": combined["candidate_id"],
                      "article_id": combined["article_id"],
                      "component_candidate_ids": ids,
                      "raw_data": combined["raw_data"],
                      "source_id": combined["source_location"].get("source_id")})
    if append:
        output["candidates"] = [row for row in output["candidates"] if row["candidate_id"] not in remove] + append
    return output, audit
