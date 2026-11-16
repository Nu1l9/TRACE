"""Input-bound checkpoints for nondeterministic analysis LLM stages.

The statistical calculation can fail after source annotation and topic review
have already completed.  A pipeline resume must reuse those successful
decisions instead of asking the model to make them again.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Activate the frozen analysis package when this module is imported directly
# by tests or maintenance tools, rather than through analysis_worker.
from extensions.common import ANALYSIS as _ANALYSIS_ROOT


SCHEMA = "analysis_llm_checkpoint/3.1"


def _review_has_valid_selection(result, candidates) -> bool:
    """A cached analysis decision may contain only unique current candidate IDs."""
    records = {record.evidence_id: record for group in candidates.groups for record in group.records}
    selected = result.selected_evidence_ids
    return len(selected) == len(set(selected)) and set(selected) <= records.keys()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _fingerprint(*values: Any) -> str:
    body = json.dumps(
        [_jsonable(value) for value in values],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_matching(path: Path, *, kind: str, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("kind") != kind
        or payload.get("status") != "completed"
        or payload.get("input_fingerprint") != fingerprint
        or not isinstance(payload.get("result"), dict)
    ):
        return None
    return payload


class CheckpointedRuntimeFacetAnnotator:
    """Reuse a completed source-facet annotation for identical inputs."""

    mode = "checkpointed_runtime_facets"

    def __init__(self, delegate: Any, checkpoint_path: Path):
        self.delegate = delegate
        self.checkpoint_path = Path(checkpoint_path)
        self.artifact_path = getattr(delegate, "artifact_path", None)
        self.calls_made = 0

    def annotate(self, plan: Any, records: list[Any]):
        from analysis_engine.runtime_facets import RuntimeFacetResult

        fingerprint = _fingerprint(plan, records)
        cached = _read_matching(
            self.checkpoint_path, kind="runtime_facets", fingerprint=fingerprint,
        )
        if cached is not None:
            result = RuntimeFacetResult.model_validate(cached["result"])
            result.save(self.artifact_path)
            print(
                f"[analysis-checkpoint] runtime facets reused for {len(records)} Evidence; model calls=0",
                flush=True,
            )
            return result

        result = self.delegate.annotate(plan, records)
        after = int(getattr(self.delegate, "calls_made", 0) or 0)
        # RuntimeFacetAnnotator resets its own counter at the start of every
        # annotate() call, so its post-call value is the count for this call.
        self.calls_made += max(0, after)
        if result.status == "completed":
            _write_atomic(self.checkpoint_path, {
                "schema_version": SCHEMA,
                "kind": "runtime_facets",
                "status": "completed",
                "input_fingerprint": fingerprint,
                "result": result.model_dump(mode="json"),
            })
        return result


class CheckpointedTopicReviewer:
    """Freeze each successful review round against its exact candidate set."""

    mode = "checkpointed_topic_reviewer"

    def __init__(self, delegate: Any, checkpoint_dir: Path):
        self.delegate = delegate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.calls_made = 0

    def _path(self, analysis_id: str, fingerprint: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", analysis_id).strip("._") or "analysis"
        return self.checkpoint_dir / f"{safe_id}.{fingerprint[:20]}.json"

    def review(self, analysis: Any, candidates: Any):
        from analysis_engine.topic_reviewer_graph import ExpandableTopicReview

        fingerprint = _fingerprint(analysis, candidates)
        path = self._path(analysis.analysis_id, fingerprint)
        cached = _read_matching(path, kind="topic_review", fingerprint=fingerprint)
        if cached is not None:
            result = ExpandableTopicReview.model_validate(cached["result"])
            if _review_has_valid_selection(result, candidates):
                print(
                    f"[analysis-checkpoint] topic review reused: {analysis.analysis_id}; "
                    f"selected={len(result.selected_evidence_ids)}; model calls=0",
                    flush=True,
                )
                return result

        before = int(getattr(self.delegate, "calls_made", 0) or 0)
        result = self.delegate.review(analysis, candidates)
        after = int(getattr(self.delegate, "calls_made", 0) or 0)
        self.calls_made += max(0, after - before)
        result = ExpandableTopicReview.model_validate(result.model_dump(mode="json"))
        if not _review_has_valid_selection(result, candidates):
            raise ValueError('Topic review selected an unknown or duplicate Evidence ID')
        _write_atomic(path, {
            "schema_version": SCHEMA,
            "kind": "topic_review",
            "status": "completed",
            "input_fingerprint": fingerprint,
            "analysis_id": analysis.analysis_id,
            "candidate_evidence_ids": sorted(
                record.evidence_id
                for group in candidates.groups
                for record in group.records
            ),
            "result": result.model_dump(mode="json"),
        })
        return result
