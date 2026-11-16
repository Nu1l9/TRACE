"""Graph evidence-handoff compatibility checks for frozen pipeline contracts.

The archived extractor occasionally emits a target-qualified candidate without
copying its target ID.  Graph can reconcile only an exact, declared match; it
never infers a target merely because it is the primary analysis target.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from metapipeline.contracts import evidence_handoff as _frozen_evidence_handoff


def _text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("label", "description", "name"):
            if value.get(key) not in (None, ""):
                return str(value[key])
        return ""
    return "" if value is None else str(value)


def _canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value).strip()).casefold()


def reconcile_target_ids(request: dict[str, Any], submission: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a copied submission with only deterministic target-ID repairs.

    A repair requires both an exact normalized outcome match and an exact
    declared estimand match.  This preserves the frozen contract's rule that a
    missing target must never be guessed from the primary-analysis position.
    """
    repaired = copy.deepcopy(submission)
    targets = [item for item in request.get("protocol", {}).get("targets", []) if isinstance(item, dict)]
    registered = {str(item.get("target_id")) for item in targets if item.get("target_id")}
    audit: list[dict[str, Any]] = []
    for row in repaired.get("predictions", []):
        if not isinstance(row, dict) or row.get("target_id") in registered:
            continue
        outcome = _canonical(row.get("outcome"))
        estimand = str(row.get("estimand") or "").strip()
        matches = [
            target for target in targets
            if outcome
            and outcome == _canonical(target.get("outcome"))
            and estimand
            and estimand in {str(value) for value in target.get("estimands", []) if value}
        ]
        if len(matches) != 1:
            continue
        previous = row.get("target_id")
        target_id = str(matches[0]["target_id"])
        row["target_id"] = target_id
        audit.append({
            "prediction_id": row.get("prediction_id"),
            "article_id": row.get("article_id"),
            "previous_target_id": previous,
            "target_id": target_id,
            "basis": "exact_normalized_outcome_and_declared_estimand_match",
        })
    return repaired, audit


def evidence_handoff(request: dict[str, Any], screening: dict[str, Any], submission: dict[str, Any], review: dict[str, Any]):
    """Apply deterministic reconciliation, then delegate to the frozen handoff."""
    normalized, reconciliations = reconcile_target_ids(request, submission)
    ledger, audit = _frozen_evidence_handoff(request, screening, normalized, review)
    from .multiarm import aggregate_disjoint_binary_arms
    ledger, aggregations = aggregate_disjoint_binary_arms(request, ledger)
    audit["target_id_reconciliation"] = reconciliations
    audit["target_id_reconciliation_count"] = len(reconciliations)
    audit["disjoint_arm_aggregations"] = aggregations
    audit["candidate_count"] = len(ledger["candidates"])
    return ledger, audit


def install() -> None:
    """Install the Graph wrapper into the frozen runner without editing vendor code."""
    import metapipeline.pipeline as pipeline

    pipeline.evidence_handoff = evidence_handoff
