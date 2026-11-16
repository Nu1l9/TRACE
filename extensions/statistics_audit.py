"""Auditable checks for statistics emitted by the frozen Analysis component."""
from __future__ import annotations

import math
from pathlib import Path

from extensions.common import write


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def conventional_i_squared(q, k):
    """Return Cochran-Q I² without changing the estimator's raw output."""
    q = _number(q)
    try:
        k = int(k)
    except (TypeError, ValueError):
        return None
    if q is None or q <= 0 or k < 2:
        return None
    return max(0.0, (q - (k - 1)) / q * 100.0)


def audit_analysis_result(result):
    """Compare reported I² with its deterministic Q-based definition.

    Analysis remains the source of the original model result. This audit prevents a
    numerical convergence issue from silently appearing as no heterogeneity.
    """
    analyses = []
    for item in result.get("analyses", []):
        statistics = item.get("statistics") or {}
        analysis = item.get("analysis") or {}
        reported = _number(statistics.get("i_squared"))
        recomputed = conventional_i_squared(statistics.get("q"), statistics.get("k"))
        mismatch = reported is not None and recomputed is not None and abs(reported - recomputed) > 0.5
        try:
            degrees_of_freedom = max(0, int(statistics.get("k")) - 1)
        except (TypeError, ValueError):
            degrees_of_freedom = None
        analyses.append({
            "analysis_id": analysis.get("analysis_id"),
            "reported_i_squared": reported,
            "q": _number(statistics.get("q")),
            "degrees_of_freedom": degrees_of_freedom,
            "q_recomputed_i_squared": recomputed,
            "status": "warning_i_squared_mismatch" if mismatch else "consistent",
            "message": (
                "Reported I² differs from the conventional Q-based calculation; retain the raw fit, but present the Q-based value with this warning."
                if mismatch else "Reported I² agrees with the conventional Q-based calculation."
            ),
        })
    return {"schema_version": "statistics_quality_audit/1.0", "analyses": analyses}


def write_statistics_quality_audit(output, result):
    audit = audit_analysis_result(result)
    write(Path(output) / "statistics_quality_audit.json", audit)
    return audit
