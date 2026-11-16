from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_result(
    task: Dict[str, Any],
    result: Dict[str, Any],
    ks: Sequence[int] = (5, 10, 20, 50, 100, 200),
) -> Dict[str, Any]:
    ground_truth = {int(value) for value in task.get("matched_corpus_ids", [])}
    retrieved = [
        int(item["corpus_id"])
        for item in result.get("retrieved_papers", [])
        if item.get("corpus_id") is not None
    ]
    included = {int(value) for value in result.get("included_paper_ids", [])}
    metrics: Dict[str, Any] = {
        "paper_id": int(task["ID"]),
        "ground_truth_count": len(ground_truth),
        "retrieved_count": len(retrieved),
        "included_count": len(included),
    }
    for k in ks:
        top = set(retrieved[:k])
        tp = len(top & ground_truth)
        metrics[f"retrieval_recall@{k}"] = round(_safe_div(tp, len(ground_truth)), 6)
        metrics[f"retrieval_precision@{k}"] = round(_safe_div(tp, len(top)), 6)

    retrieved_set = set(retrieved)
    screening_was_run = result.get("run_config", {}).get("screen", True)
    if not screening_was_run:
        metrics.update(
            {
                "screening_tp": None,
                "screening_fp": None,
                "screening_fn_within_pool": None,
                "screening_tn": None,
                "screening_precision": None,
                "screening_recall_within_pool": None,
                "screening_f1_within_pool": None,
                "screening_accuracy": None,
                "end_to_end_inclusion_precision": None,
                "end_to_end_inclusion_recall": None,
                "end_to_end_inclusion_f1": None,
                "unresolved_count": None,
            }
        )
        return metrics
    pool_ground_truth = ground_truth & retrieved_set
    predicted_in_pool = included & retrieved_set
    tp = len(predicted_in_pool & pool_ground_truth)
    fp = len(predicted_in_pool - pool_ground_truth)
    fn_pool = len(pool_ground_truth - predicted_in_pool)
    tn = len(retrieved_set - pool_ground_truth - predicted_in_pool)
    screening_precision = _safe_div(tp, tp + fp)
    screening_recall = _safe_div(tp, tp + fn_pool)
    metrics.update(
        {
            "screening_tp": tp,
            "screening_fp": fp,
            "screening_fn_within_pool": fn_pool,
            "screening_tn": tn,
            "screening_precision": round(screening_precision, 6),
            "screening_recall_within_pool": round(screening_recall, 6),
            "screening_f1_within_pool": round(_f1(screening_precision, screening_recall), 6),
            "screening_accuracy": round(_safe_div(tp + tn, len(retrieved_set)), 6),
        }
    )
    end_tp = len(included & ground_truth)
    end_precision = _safe_div(end_tp, len(included))
    end_recall = _safe_div(end_tp, len(ground_truth))
    metrics.update(
        {
            "end_to_end_inclusion_precision": round(end_precision, 6),
            "end_to_end_inclusion_recall": round(end_recall, 6),
            "end_to_end_inclusion_f1": round(_f1(end_precision, end_recall), 6),
            "unresolved_count": len(result.get("unresolved_paper_ids", [])),
        }
    )
    return metrics


def aggregate_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key != "paper_id" and isinstance(value, (int, float))
        }
    )
    aggregate: Dict[str, Any] = {"num_papers": len(rows)}
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float))
        ]
        aggregate[key] = round(statistics.mean(values), 6) if values else None
        aggregate[f"{key}_std"] = (
            round(statistics.stdev(values), 6) if len(values) > 1 else 0.0
        )
    return aggregate


def load_result(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
