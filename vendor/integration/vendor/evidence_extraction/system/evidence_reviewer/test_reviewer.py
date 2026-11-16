from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evidence_reviewer.reviewer import (
    _compact_batch_payload,
    _compact_candidate,
    _review_contract,
    _article_batches,
    review_batch_with_retries,
)


class ReviewerBatchTests(unittest.TestCase):
    def test_batches_stay_within_article(self) -> None:
        predictions = [
            {"prediction_id": "a1", "article_id": "article_a"},
            {"prediction_id": "a2", "article_id": "article_a"},
            {"prediction_id": "a3", "article_id": "article_a"},
            {"prediction_id": "b1", "article_id": "article_b"},
        ]
        batches = _article_batches(predictions, 2)
        self.assertEqual(
            [[row["prediction_id"] for row in batch] for batch in batches],
            [["a1", "a2"], ["a3"], ["b1"]],
        )
        self.assertTrue(
            all(len({row["article_id"] for row in batch}) == 1 for batch in batches)
        )

    def test_rejects_single_item_batch_configuration(self) -> None:
        with self.assertRaises(ValueError):
            _article_batches([], 1)

    def test_retries_only_the_failed_batch(self) -> None:
        batch = [
            {"prediction_id": "a1", "article_id": "article_a"},
            {"prediction_id": "a2", "article_id": "article_a"},
        ]
        expected = object()
        with patch(
            "evidence_reviewer.reviewer.review_batch",
            side_effect=[RuntimeError("temporary"), RuntimeError("temporary"), expected],
        ) as mocked:
            result, attempts = review_batch_with_retries(
                task=None,
                predictions=batch,
                configured_model=None,
                max_retries=3,
                retry_delay_seconds=0,
            )
        self.assertIs(result, expected)
        self.assertEqual(attempts, 3)
        self.assertEqual(mocked.call_count, 3)

    def test_compact_candidate_excludes_raw_data_and_quote(self) -> None:
        compact = _compact_candidate(
            {
                "prediction_id": "a1",
                "article_id": "article_a",
                "raw_data": {"large": "payload"},
                "used_effect": {"effect_type": "HR", "value": 1.2},
                "source_location": {"source_quote": "x" * 2000},
            },
            "source_1",
        )
        self.assertNotIn("raw_data", compact)
        self.assertNotIn("source_quote", compact)
        self.assertEqual(compact["source_evidence_id"], "source_1")

    def test_shared_source_quote_is_sent_once_per_batch(self) -> None:
        predictions = [
            {
                "prediction_id": "a1",
                "source_location": {"source_quote": "same table row"},
            },
            {
                "prediction_id": "a2",
                "source_location": {"source_quote": "same table row"},
            },
            {
                "prediction_id": "a3",
                "source_location": {"source_quote": "other table row"},
            },
        ]
        source_evidence, candidates = _compact_batch_payload(predictions)
        self.assertEqual(len(source_evidence), 2)
        self.assertEqual(candidates[0]["source_evidence_id"], "source_1")
        self.assertEqual(candidates[1]["source_evidence_id"], "source_1")
        self.assertEqual(candidates[2]["source_evidence_id"], "source_2")

    def test_compact_contract_preserves_config_exposure_field(self) -> None:
        task = SimpleNamespace(
            pico={
                "population": "adults",
                "exposure": "weight variability",
                "comparator": "stable weight",
            },
            raw={"extraction": {"targets": []}},
            allowed_effect_types=["HR"],
            estimands=[],
            exclusion_concepts="none",
        )
        contract = _review_contract(task)
        self.assertEqual(
            contract["pico"]["exposure"]["description"], "weight variability"
        )


if __name__ == "__main__":
    unittest.main()
