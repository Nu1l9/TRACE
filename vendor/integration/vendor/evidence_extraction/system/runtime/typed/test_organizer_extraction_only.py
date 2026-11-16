from __future__ import annotations

import sys
import unittest
from pathlib import Path


SNAPSHOT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SNAPSHOT_ROOT.parents[1]
sys.path.insert(0, str(SNAPSHOT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

from constrained_meta_agent.models import DocumentInput, LoadedTask
from locator_extractor_agent.extraction_models import (
    BinaryArmCounts,
    ContinuousArmSummary,
    DataFamilyExtraction,
    EffectExtraction,
    EventCountsWithPersonTime,
    ExtractedBinaryArmCounts,
    ExtractedContinuousArmSummary,
    ExtractedEffect,
    ExtractedEventCountsWithPersonTime,
)
from locator_extractor_agent.organizer import normalize_and_verify_effects


class OrganizerExtractionOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.article_id = "article_1"
        self.task = LoadedTask(
            case_id="test_case",
            task_id="test_task",
            topic="test",
            pico={},
            target_exposure="intervention",
            allowed_effect_types=["HR", "RR", "OR", "RD"],
            estimands=[],
            model_selection_policy="",
            exclusion_concepts="",
            allowed_representations=[],
            included_article_ids=[self.article_id],
            execution_policy={},
            documents={
                self.article_id: DocumentInput(
                    article_id=self.article_id,
                    document_id="document_1",
                    representations={},
                )
            },
            raw={
                "inputs": {"study_ids": {self.article_id: "study_1"}},
                "extraction": {"targets": []},
            },
        )
        self.catalog = [
            {
                "source_id": "content_0",
                "source_type": "text",
                "content": "Source evidence: 10 40 20 50 1.20 1.00 1.40 2.0 1.0 15 2.5 1.2 16 8 100 12 120 6.0",
            }
        ]

    def test_source_data_is_not_converted_to_an_effect(self) -> None:
        extraction = DataFamilyExtraction(
            rows=[
                ExtractedBinaryArmCounts(
                    outcome="binary outcome",
                    exposure="intervention",
                    comparator="control",
                    population_or_subgroup="all",
                    estimand="binary",
                    model_name="source table",
                    source_id="content_0",
                    source_quote="Source evidence",
                    raw_data=BinaryArmCounts(
                        intervention_events=10,
                        intervention_total=40,
                        control_events=20,
                        control_total=50,
                        outcome_unit="participants",
                    ),
                ),
                ExtractedContinuousArmSummary(
                    outcome="continuous outcome",
                    exposure="intervention",
                    comparator="control",
                    population_or_subgroup="all",
                    estimand="continuous",
                    model_name="source table",
                    source_id="content_0",
                    source_quote="Source evidence",
                    raw_data=ContinuousArmSummary(
                        intervention_mean=2.0,
                        intervention_sd=1.0,
                        intervention_total=15,
                        control_mean=2.5,
                        control_sd=1.2,
                        control_total=16,
                        outcome_unit="score",
                        effect_orientation="intervention_minus_control",
                    ),
                ),
                ExtractedEventCountsWithPersonTime(
                    outcome="rate outcome",
                    exposure="intervention",
                    comparator="control",
                    population_or_subgroup="all",
                    estimand="rate",
                    model_name="source table",
                    source_id="content_0",
                    source_quote="Source evidence",
                    raw_data=EventCountsWithPersonTime(
                        intervention_episodes=8,
                        intervention_total=100,
                        control_episodes=12,
                        control_total=120,
                        follow_up_months=6.0,
                    ),
                ),
            ]
        )

        predictions, _ = normalize_and_verify_effects(
            self.task, [extraction], self.catalog, self.article_id
        )

        self.assertEqual(len(predictions), 3)
        for prediction in predictions:
            self.assertEqual(
                prediction["used_effect"],
                {
                    "effect_type": None,
                    "value": None,
                    "ci_lower": None,
                    "ci_upper": None,
                },
            )
        self.assertEqual(predictions[0]["raw_data"]["intervention_events"], 10)
        self.assertEqual(predictions[1]["raw_data"]["intervention_mean"], 2.0)
        self.assertEqual(predictions[2]["raw_data"]["follow_up_months"], 6.0)

    def test_source_reported_effect_is_preserved(self) -> None:
        extraction = EffectExtraction(
            rows=[
                ExtractedEffect(
                    outcome="reported outcome",
                    exposure="intervention",
                    comparator="control",
                    population_or_subgroup="all",
                    estimand="reported ratio",
                    effect_type="HR",
                    effect_value=1.2,
                    ci_lower=1.0,
                    ci_upper=1.4,
                    model_name="adjusted model",
                    adjustment="age",
                    source_id="content_0",
                    source_quote="Source evidence",
                )
            ]
        )

        predictions, _ = normalize_and_verify_effects(
            self.task, [extraction], self.catalog, self.article_id
        )

        self.assertEqual(
            predictions[0]["used_effect"],
            {"effect_type": "HR", "value": 1.2, "ci_lower": 1.0, "ci_upper": 1.4},
        )


if __name__ == "__main__":
    unittest.main()
