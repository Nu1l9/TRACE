"""Reviewed source arms are summed once, only under an explicit disjoint-arm rule."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor/integration"),
                str(ROOT / "vendor/integration/vendor/evidence_analysis")]

from extensions.multiarm import aggregate_disjoint_binary_arms


def fixture():
    request = {"analysis_profile": {"primary_analysis_spec": {
        "target_id": "mortality", "selection_rules": ["Permit merging explicitly disjoint matching arms."]}}}
    quote = "<table><tr><td>Death</td><td>0</td><td>3</td><td>5</td><td>6</td></tr></table> Three distinct trial arms/groups."
    rows = []
    for i, (exposure, ai) in enumerate((("Ivermectin A", 0), ("Ivermectin B", 3))):
        for j, (control, ac) in enumerate((("Control A", 5), ("Control B", 6))):
            rows.append({
                "candidate_id": f"e{i}{j}", "article_id": "article_1", "study_id": "study_1",
                "target_id": "mortality", "data_family": "binary_arm_counts",
                "outcome": {"label": "All-cause mortality"}, "estimand": "risk_ratio",
                "timepoint": "day 28", "exposure": exposure, "comparison": control,
                "raw_data": {"intervention_events": ai, "intervention_total": 30,
                             "control_events": ac, "control_total": 30},
                "source_location": {"source_id": "Table 3", "source_quote": quote,
                                    "review_context": {"evidence_role": "main_analysis_candidate"}},
            })
    return request, {"candidates": rows}


class MultiArmHandoffTests(unittest.TestCase):
    def test_short_pair_quotes_and_uncertain_review_use_original_arm_table(self):
        request, ledger = fixture()
        table = ("<table><tr><td>Outcome</td><td>Control A</td><td>Control B</td>"
                 "<td>Arm 1</td><td>Arm 2</td></tr>"
                 "<tr><td>Alive</td><td>25</td><td>24</td><td>30</td><td>27</td></tr>"
                 "<tr><td>Death</td><td>5</td><td>6</td><td>0</td><td>3</td></tr></table>")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "content_list.json"
            path.write_text(json.dumps([{"type": "table", "table_caption": "Table 3",
                                         "table_body": table,
                                         "table_footnote": "Four distinct study groups."}]), encoding="utf-8")
            request["documents"] = [{"article_id": "article_1", "paths": {"content_list": str(path)}}]
            for index, row in enumerate(ledger["candidates"]):
                row["source_location"]["content_list_item_index"] = 0
                row["source_location"]["source_quote"] = f"Pair {index}: Death counts."
                row["source_location"]["review_context"]["evidence_role"] = "uncertain"
            output, audit = aggregate_disjoint_binary_arms(request, ledger)
        self.assertEqual(1, len(audit))
        self.assertEqual(1, len(output["candidates"]))
        row = output["candidates"][0]
        self.assertEqual((3, 60, 11, 60), tuple(row["raw_data"][key] for key in
                         ("intervention_events", "intervention_total", "control_events", "control_total")))
        self.assertIn("<table>", row["source_location"]["source_quote"])
        self.assertEqual("uncertain", row["source_location"]["review_context"]["evidence_role"])

    def test_independent_study_guard_rejects_duplicate_effects(self):
        from analysis_engine.pipeline_agent import _dependent_study_ids
        definition = SimpleNamespace(dependence_policy="one_effect_per_study_outcome",
                                     selection_rules=["Select at most one eligible effect per independent study for this analysis."])
        effects = [SimpleNamespace(study_id="study_1"), SimpleNamespace(study_id="study_1"),
                   SimpleNamespace(study_id="study_2")]
        self.assertEqual(["study_1"], _dependent_study_ids(definition, effects))
        self.assertEqual([], _dependent_study_ids(definition, effects[1:]))


    def test_typed_cleanup_preserves_outcome_table_after_references(self):
        root = Path(__file__).resolve().parents[1] / "vendor/integration/vendor/evidence_extraction/system"
        sys.path[:0] = [str(root), str(root / "runtime/typed")]
        from locator_extractor_agent.tools import static_clean_document
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "content_list.json"
            path.write_text(json.dumps([
                {"type": "text", "text": "Results"},
                {"type": "text", "text": "References"},
                {"type": "ref_text", "text": "Bibliography entry"},
                {"type": "table", "table_caption": "Table 6. Outcomes", "table_body": "Death: 5/36 versus 6/37"},
            ]), encoding="utf-8")
            catalog, _ = static_clean_document(SimpleNamespace(representations={"content_list": path}), remove_reference_items_only=True)
        self.assertTrue(any("Death: 5/36 versus 6/37" in source["content"] for source in catalog))
        self.assertFalse(any("Bibliography entry" in source["content"] for source in catalog))

    def test_locator_fallback_includes_explicit_arm_level_death_table(self):
        from extensions.resilient_extraction import _include_explicit_mortality_tables
        task = SimpleNamespace(raw={"extraction": {"targets": [{
            "outcome": "All-cause mortality", "accepted_data_families": ["binary_arm_counts"]}]}})
        catalog = [
            {"source_id": "prose", "source_type": "text", "content": "Thirteen deaths overall."},
            {"source_id": "table_6", "source_type": "table", "content":
                "<table><tr><td>Group</td><td>Ivermectin (n=36)</td><td>Placebo (n=37)</td></tr>"
                "<tr><td>Death, n (%)</td><td>5 (13.8)</td><td>6 (16.2)</td></tr></table>"},
        ]
        selected, added = _include_explicit_mortality_tables(task, catalog, [catalog[0]])
        self.assertEqual(["table_6"], added)
        self.assertEqual(["prose", "table_6"], [source["source_id"] for source in selected])

    def test_complete_disjoint_grid_becomes_one_study_record_with_provenance(self):
        request, ledger = fixture()
        original = copy.deepcopy(ledger)
        output, audit = aggregate_disjoint_binary_arms(request, ledger)
        self.assertEqual(ledger, original)
        self.assertEqual(1, len(output["candidates"]))
        row = output["candidates"][0]
        self.assertEqual({"intervention_events": 3, "intervention_total": 60,
                          "control_events": 11, "control_total": 60, "outcome_unit": "events"}, row["raw_data"])
        self.assertEqual(["e00", "e01", "e10", "e11"], audit[0]["component_candidate_ids"])
        self.assertEqual(2, len(row["source_location"]["arm_aggregation"]["intervention_arms"]))

    def test_missing_pair_or_conflicting_count_never_combines(self):
        request, ledger = fixture()
        ledger["candidates"].pop()
        self.assertEqual([], aggregate_disjoint_binary_arms(request, ledger)[1])
        request, ledger = fixture()
        ledger["candidates"][0]["raw_data"]["intervention_events"] = 1
        self.assertEqual([], aggregate_disjoint_binary_arms(request, ledger)[1])

    def test_requires_protocol_permission_and_positive_review(self):
        request, ledger = fixture()
        request["analysis_profile"]["primary_analysis_spec"]["selection_rules"] = []
        self.assertEqual([], aggregate_disjoint_binary_arms(request, ledger)[1])
        request, ledger = fixture()
        ledger["candidates"][0]["source_location"]["review_context"]["evidence_role"] = "uncertain"
        self.assertEqual([], aggregate_disjoint_binary_arms(request, ledger)[1])


if __name__ == "__main__":
    unittest.main()
