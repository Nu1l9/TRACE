import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, ValidationError

from extensions import resilient_extraction as resilient
from extensions import parallel_worker
from trace_runtime.contracts import ContractError, read, write


class FakeExtraction:
    def __init__(self, rows, unresolved_items=None):
        self.rows = rows
        self.unresolved_items = unresolved_items or []

    def model_dump(self):
        return {"rows": self.rows, "unresolved_items": self.unresolved_items}

    @classmethod
    def model_validate(cls, value):
        return cls(value["rows"], value.get("unresolved_items"))


class FakeLocation:
    def __init__(self, source_ids):
        self.source_ids = source_ids
        self.unresolved_location_terms = []


class FakeDocument:
    pass


class FakeTask:
    case_id = "case"
    task_id = "case_extraction"
    allowed_representations = ["content_list"]
    included_article_ids = ["article"]
    pico = {"population": "P"}
    execution_policy = {"max_article_retries": 0}
    raw = {
        "extraction": {"targets": [{"accepted_data_families": ["adjusted_ratio_effect"]}]},
    }
    documents = {"article": FakeDocument()}


class FakeController:
    @staticmethod
    def inspect_document(document):
        return {"document": "ok"}


class FakeTools:
    LEGACY_EFFECT_FAMILIES = {"adjusted_ratio_effect"}

    def __init__(self):
        self.locator_calls = 0
        self.extract_calls = 0

    def static_clean_document(self, document):
        return ([
            {"source_id": "s1", "source_type": "text", "page": 1, "content": "a" * 15000},
            {"source_id": "s2", "source_type": "text", "page": 2, "content": "b" * 15000},
        ], {"retained_items": 2})

    def locate_relevant_sources(self, task, catalog):
        self.locator_calls += 1
        return FakeLocation(["s1", "s2"]), list(catalog), []

    def extract_from_locations(self, task, sources, feedback):
        self.extract_calls += 1
        return FakeExtraction([{"source": sources[0]["source_id"]}])

    @staticmethod
    def normalize_and_verify_effects(task, extractions, catalog, article_id):
        return ([{"article_id": article_id, "prediction_id": f"p{index}"}
                 for index, extraction in enumerate(extractions, start=1)
                 for _ in extraction.rows], [])


class FakeModels:
    EffectExtraction = FakeExtraction
    DataFamilyExtraction = FakeExtraction


class FakeProviderError(RuntimeError):
    def __init__(self, status_code, message="provider error"):
        super().__init__(message)
        self.status_code = status_code


class GraphResilientExtractionTests(unittest.TestCase):
    def test_routed_legacy_effect_requires_the_exact_data_family(self):
        class LegacyEffect(BaseModel):
            model_config = ConfigDict(extra="forbid")
            outcome: str
            data_family: str | None = None

        tools = type(
            "LegacyTools",
            (),
            {
                "LEGACY_EFFECT_FAMILIES": {"adjusted_ratio_effect"},
                "DATA_FAMILY_TEMPLATE_MODELS": {
                    "adjusted_ratio_effect": LegacyEffect,
                },
            },
        )()
        envelope = resilient._single_family_output_model(
            tools, "adjusted_ratio_effect"
        )
        parsed = envelope.model_validate({
            "rows": [{"outcome": "death"}],
            "unresolved_items": [],
        })
        self.assertEqual("adjusted_ratio_effect", parsed.rows[0].data_family)
        with self.assertRaises(ValidationError):
            envelope.model_validate({
                "rows": [{"outcome": "death", "data_family": None}],
                "unresolved_items": [],
            })
        with self.assertRaises(ValidationError):
            envelope.model_validate({
                "rows": [{"outcome": "death", "data_family": "reported_ratio_effect"}],
                "unresolved_items": [],
            })

    def test_extraction_timeout_defaults_to_300_seconds(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(300.0, resilient._extract_timeout_seconds())
        with patch.dict(os.environ, {"TRACE_EXTRACT_TIMEOUT_SECONDS": "42.5"}, clear=True):
            self.assertEqual(42.5, resilient._extract_timeout_seconds())

    def test_typed_extraction_payload_omits_redundant_label_and_page(self):
        payload = resilient._compact_extraction_sources([
            {
                "source_id": "content_0001",
                "source_type": "text",
                "page": 4,
                "label": "This is duplicated preview text.",
                "content": "This is the full source text.",
            }
        ])
        self.assertEqual(
            [{
                "source_id": "content_0001",
                "source_type": "text",
                "content": "This is the full source text.",
            }],
            payload,
        )

    def test_batch_budget_uses_full_formatted_prompt_not_source_text(self):
        """Fixed contracts can push a small source batch over its true limit."""
        tools = type("PromptCountingTools", (), {"_contract": object()})()
        selected = [
            {"source_id": "s1", "source_type": "text", "content": "a" * 100},
            {"source_id": "s2", "source_type": "text", "content": "b" * 100},
        ]
        with patch.object(
            resilient,
            "_compact_prompt_chars",
            side_effect=lambda _tools, _task, sources, _is_typed: 19000 * len(sources),
        ):
            plan = resilient._batch_plan(
                selected, 20000, task=FakeTask(), tools=tools, is_typed=True
            )
        self.assertEqual([["s1"], ["s2"]], [batch["source_ids"] for batch in plan])
        self.assertEqual([19000, 19000], [batch["prompt_chars"] for batch in plan])
        self.assertTrue(all(batch["source_chars"] < 20000 for batch in plan))

    def test_extraction_uses_compact_prompt_adapter(self):
        expected = FakeExtraction([{"source": "s1"}])
        tools = type("CompactTools", (), {"_llm": object()})()
        sources = [{"source_id": "s1", "source_type": "text", "page": 1,
                    "label": "preview", "content": "body"}]
        with patch.object(resilient, "_compact_extractor", return_value=expected) as compact:
            result = resilient._call_extractor(tools, FakeTask(), FakeExtraction, sources, "", True, "openai_compatible")
        self.assertIs(expected, result)
        self.assertEqual(tools, compact.call_args.args[0])
        self.assertIsInstance(compact.call_args.args[1], FakeTask)
        self.assertEqual((FakeExtraction, sources, "", True, "openai_compatible"), compact.call_args.args[2:])

    def test_failed_large_batch_splits_on_resume_without_repeating_locator(self):
        tools = FakeTools()
        runtime = (lambda _: FakeTask(), FakeController, tools, FakeModels)
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            with patch.dict(os.environ, {'TRACE_EXTRACT_BATCH_CHARS':'40000'}), patch.object(
                    tools, 'extract_from_locations', side_effect=FakeProviderError(524)):
                with self.assertRaises(FakeProviderError):
                    resilient.run_article(self.request(), FakeTask(), 'article', work, runtime=runtime)
            with patch.dict(os.environ, {'TRACE_EXTRACT_BATCH_CHARS':'20000'}):
                result = resilient.run_article(self.request(), FakeTask(), 'article', work, runtime=runtime)
                self.assertEqual(2, len(result['extraction_submission']['predictions']))
                self.assertEqual(1, tools.locator_calls)
                state = read(work/'resilient_article_state.json')
                self.assertEqual(['s1','s2'], [sid for b in state['batches'] for sid in b['source_ids']])
                self.assertEqual(1, len(state['batch_plan_history']))
                resilient.run_article(self.request(), FakeTask(), 'article', work, runtime=runtime)
                self.assertEqual(2, tools.extract_calls)

    def request(self, workers=2):
        return {
            "case_id": "case",
            # These tests stub article preparation but still exercise the real
            # Resilient stage-input binding before scheduling work.
            "documents": [], "protocol": {}, "analysis_profile": {},
            "options": {"extraction_runtime": "baseline", "provider": "openai_compatible", "max_retries": 0, "max_workers": workers},
        }

    def test_completed_source_batches_are_reused(self):
        tools = FakeTools()
        runtime = (lambda _: FakeTask(), FakeController, tools, FakeModels)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            "TRACE_EXTRACT_BATCH_CHARS": "20000",
        }, clear=False):
            result = resilient.run_article(self.request(), FakeTask(), "article", Path(temp), runtime=runtime)
            self.assertEqual(2, len(result["extraction_submission"]["predictions"]))
            self.assertEqual(1, tools.locator_calls)
            self.assertEqual(2, tools.extract_calls)
            resumed = resilient.run_article(self.request(), FakeTask(), "article", Path(temp), runtime=runtime)
            self.assertEqual(2, len(resumed["extraction_submission"]["predictions"]))
            self.assertEqual(1, tools.locator_calls)
            self.assertEqual(2, tools.extract_calls)

    def test_failure_does_not_discard_other_article_checkpoints(self):
        task = type("Task", (), {"included_article_ids": ["bad", "good"], "task_id": "case_extraction"})()

        def fake_article(request, task, aid, work, runtime=None):
            if aid == "bad":
                raise RuntimeError("temporary endpoint error")
            return {
                "manifest": {"article_id": aid},
                "raw_system_output": {},
                "controller_state": {"status": "completed"},
                "extraction_submission": {"predictions": [{"article_id": aid, "prediction_id": aid}]},
                "issues": [],
            }

        request = self.request()
        with tempfile.TemporaryDirectory() as temp, patch.object(resilient, "_load_runtime", return_value=(lambda _: task, object(), object(), object())), patch.object(resilient, "run_article", side_effect=fake_article):
            with self.assertRaises(ContractError):
                parallel_worker.extract(request, Path(temp) / "config.json", Path(temp) / "out")
            progress = read(Path(temp) / "out" / "extraction_progress.json")
            self.assertEqual(["good"], [row["article_id"] for row in progress["article_runs"]])
            failures = read(Path(temp) / "out" / "extraction_failures.json")
            self.assertEqual(["bad"], [row["article_id"] for row in failures["failures"]])

    def test_provider_capacity_and_outage_are_checkpointed_not_retried_inline(self):
        capacity = FakeProviderError(429, "gateway_concurrency_limit")
        gateway_rate_limit = FakeProviderError(
            503,
            "model:rate_limited: You are sending requests to this model too frequently.",
        )
        outage = FakeProviderError(524, "origin response timeout")
        self.assertFalse(resilient.is_retryable_provider_error(capacity))
        self.assertFalse(resilient.is_retryable_provider_error(gateway_rate_limit))
        self.assertFalse(resilient.is_retryable_provider_error(outage))
        self.assertTrue(resilient.is_capacity_error(capacity))
        self.assertTrue(resilient.is_capacity_error(gateway_rate_limit))
        self.assertTrue(resilient.is_provider_outage_error(outage))

    def test_outage_defers_only_failed_article_and_runs_remaining_articles(self):
        task = type("Task", (), {"included_article_ids": ["outage", "good", "later"], "task_id": "case_extraction"})()
        calls = []

        def fake_article(request, task, aid, work, runtime=None):
            calls.append(aid)
            if aid == "outage":
                raise FakeProviderError(524, "origin response timeout")
            return {
                "manifest": {"article_id": aid},
                "raw_system_output": {},
                "controller_state": {"status": "completed"},
                "extraction_submission": {"predictions": [{"article_id": aid, "prediction_id": aid}]},
                "issues": [],
            }

        request = self.request(workers=1)
        with tempfile.TemporaryDirectory() as temp, patch.object(resilient, "_load_runtime", return_value=(lambda _: task, object(), object(), object())), patch.object(resilient, "run_article", side_effect=fake_article):
            with self.assertRaises(ContractError):
                parallel_worker.extract(request, Path(temp) / "config.json", Path(temp) / "out")
            failures = read(Path(temp) / "out" / "extraction_failures.json")
            progress = read(Path(temp) / "out" / "extraction_progress.json")
            self.assertEqual(["outage", "good", "later"], calls)
            self.assertEqual(["good", "later"], [row["article_id"] for row in progress["article_runs"]])
            self.assertEqual(["outage"], failures["deferred_article_ids"])
            self.assertEqual([], failures["not_submitted_article_ids"])

    def test_normal_resume_skips_known_524_but_retries_it_on_request(self):
        task = type("Task", (), {"included_article_ids": ["deferred", "good"], "task_id": "case_extraction"})()

        def successful_article(calls):
            def run_article(request, task, aid, work, runtime=None):
                calls.append(aid)
                return {
                    "manifest": {"article_id": aid},
                    "raw_system_output": {},
                    "controller_state": {"status": "completed"},
                    "extraction_submission": {"predictions": [{"article_id": aid, "prediction_id": aid}]},
                    "issues": [],
                }
            return run_article

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            out.mkdir()
            write(out / "extraction_failures.json", {
                "case_id": "case",
                "failures": [{"article_id": "deferred", "error_type": "InternalServerError", "status_code": 524}],
            })
            calls = []
            with patch.object(resilient, "_load_runtime", return_value=(lambda _: task, object(), object(), object())), patch.object(resilient, "run_article", side_effect=successful_article(calls)):
                with self.assertRaises(ContractError):
                    parallel_worker.extract(self.request(workers=1), Path(temp) / "config.json", out)
            self.assertEqual(["good"], calls)
            self.assertEqual(["deferred"], read(out / "extraction_failures.json")["deferred_article_ids"])

            calls = []
            with patch.dict(os.environ, {"TRACE_RETRY_DEFERRED_524": "1"}, clear=False), patch.object(resilient, "_load_runtime", return_value=(lambda _: task, object(), object(), object())), patch.object(resilient, "run_article", side_effect=successful_article(calls)):
                parallel_worker.extract(self.request(workers=1), Path(temp) / "config.json", out)
            self.assertEqual(["deferred"], calls)
            self.assertTrue((out / "extraction_complete.json").is_file())
            self.assertFalse((out / "extraction_failures.json").exists())
            self.assertTrue((out / "extraction_failure_history.json").is_file())


if __name__ == "__main__":
    unittest.main()
