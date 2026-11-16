import unittest
from types import SimpleNamespace
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from extensions.resilient_extraction import _structured_family_extract, _check_reported_sd


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(gt=1)


class Output(BaseModel):
    rows: list[Row]
    unresolved_items: list[str]


def invalid_result():
    return {"rows": [{"n": 1}], "unresolved_items": []}


class Model:
    def __init__(self, results):
        self.results = iter(results)
        self.messages = []

    def with_structured_output(self, schema, **kwargs):
        assert schema is Output
        assert kwargs == {"method": "function_calling", "include_raw": True}
        return self

    def invoke(self, messages):
        self.messages.append(list(messages))
        return next(self.results)


class StructuredRepairTests(unittest.TestCase):
    def test_sd_placeholder_is_rejected_without_changing_value(self):
        payload = {"rows": [{"raw_data": {"reported_summary": {"version": {"mean": 1.8, "sd": 1e-6}}}}]}
        parsed = SimpleNamespace(model_dump=lambda: payload)
        with self.assertRaisesRegex(ValueError, "SD not present"):
            _check_reported_sd(parsed, '[{"content":"n=20; mean deviation 1.8 degrees. SD not reported."}]')
        self.assertEqual(payload["rows"][0]["raw_data"]["reported_summary"]["version"]["sd"], 1e-6)

    def test_sd_in_adjacent_source_table_is_accepted(self):
        parsed = SimpleNamespace(model_dump=lambda: {"sd": 1.1})
        _check_reported_sd(parsed, '[{"content":"Mean 3.2."},{"content":"Table: SD 1.10"}]')

    def call(self, model):
        return _structured_family_extract(
            model, Output, ChatPromptTemplate.from_messages([("user", "Source: {source}")]),
            {"source": "Individual measurements; no independent groups."},
            "independent_continuous_arms",
        )

    def test_repairs_invalid_rows_without_fabricating_n(self):
        model = Model([invalid_result(), Output(rows=[], unresolved_items=["s1: individual data"])])
        result = self.call(model)
        self.assertEqual(result.rows, [])
        feedback = model.messages[1][-1].content
        self.assertIn("rows.0.n", feedback)
        self.assertIn("never invent", feedback)
        self.assertEqual(model.messages[0][0], model.messages[1][0])

    def test_unrepaired_errors_remain_failures_after_three_calls(self):
        model = Model([invalid_result(), invalid_result(), invalid_result()])
        with self.assertRaises(ValidationError):
            self.call(model)
        self.assertEqual(len(model.messages), 3)

    def test_valid_rows_preserved(self):
        model = Model([Output(rows=[Row(n=20)], unresolved_items=[])])
        self.assertEqual(self.call(model).rows[0].n, 20)
        self.assertEqual(len(model.messages), 1)


if __name__ == "__main__":
    unittest.main()
