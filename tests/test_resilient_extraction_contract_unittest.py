import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from extensions import resilient_extraction as extraction
from metapipeline.contracts import ContractError, read, write


class BinaryRaw(BaseModel):
    intervention_events: int
    intervention_total: int
    control_events: int
    control_total: int


class BinaryRow(BaseModel):
    model_config = ConfigDict(extra='forbid')
    target_id: str | None = None
    data_family: str = 'binary_arm_counts'
    source_id: str
    raw_data: BinaryRaw


class Envelope(BaseModel):
    rows: list[BinaryRow]
    unresolved_items: list[str]


class StrictContinuousRaw(BaseModel):
    n: int = Field(gt=1)
    mean: float
    sd: float


class StrictContinuousRow(BaseModel):
    source_id: str
    raw_data: StrictContinuousRaw


class StrictContinuousEnvelope(BaseModel):
    rows: list[StrictContinuousRow]
    unresolved_items: list[str]


class Tools:
    DATA_FAMILY_TEMPLATE_MODELS = {'binary_arm_counts': BinaryRow}
    LEGACY_EFFECT_FAMILIES = set()
    DATA_FAMILY_ALIASES = {}
    @staticmethod
    def _contract(task):
        return {'pico': task.pico, 'targets': task.raw['extraction']['targets']}
    @staticmethod
    def _family_instructions(families):
        return 'Transcribe reported values; no derived effects.'


def task():
    return SimpleNamespace(pico={'population': 'adults'}, raw={'extraction': {'targets': [
        {'target_id': 'mortality', 'accepted_data_families': ['binary_arm_counts'],
         'source_constraints': ['No counts derived from Kaplan–Meier estimates.']}]}},
        documents={'article': object()}, execution_policy={'max_article_retries': 3})


def sources():
    return [{'source_id': 's1', 'source_type': 'table', 'content': 'Deaths: 5/19 versus 9/37.'}]


class FakeProviderError(RuntimeError):
    status_code = 524


class StructuredChain:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
    def invoke(self, messages):
        self.calls += 1
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class ResilientExtractionContractTests(unittest.TestCase):

    def test_all_declared_continuous_families_survive_document_routing(self):
        families = {'independent_continuous_arms', 'single_arm_continuous_summary'}
        article = [{'content': 'randomized comparison versus standard surgical control group'}]
        self.assertEqual(families, extraction._route_template_families(article, families))

    def test_schema_binds_target_and_requires_id(self):
        model = extraction._single_family_output_model(Tools(), 'binary_arm_counts', ['mortality'])
        row = {'data_family': 'binary_arm_counts', 'source_id': 's1', 'raw_data': {
            'intervention_events': 5, 'intervention_total': 19, 'control_events': 9, 'control_total': 37}}
        for value in ({**row}, {**row, 'target_id': 'unknown'}, {**row, 'target_id': None}):
            with self.assertRaises(ValidationError):
                model.model_validate({'rows': [value], 'unresolved_items': []})
        self.assertEqual([], model.model_validate({'rows': [], 'unresolved_items': ['Not reported']}).rows)
        self.assertEqual('mortality', model.model_validate({'rows': [{**row, 'target_id': 'mortality'}], 'unresolved_items': []}).rows[0].target_id)

    def test_binary_prompt_restores_source_constraints_without_schema_prose(self):
        model = extraction._single_family_output_model(Tools(), 'binary_arm_counts', ['mortality'])
        prompt, inputs = extraction._single_family_prompt(Tools(), task(), sources(), '', model, 'binary_arm_counts')
        text = '\n'.join(message.content for message in prompt.format_messages(**inputs))
        self.assertIn('Never reconstruct counts', text)
        self.assertIn('Kaplan–Meier', text)
        self.assertIn('source_constraints', text)
        self.assertNotIn('"$defs"', text)
        self.assertNotIn('reported_summary maps', text)

    def test_transport_failure_is_not_repeated_as_schema_repair(self):
        chain = StructuredChain([FakeProviderError('origin timed out')])
        model = SimpleNamespace(with_structured_output=lambda *a, **k: chain)
        prompt, inputs = extraction._single_family_prompt(Tools(), task(), sources(), '', Envelope, 'binary_arm_counts')
        with self.assertRaises(FakeProviderError):
            extraction._structured_family_extract(model, Envelope, prompt, inputs, 'binary_arm_counts')
        self.assertEqual(1, chain.calls)

    def test_arithmetic_failure_gets_one_supported_repair(self):
        raw = {'intervention_events': 20, 'intervention_total': 19, 'control_events': 9, 'control_total': 37}
        bad = {'rows': [{'target_id': 'mortality', 'source_id': 's1', 'raw_data': raw}], 'unresolved_items': []}
        good = {'rows': [], 'unresolved_items': ['Invalid count: inspect source']}
        chain = StructuredChain([bad, good])
        model = SimpleNamespace(with_structured_output=lambda *a, **k: chain)
        prompt, inputs = extraction._single_family_prompt(Tools(), task(), sources(), '', Envelope, 'binary_arm_counts')
        self.assertEqual([], extraction._structured_family_extract(model, Envelope, prompt, inputs, 'binary_arm_counts').rows)
        self.assertEqual(2, chain.calls)

    def test_mixed_numeric_failure_preserves_valid_rows_after_bounded_repair(self):
        bad_raw = {'intervention_events': 20, 'intervention_total': 19, 'control_events': 9, 'control_total': 37}
        good_raw = {**bad_raw, 'intervention_events': 5}
        result = {'rows': [{'target_id': 'mortality', 'source_id': 's1', 'raw_data': raw}
                           for raw in (good_raw, bad_raw)], 'unresolved_items': []}
        chain = StructuredChain([result, result, result])
        model = SimpleNamespace(with_structured_output=lambda *a, **k: chain)
        prompt, inputs = extraction._single_family_prompt(Tools(), task(), sources(), '', Envelope, 'binary_arm_counts')
        parsed = extraction._structured_family_extract(model, Envelope, prompt, inputs, 'binary_arm_counts')
        self.assertEqual(3, chain.calls)
        self.assertEqual(1, len(parsed.rows))
        self.assertEqual(5, parsed.rows[0].raw_data.intervention_events)
        self.assertIn('events exceed its denominator', parsed.unresolved_items[0])
        self.assertIn('"intervention_events": 20', parsed.unresolved_items[0])

    def test_schema_invalid_tool_rows_are_quarantined_without_losing_valid_rows(self):
        payload = {'rows': [
            {'source_id': 's1', 'raw_data': {'n': 20, 'mean': 5.0, 'sd': 1.2}},
            {'source_id': 'table-index', 'raw_data': {'n': 1, 'mean': 9.0, 'sd': 2.1}},
        ], 'unresolved_items': []}
        wrapped = {
            'raw': {'tool_calls': [{'args': payload}]},
            'parsed': None,
            'parsing_error': ValueError('n must be greater than 1'),
        }
        chain = StructuredChain([wrapped, wrapped, wrapped])
        model = SimpleNamespace(with_structured_output=lambda *a, **k: chain)
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([('user', 'Sources: {sources}')])
        parsed = extraction._structured_family_extract(
            model, StrictContinuousEnvelope, prompt,
            {'sources': json.dumps(sources())}, 'single_arm_continuous_summary',
        )
        self.assertEqual(3, chain.calls)
        self.assertEqual(['s1'], [row.source_id for row in parsed.rows])
        self.assertIn('Schema-invalid extraction row omitted', parsed.unresolved_items[0])
        self.assertIn('"n": 1', parsed.unresolved_items[0])

    def test_real_count_schemas_reject_negative_and_preserve_recurrence(self):
        import importlib
        from metapipeline.worker import load_extractor
        _, run = load_extractor({'options': {'extraction_runtime': 'typed'}})
        models = importlib.import_module(run.__module__.rsplit('.', 1)[0] + '.extraction_models')
        with self.assertRaises(ValidationError):
            models.BinaryArmCounts(intervention_events=-1, intervention_total=19,
                control_events=9, control_total=37, outcome_unit='deaths')
        with self.assertRaises(ValidationError):
            models.SingleArmEventCounts(events=10, total=9, outcome_unit='patients')
        raw = models.EventCountsWithPersonTime(intervention_episodes=40, intervention_total=19,
                control_episodes=60, control_total=37, follow_up_months=12)
        payload = {'rows': [{'source_id': 's1', 'data_family': 'event_counts_with_person_time', 'raw_data': raw.model_dump()}]}
        extraction._validate_numeric_rows(SimpleNamespace(model_dump=lambda:payload), json.dumps(sources()))
        raw.reported_rates_per_person_month = {'intervention': -0.5}
        payload['rows'][0]['raw_data'] = raw.model_dump()
        with self.assertRaisesRegex(ValueError, 'rates must be nonnegative'):
            extraction._validate_numeric_rows(SimpleNamespace(model_dump=lambda:payload), json.dumps(sources()))

    def test_nonpositive_ratio_never_becomes_a_quantitative_candidate(self):
        payload = {'rows': [{'source_id': 's1', 'data_family': 'time_to_event',
                            'effect_value': -0.1, 'ci_lower': -0.2, 'ci_upper': 0.4}]}
        with self.assertRaisesRegex(ValueError, 'ratio effects.*positive'):
            extraction._validate_numeric_rows(SimpleNamespace(model_dump=lambda:payload), json.dumps(sources()))

    def test_empty_diagnostic_set_is_recorded_without_numeric_repair_calls(self):
        import importlib
        from metapipeline.worker import load_extractor
        _, run = load_extractor({'options': {'extraction_runtime': 'typed'}})
        tools = importlib.import_module(run.__module__.rsplit('.', 1)[0] + '.tools')
        model_class = extraction._single_family_output_model(tools, 'diagnostic_four_cell_counts', ['accuracy'])
        row = {'target_id': 'accuracy', 'outcome': 'diagnosis', 'exposure': 'test', 'comparator': 'reference',
               'population_or_subgroup': 'test set', 'estimand': 'accuracy', 'source_id': 's1',
               'source_quote': 'No participants', 'raw_data': {'true_positive': 0, 'false_positive': 0,
               'false_negative': 0, 'true_negative': 0, 'test_set_label': 'empty test set'}}
        chain = StructuredChain([{'rows':[row], 'unresolved_items':[]}])
        model = SimpleNamespace(with_structured_output=lambda *a, **k: chain)
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([('user','Sources: {sources}')])
        parsed = extraction._structured_family_extract(model, model_class, prompt,
                    {'sources':json.dumps(sources())}, 'diagnostic_four_cell_counts')
        self.assertEqual(1, chain.calls)
        self.assertEqual([], parsed.rows)
        self.assertIn('at least one participant', parsed.unresolved_items[0])
        self.assertIn('empty test set', parsed.unresolved_items[0])

    def test_zero_sd_remains_raw_evidence_and_variance_controls_md_poolability(self):
        import importlib
        from metapipeline.worker import load_extractor
        _, run = load_extractor({'options': {'extraction_runtime': 'typed'}})
        models = importlib.import_module(run.__module__.rsplit('.', 1)[0] + '.extraction_models')
        from analysis_core.models import EvidenceRecord, ResolvedAnalysisProfile
        from analysis_engine.pairwise_tools import PairwiseEffectTool
        zero = models.NamedContinuousArm(label='intervention', n=10, mean=1.0, sd=0.0)
        other = models.NamedContinuousArm(label='control', n=10, mean=2.0, sd=2.0)
        with self.assertRaises(ValidationError):
            models.NamedContinuousArm(label='one person', n=1, mean=1.0, sd=0.0)
        profile = ResolvedAnalysisProfile(analysis_id='analysis', task_id='task', protocol_id='protocol',
            target_id='target', outcome='error', comparison='guide vs control',
            data_family='independent_continuous_arms', effect_type='MD', pooling_model='random_effects')
        record = EvidenceRecord(evidence_id='e1', article_id='a1', study_id='s1', target_id='target',
            data_family='independent_continuous_arms', raw_data={'intervention_arm':zero.model_dump(),
            'comparator_arm':other.model_dump()})
        result = PairwiseEffectTool().qualify(profile,[record])
        self.assertEqual('poolable',result.decisions[0].status)
        self.assertGreater(result.study_effects[0].standard_error,0)
        record.raw_data['comparator_arm']['sd'] = 0.0
        result = PairwiseEffectTool().qualify(profile,[record])
        self.assertEqual([], result.study_effects)
        self.assertEqual('insufficient_for_analysis',result.decisions[0].status)
        self.assertIn('non-positive within-study variance', result.decisions[0].reasons[0])
        self.assertEqual(0.0,record.raw_data['comparator_arm']['sd'])

    def test_sd_common_decimal_and_exponent_typography_is_recognized(self):
        for value, text in ((0.3,'SD .3'),(0.001,'SD 1e-3'),(0.001,'SD 1e−3')):
            payload = {'raw_data': {'sd':value}}
            with self.subTest(text=text):
                extraction._check_reported_sd(SimpleNamespace(model_dump=lambda:payload), json.dumps([{'content':text}]))

    def test_unrecognized_sd_typography_preserves_row_for_semantic_review(self):
        class ContinuousRow(BaseModel):
            source_id: str
            quantitative_usability: str = 'calculator_eligible'
            raw_data: dict[str,float]
        class ContinuousEnvelope(BaseModel):
            rows: list[ContinuousRow]
            unresolved_items: list[str]
        original = ContinuousEnvelope(rows=[ContinuousRow(source_id='s1', raw_data={'sd':0.3})], unresolved_items=[])
        result = extraction._annotate_sd_provenance(original, json.dumps([{'source_id':'s1','content':'SD 0 . 3'}]))
        self.assertEqual(0.3,result.rows[0].raw_data['sd'])
        self.assertEqual('needs_extraction',result.rows[0].quantitative_usability)
        self.assertIn('semantic review',result.unresolved_items[0])
        self.assertEqual('calculator_eligible',original.rows[0].quantitative_usability)

    def test_validated_family_is_reused_across_batch_retries(self):
        prompt, inputs = extraction._single_family_prompt(Tools(), task(), sources(), '', Envelope, 'binary_arm_counts')
        result = {'rows': [], 'unresolved_items': ['No counts reported']}
        chain = StructuredChain([result, result])
        model = SimpleNamespace(with_structured_output=lambda *a, **k: chain)
        with tempfile.TemporaryDirectory() as temp:
            token = extraction._FAMILY_CHECKPOINT_DIR.set(Path(temp))
            try:
                first = extraction._cached_family_extract(model, Envelope, prompt, inputs, 'binary_arm_counts')
                second = extraction._cached_family_extract(model, Envelope, prompt, inputs, 'binary_arm_counts')
                self.assertEqual(first.model_dump(), second.model_dump())
                self.assertEqual(1, chain.calls)
                replaced = extraction._cached_family_extract(
                    model, Envelope, prompt, {**inputs, 'sources': '[]'}, 'binary_arm_counts')
                self.assertEqual([], replaced.rows)
                self.assertEqual(2, chain.calls)
                self.assertEqual(1, len(list((Path(temp) / 'superseded').glob('binary_arm_counts.*.json'))))
            finally:
                extraction._FAMILY_CHECKPOINT_DIR.reset(token)

    def test_valid_empty_batch_does_not_force_more_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            batch = {'batch_id': 'batch_001', 'source_ids': ['s1'], 'source_chars': 25, 'prompt_chars': 200}
            state = {'batches': [{**batch, 'attempts': []}], 'semantic_fingerprint': 'fixed'}
            with patch.object(extraction, '_call_extractor', return_value=Envelope(rows=[], unresolved_items=['No counts reported'])) as call:
                result = extraction._extract_batch(task(), Tools(), Envelope, sources(), batch, state,
                    work/'state.json', work, 'article', {'options': {}}, True, 'test')
            self.assertEqual([], result.rows)
            self.assertEqual(1, call.call_count)
            self.assertEqual('completed', read(work/'batches/batch_001.json')['status'])

    def test_schema_is_in_request_budget_and_reported_separately(self):
        selected = [{'source_id': f's{i}', 'source_type': 'table', 'content': 'x'*1000} for i in range(2)]
        with patch.object(extraction, '_compact_prompt_chars', side_effect=lambda *args: 2000+sum(len(x['content']) for x in args[2])), patch.object(extraction, '_request_schema_chars', return_value=1200):
            plan = extraction._batch_plan(selected, 4700, task=task(), tools=Tools(), is_typed=True)
        self.assertEqual([['s0'], ['s1']], [b['source_ids'] for b in plan])
        self.assertTrue(all(b['request_chars'] == b['prompt_chars']+1200 for b in plan))

    def test_bounded_context_preserves_primary_coverage_and_source_ids(self):
        selected = [
            {'source_id': 'methods', 'source_type': 'text', 'content': 'Follow-up denominator was 19 and 37 participants.'},
            {'source_id': 'table1', 'source_type': 'table', 'content': 'a'*3900},
            {'source_id': 'table2', 'source_type': 'table', 'content': 'b'*3900}]
        plan = extraction._batch_plan(selected, 4500)
        self.assertEqual([s['source_id'] for s in selected], [sid for b in plan for sid in b['source_ids']])
        self.assertIn('methods', plan[-1]['context_source_ids'])
        self.assertLessEqual(plan[-1]['context_chars'], 4000)
        self.assertLessEqual(plan[-1]['request_chars'], 4500)

    def test_batch_packing_reserves_design_context_for_later_numeric_table(self):
        selected = [
            {'source_id': 'allocation', 'source_type': 'text',
             'content': 'Seventy specimens were randomized 1:1; each surgeon performed 7 per group.'},
            {'source_id': 'filler', 'source_type': 'text', 'content': 'x' * 900},
            {'source_id': 'results', 'source_type': 'table', 'content': 'y' * 900},
        ]
        with patch.object(extraction, '_compact_prompt_chars',
                          side_effect=lambda *args: 1000 + sum(len(x['content']) for x in args[2])), \
             patch.object(extraction, '_request_schema_chars', return_value=200):
            plan = extraction._batch_plan(
                selected, 3000, task=task(), tools=Tools(), is_typed=True)
        result_batch = next(batch for batch in plan if 'results' in batch['source_ids'])
        self.assertIn('allocation', result_batch['context_source_ids'])
        self.assertLessEqual(result_batch['request_chars'], 3000)

    def test_design_context_matches_randomized_and_allocation_language(self):
        selected = [
            {'source_id': 'methods', 'source_type': 'text',
             'content': 'Specimens were randomized and allocated in a 1:1 ratio.'},
            {'source_id': 'results', 'source_type': 'table', 'content': 'Mean 5, SD 4.'},
        ]
        contexts = extraction._design_context_sources(selected, {'results'})
        self.assertEqual(['methods'], [row['source_id'] for row in contexts])

    def test_semantic_constraint_changes_invalidate_checkpoint(self):
        current = task()
        controller = SimpleNamespace(inspect_document=lambda document: {})
        tools = Tools()
        tools.static_clean_document = lambda document, **kwargs: (sources(), {})
        with tempfile.TemporaryDirectory() as temp:
            args = (current, controller, tools, 'article', Path(temp), True)
            extraction._load_or_prepare_state(*args)
            current.raw['extraction']['targets'][0]['source_constraints'].append('Exclude pre-existing exposure.')
            with self.assertRaisesRegex(ContractError, 'extraction task/schema changed'):
                extraction._load_or_prepare_state(*args)

    def test_typed_cleanup_does_not_truncate_after_reference_heading(self):
        from unittest.mock import Mock
        tools = SimpleNamespace(static_clean_document=Mock(return_value=([], {})))
        extraction._call_static_clean(tools, task(), object(), True)
        self.assertTrue(tools.static_clean_document.call_args.kwargs['remove_reference_items_only'])

    def test_reviewer_receives_primary_and_interpretation_source_blocks(self):
        catalog = [*sources(), {'source_id': 'methods', 'source_type': 'text', 'content': 'Outcome-specific denominators: 19, 37.'}]
        predictions = [{'source_location': {'source_id': 's1', 'source_quote': 'Deaths'}}]
        with tempfile.TemporaryDirectory() as temp:
            extraction._attach_prediction_sources(predictions, catalog,
                [{'source_ids': ['s1'], 'context_source_ids': ['methods']}], Path(temp))
            evidence = predictions[0]['source_location']['source_evidence']
            self.assertEqual(['s1', 'methods'], [row['source_id'] for row in evidence])
            self.assertEqual(catalog[0]['content'], evidence[0]['text'])
            self.assertTrue(evidence[1]['context_only'])
            self.assertTrue(Path(predictions[0]['source_location']['source_evidence_file']).is_file())


if __name__ == '__main__':
    unittest.main()
