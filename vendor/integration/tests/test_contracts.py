from pathlib import Path
import tempfile
import unittest
import json
from metapipeline.contracts import *
ROOT=Path(__file__).resolve().parents[1];EX=ROOT/'examples'

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.r=normalize_request(read(EX/'request.json'),EX)
        self.s=read(EX/'screening.json');self.e=read(EX/'submission.json');self.v=read(EX/'review.json')
    def handoff(self):return evidence_handoff(self.r,self.s,self.e,self.v)
    def test_fields_and_numeric_values_preserved(self):
        c,a=self.handoff()
        self.assertEqual(set(c['candidates'][0]),set(ENVELOPE))
        self.assertEqual(c['candidates'][0]['raw_data'],self.e['predictions'][0]['raw_data'])
        self.assertEqual(c['candidates'][0]['comparison'],'comparator')
        self.assertEqual(a['candidate_count'],2)
    def test_native_miners_output(self):
        self.s={'protocol':{'case_id':'demo_case'},'screening':self.s['decisions']}
        self.assertEqual(len(self.handoff()[0]['candidates']),2)
    def test_excluded_article_cannot_enter_analysis(self):
        self.e['predictions'][0]['article_id']='article_c'
        with self.assertRaisesRegex(ContractError,'non-included'):self.handoff()
    def test_missing_screening_decision(self):
        self.s['decisions'].pop()
        with self.assertRaises(ContractError):self.handoff()
    def test_duplicate_prediction(self):
        self.e['predictions'][1]['prediction_id']=self.e['predictions'][0]['prediction_id']
        with self.assertRaisesRegex(ContractError,'unique'):self.handoff()
    def test_incomplete_review(self):
        self.v['candidate_reviews'].pop()
        with self.assertRaisesRegex(ContractError,'cover'):self.handoff()
    def test_failed_review(self):
        self.v['failed_review_count']=1
        with self.assertRaisesRegex(ContractError,'review failed'):self.handoff()
    def test_only_clear_mismatch_discarded(self):
        self.v['candidate_reviews'][0]['review']['disposition']='discard_clear_mismatch'
        self.v['candidate_reviews'][1]['review']['disposition']='needs_further_review'
        c,a=self.handoff()
        self.assertEqual([x['candidate_id'] for x in c['candidates']],['ev_2'])
        self.assertEqual(a['discarded_count'],1)
    def test_same_study_multiple_reports(self):
        self.r['documents'][1]['study_id']='study_a';self.e['predictions'][1]['study_id']=None
        c,_=self.handoff();self.assertEqual({x['study_id'] for x in c['candidates']},{'study_a'})
    def test_conflicting_study(self):
        self.e['predictions'][0]['study_id']='another_study'
        with self.assertRaisesRegex(ContractError,'study_id conflicts'):self.handoff()
    def test_missing_target_not_guessed(self):
        self.e['predictions'][0]['target_id']=None
        with self.assertRaisesRegex(ContractError,'target_id'):self.handoff()
    def test_baseline_derived_effect_omitted(self):
        self.e['predictions'][0]['used_effect']={'effect_type':'RR','value':0.5,'ci_lower':0.1,'ci_upper':0.9}
        c,a=self.handoff();self.assertEqual(c['candidates'][0]['effect'],{});self.assertTrue(a['records'][0]['warnings'])
    def test_reported_effect_preserved(self):
        self.e['predictions'][0]['data_family']='adjusted_time_to_event_effect'
        effect={'effect_type':'HR','value':0.8,'ci_lower':0.7,'ci_upper':0.9};self.e['predictions'][0]['used_effect']=effect
        c,_=self.handoff();self.assertEqual(c['candidates'][0]['effect'],effect);self.assertEqual(c['candidates'][0]['data_family'],'adjusted_ratio_effect')
    def test_upstream_labels_not_passed(self):
        self.e['predictions'][0].update(quantitative_usability='poolable',expected_label='Gold',selected=True)
        c,_=self.handoff();self.assertNotIn('quantitative_usability',json.dumps(c));self.assertNotIn('expected_label',json.dumps(c))
    def test_nested_selection_label_rejected(self):
        self.e['predictions'][0]['raw_data']['selected_candidate_ids']=['x']
        with self.assertRaises(ContractError):self.handoff()
    def test_conflicting_pico(self):
        self.r['protocol']['pico']['comparator']='different'
        with self.assertRaisesRegex(ContractError,'Conflicting'):normalize_request(self.r)
    def test_invalid_profile_target(self):
        self.r['analysis_profile']['primary_analysis_spec']['target_id']='other'
        with self.assertRaisesRegex(ContractError,'target_id'):normalize_request(self.r)
    def test_missing_study_rejected(self):
        self.r['documents'][0].pop('study_id')
        with self.assertRaises(ContractError):normalize_request(self.r)
    def test_multiple_families_rejected(self):
        self.r['analysis_profile']['primary_analysis_spec']['eligible_data_families']=['binary_arm_counts','reported_ratio_effect']
        with self.assertRaisesRegex(ContractError,'exactly one'):normalize_request(self.r)
    def test_credentials_rejected(self):
        self.r['options']['api_key']='fixture'
        with self.assertRaisesRegex(ContractError,'Credentials'):normalize_request(self.r)
    def test_path_identifier_rejected(self):
        self.r['documents'][0]['article_id']='../../escape'
        with self.assertRaises(ContractError):normalize_request(self.r)
    def test_typed_runtime_is_not_restricted_by_case_id(self):
        self.r['case_id']=self.r['protocol']['case_id']='ma_0001'
        self.r['options']['extraction_runtime']='typed'
        self.assertEqual(normalize_request(self.r)['options']['extraction_runtime'], 'typed')
        self.r['options']['extraction_runtime']='nonexistent'
        with self.assertRaises(ContractError):normalize_request(self.r)
    def test_extraction_documents_filtered(self):
        with tempfile.TemporaryDirectory() as d:
            p=prepare_extraction(self.r,['article_a'],d)
            self.assertEqual(read(p)['inputs']['included_article_ids'],['article_a'])
            self.assertEqual(set(read(Path(d)/'inputs/paths.json')['documents']),{'article_a'})
    def test_source_document_conflict(self):
        self.e['predictions'][0]['source_location']['document_id']='article_c'
        with self.assertRaisesRegex(ContractError,'source document mismatch'):self.handoff()
    def test_genotype_counts_need_explicit_roles(self):
        row=self.e['predictions'][0];row['data_family']='complete_genotype_case_control_counts'
        row['raw_data']={'rsid':'rs_demo','genotype_labels':['AA','AG'],'case_counts':{'AA':30,'AG':10},'control_counts':{'AA':40,'AG':5}}
        with self.assertRaisesRegex(ContractError,'genotype_contrast'):self.handoff()
        self.r['documents'][0]['genotype_contrast']={'rsid':'rs_demo','heterozygote_label':'AG','common_homozygote_label':'AA','source_note':'Synthetic source explicitly defines AA as reference'}
        c,a=self.handoff();raw=c['candidates'][0]['raw_data']
        self.assertEqual(raw['case_heterozygote'],10);self.assertEqual(raw['control_common_homozygote'],40)
        self.assertEqual(raw['case_counts'],{'AA':30,'AG':10});self.assertIn('raw_field_mapping',a['records'][0])
    def test_continuous_semantics_preserved(self):
        self.e['predictions'][0]['raw_data'].update(outcome_axis='version',unit='degrees',measurement_semantics='absolute_deviation')
        c,_=self.handoff();self.assertEqual(c['candidates'][0]['outcome']['analysis_semantics']['outcome_axis'],'version')
