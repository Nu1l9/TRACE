"""Offline integration of real sparse retrieval with the existing input contract."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extensions import common
from literature_retrieval import pipeline as p
from literature_retrieval.retrieval import HybridRetriever, QueryVariant

ROOT = Path(__file__).resolve().parents[1]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.corpus = self.root / 'corpus.json'
        self.records = [
            {'ID': 1, 'title':'Intervention comparator eligible population',
             'abstract':'intervention comparator prespecified outcome', 'year':2020,
             'study_id':'shared_study', 'sections':{'Results':'Five of ten compared with ten of ten.'},
             'expected_label':'Include'},
            {'ID': 2, 'title':'Intervention eligible population', 'abstract':'prespecified outcome',
             'year':2021, 'study_id':'shared_study', 'sections':{'Results':'Follow-up report.'}},
            {'ID': 3, 'title':'Other geography topic', 'abstract':'mountains glaciers', 'year':2020}]
        self.write_corpus()
        self.store = self.root / 'store.sqlite'; self.index = self.root / 'bm25'
        p.prepare_store(self.corpus, self.store)
        p.build_index('bm25', self.store, self.index, min_df=1)
        raw = json.loads((ROOT/'vendor/integration/examples/request.json').read_text())
        raw.pop('documents'); raw['options']['extraction_runtime']='typed'
        self.request = self.root / 'task.json'; p.atomic(self.request, raw)
        self.args = argparse.Namespace(command='review', request=self.request, output=self.root/'run',
            store=self.store,bm25_index=self.index,dense_index=self.root/'dense',mode='bm25',
            per_query_k=10,fusion_k=10,final_k=10,max_year=None,reranker_model=None,
            document_map=None,prepare_only=False,resume=False,env_file=None,extensions=None,
            workers=2,retry_deferred=False)

    def write_corpus(self):
        self.corpus.write_text('[\n'+',\n'.join(json.dumps(r) for r in self.records)+'\n]')

    def test_real_retrieval_handoff_preserves_contract_and_study_grouping(self):
        path=p.prepare_review(self.args); result=p.read(path)
        self.assertEqual([d['article_id'] for d in result['documents']],['corpus_1','corpus_2'])
        self.assertEqual({d['study_id'] for d in result['documents']},{'shared_study'})
        self.assertNotIn('expected_label', json.dumps(result))
        self.assertIn('Five of ten',Path(result['documents'][0]['paths']['markdown']).read_text())
        p.validate_seed(self.args.output, result)
        changed=deepcopy(result); changed['documents'][0]['paths']['markdown']='another.md'
        with self.assertRaisesRegex(ValueError,'source path changed'):
            p.validate_seed(self.args.output, changed)
        from trace_runtime.pipeline import Runner
        runner=Runner(result,self.args.output,{},initial_artifacts=('00_retrieval',))
        self.assertEqual(runner.state['status'],'running')

    def test_prepared_text_can_drop_archival_pdf_but_not_change_it(self):
        pdf=self.root/'source.pdf';pdf.write_bytes(b'%PDF- fixture')
        self.records[0]['paths']={'pdf':str(pdf),'markdown':str(ROOT/'vendor/integration/examples/article_a.md')}
        self.write_corpus();p.prepare_store(self.corpus,self.store,True)
        p.build_index('bm25',self.store,self.index,rebuild=True,min_df=1)
        result=p.read(p.prepare_review(self.args));result['documents'][0]['paths'].pop('pdf')
        p.validate_seed(self.args.output,result)
        pdf.write_bytes(b'%PDF- changed')
        with self.assertRaisesRegex(ValueError,'Retrieval source changed'):p.validate_seed(self.args.output,result)

    def test_resume_does_not_retrieve_or_require_live_index(self):
        path=p.prepare_review(self.args); self.args.resume=True
        with patch.object(HybridRetriever,'retrieve',side_effect=AssertionError('must not retrieve')):
            self.assertEqual(p.prepare_review(self.args),path)
        data=p.read(path); data['documents'][0]['study_id']='changed';p.atomic(path,data)
        with self.assertRaisesRegex(ValueError,'Frozen retrieval request changed'):p.prepare_review(self.args)

    def test_missing_source_is_recorded_not_silently_excluded(self):
        self.records[0]['sections']={};self.write_corpus()
        p.prepare_store(self.corpus,self.store,True);p.build_index('bm25',self.store,self.index,rebuild=True,min_df=1)
        with self.assertRaisesRegex(ValueError,'downstream not started'):p.prepare_review(self.args)
        report=p.read(self.args.output/'00_retrieval/retrieval.json')
        self.assertEqual(report['candidate_count'],2);self.assertFalse(report['ready'])
        self.assertEqual(report['missing_inputs'][0]['corpus_id'],1)
        self.assertFalse((self.args.output/'00_retrieval/request.json').exists())

    def test_missing_study_id_needs_explicit_mapping(self):
        record=deepcopy(self.records[0]);record.pop('study_id')
        with self.assertRaisesRegex(ValueError,'study_id required'):
            p.source_document(record,self.root,self.root/'sources')
        result=p.source_document(record,self.root,self.root/'sources',{'study_id':'trial_A'})
        self.assertEqual(result['study_id'],'trial_A')

    def test_changed_corpus_and_index_settings_rejected(self):
        with self.assertRaisesRegex(ValueError,'settings/model changed'):
            p.build_index('bm25',self.store,self.index,min_df=2)
        self.corpus.write_text(self.corpus.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Corpus moved or changed'):p.prepare_review(self.args)

    def test_existing_candidates_are_not_silently_replaced(self):
        raw=p.read(self.request);raw['documents']=[{'article_id':'existing'}];p.atomic(self.request,raw)
        with self.assertRaisesRegex(ValueError,'documents omitted or empty'):p.prepare_review(self.args)

    def test_execute_delegates_once_to_existing_run_command(self):
        with patch.object(p,'prepare_review',return_value=self.request):
            calls=[]
            result=p.execute(self.args,lambda argv: calls.append(argv) or 7)
        self.assertEqual(result,7);self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],'run');self.assertNotIn('--screen',calls[0])

    def test_date_filter_and_empty_candidates(self):
        self.args.max_year=2019;self.args.prepare_only=True
        self.assertIsNone(p.prepare_review(self.args))
        self.assertEqual(p.read(self.args.output/'00_retrieval/retrieval.json')['status'],'no_candidates')

    def test_source_mutation_invalidates_resume(self):
        request=p.read(p.prepare_review(self.args));self.args.resume=True
        Path(request['documents'][0]['paths']['markdown']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'source changed'):p.prepare_review(self.args)

    def test_two_retrievers_fuse_same_candidate_once(self):
        class Index:
            def __init__(self, ranking):self.ranking=ranking
            def search(self,*args,**kwargs):return self.ranking
        store=p.DocumentStore(self.store)
        r=HybridRetriever(store,bm25=Index([(1,5),(2,4)]),dense=Index([(2,.8),(1,.7)]))
        rows=r.retrieve({},variants=[QueryVariant('topic','intervention')],final_k=10)
        self.assertEqual({c.corpus_id for c in rows},{1,2})
        self.assertTrue(all(len(c.trace)==2 for c in rows))
        self.assertAlmostEqual(rows[0].score,1/61+1/62)


if __name__ == '__main__':unittest.main()
