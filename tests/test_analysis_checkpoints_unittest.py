import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from extensions.analysis_checkpoints import (
    CheckpointedRuntimeFacetAnnotator,
    CheckpointedTopicReviewer,
)


class Item(BaseModel):
    evidence_id: str


class Group(BaseModel):
    records: list[Item]


class Candidates(BaseModel):
    groups: list[Group]


class Analysis(BaseModel):
    analysis_id: str


class Plan(BaseModel):
    name: str


class RuntimeDelegate:
    calls_made = 0
    artifact_path = None

    def annotate(self, plan, records):
        from analysis_engine.runtime_facets import RuntimeFacetResult
        self.calls_made += 1
        return RuntimeFacetResult(mode="test", status="completed")


class ReviewerDelegate:
    calls_made = 0

    def review(self, analysis, candidates):
        from analysis_engine.topic_reviewer_graph import ExpandableTopicReview
        self.calls_made += 1
        evidence_id = candidates.groups[0].records[0].evidence_id
        return ExpandableTopicReview(
            analysis_id=analysis.analysis_id,
            assignments=[],
            selected_evidence_ids=[evidence_id],
            summary="fixed",
            reviewer_mode="llm",
            request_expansion=False,
        )


class AnalysisCheckpointTests(unittest.TestCase):
    def test_cached_selection_reuses_valid_analysis_candidate_id(self):
        from extensions.analysis_checkpoints import SCHEMA, _fingerprint, _write_atomic
        from analysis_core.models import EvidenceRecord, CandidateGroup, CandidateSet
        from analysis_engine.topic_reviewer_graph import ExpandableTopicReview
        record = EvidenceRecord(evidence_id='disputed', article_id='a', study_id='s', target_id='mortality',
            raw_data={'events': 11, 'total': 39},
            source_location={'source_id': 'figure', 'source_quote': 'Survival percentage only.',
                             'review_context': {'evidence_role': 'source_only'}})
        candidates = CandidateSet(target_id='mortality', groups=[CandidateGroup(
            group_id='g', study_id='s', article_ids=['a'], evidence_ids=['disputed'], records=[record])])
        analysis = Analysis(analysis_id='mortality')
        class Delegate:
            calls_made = 0
            def review(self, analysis, candidates):
                self.calls_made += 1
                return ExpandableTopicReview(analysis_id=analysis.analysis_id,
                    assignments=[], selected_evidence_ids=[], reviewer_mode='llm', summary='Insufficient source support')
        with tempfile.TemporaryDirectory() as directory:
            delegate = Delegate()
            wrapper = CheckpointedTopicReviewer(delegate, Path(directory))
            signature = _fingerprint(analysis, candidates)
            stale = ExpandableTopicReview(analysis_id='mortality', assignments=[],
                selected_evidence_ids=['disputed'], reviewer_mode='llm', summary='Unchecked cached selection')
            _write_atomic(wrapper._path('mortality', signature), {
                'schema_version': SCHEMA, 'kind': 'topic_review', 'status': 'completed',
                'input_fingerprint': signature, 'result': stale.model_dump(mode='json')})
            result = wrapper.review(analysis, candidates)
            self.assertEqual(result.selected_evidence_ids, ['disputed'])
            self.assertEqual(delegate.calls_made, 0)

    def test_runtime_facets_are_reused_only_for_identical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            first = RuntimeDelegate()
            CheckpointedRuntimeFacetAnnotator(first, path).annotate(Plan(name="p"), [Item(evidence_id="e1")])
            self.assertEqual(first.calls_made, 1)

            second = RuntimeDelegate()
            CheckpointedRuntimeFacetAnnotator(second, path).annotate(Plan(name="p"), [Item(evidence_id="e1")])
            self.assertEqual(second.calls_made, 0)

            CheckpointedRuntimeFacetAnnotator(second, path).annotate(Plan(name="changed"), [Item(evidence_id="e1")])
            self.assertEqual(second.calls_made, 1)

    def test_each_topic_review_round_is_reused_by_candidate_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis = Analysis(analysis_id="primary/topic")
            candidates = Candidates(groups=[Group(records=[Item(evidence_id="e1")])])
            first = ReviewerDelegate()
            result = CheckpointedTopicReviewer(first, Path(directory)).review(analysis, candidates)
            self.assertEqual(result.selected_evidence_ids, ["e1"])
            self.assertEqual(first.calls_made, 1)

            second = ReviewerDelegate()
            reused = CheckpointedTopicReviewer(second, Path(directory)).review(analysis, candidates)
            self.assertEqual(reused.selected_evidence_ids, ["e1"])
            self.assertEqual(second.calls_made, 0)

            changed = Candidates(groups=[Group(records=[Item(evidence_id="e2")])])
            CheckpointedTopicReviewer(second, Path(directory)).review(analysis, changed)
            self.assertEqual(second.calls_made, 1)


if __name__ == "__main__":
    unittest.main()
