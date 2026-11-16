import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor/integration'),
               str(ROOT / 'vendor/integration/vendor/evidence_analysis')]
from analysis_core.models import AnalysisDefinition, EvidenceRecord, ResolvedAnalysisPlan
from analysis_engine.planner import DeterministicAnalysisPlanner
from analysis_engine.retrieval import HashEmbeddingBackend
from analysis_engine.semantic_graph import (
    SemanticEvidenceGraph, SemanticRetriever, semantic_dimensions,
)


class SemanticRecallTests(unittest.TestCase):
    def test_two_remaining_studies_do_not_hide_a_third_semantic_misclassification(self):
        analysis = AnalysisDefinition(analysis_id='mortality', pooled_result_id='pooled', title='Mortality',
            kind='primary', target_id='death', outcome_group='All-cause mortality', outcome_terms=['All-cause mortality'],
            eligible_data_families=['binary_arm_counts'], accepted_effect_types=['RR'], population_policy='any')
        plan = ResolvedAnalysisPlan(plan_id='plan', profile_version='2.0', task_id='test', protocol_id='test',
            primary_analysis_id=analysis.analysis_id, analyses=[analysis])
        records = [EvidenceRecord(evidence_id=f'e{i}', article_id=f'a{i}', study_id=f's{i}', target_id='death',
            outcome='All-cause mortality', data_family='binary_arm_counts',
            raw_data={'intervention_events': 5, 'intervention_total': 19, 'control_events': 9, 'control_total': 37})
            for i in range(3)]
        predicate = semantic_dimensions(plan)[1][analysis.analysis_id][0]
        # A valid source pointer does not guarantee the model's interpretation is correct.
        facets = {'e2': {'dimensions': [{'field': predicate, 'status': 'explicit_other',
            'level_ids': [], 'contradicted_level_ids': ['scope'], 'source_path': ['outcome'],
            'source_quote': 'All-cause mortality', 'source_basis': 'Simulated semantic misclassification.'}]}}
        graph = SemanticEvidenceGraph(records, plan=plan, runtime_facets=facets)
        candidates = graph.topic_candidates(analysis, plan)
        self.assertEqual({r.evidence_id for g in candidates.groups for r in g.records}, {'e0', 'e1', 'e2'})
        audit = graph.topic_audit[analysis.analysis_id]
        self.assertEqual(audit['reviewable_count'], 3)
        self.assertEqual(audit['mode'], 'graph_outcome_subgroup')
        self.assertEqual(audit['runtime_facet_calls'], 0)

    def test_outcomes_and_subgroups_are_traversed_and_unknowns_retained(self):
        primary = AnalysisDefinition(analysis_id='death', pooled_result_id='p', title='Death',
            kind='primary', target_id='death', outcome_group='Mortality', outcome_terms=['Mortality'],
            eligible_data_families=['binary_arm_counts'], accepted_effect_types=['RR'], population_policy='any')
        subgroup = primary.model_copy(update={'analysis_id': 'adults', 'kind': 'subgroup',
            'dimension_field': 'population_or_subgroup', 'dimension_value_id': 'adult', 'dimension_terms': ['adult']})
        plan = ResolvedAnalysisPlan(plan_id='p', profile_version='2.0', task_id='t', protocol_id='t',
            primary_analysis_id='death', analyses=[primary, subgroup])
        records = [EvidenceRecord(evidence_id=eid, article_id=eid, study_id=eid, target_id='unresolved',
            outcome=outcome, population_or_subgroup=population, data_family='binary_arm_counts')
            for eid, outcome, population in [('a','Mortality','adult'), ('b','Mortality','children'),
                ('c','Mortality',''), ('d','Pain','adult'), ('e','','adult')]]
        graph = SemanticEvidenceGraph(records, plan=plan)
        ids = lambda c: {r.evidence_id for g in c.groups for r in g.records}
        self.assertEqual(ids(graph.topic_candidates(primary, plan)), {'a','b','c','e'})
        self.assertEqual(ids(graph.topic_candidates(subgroup, plan)), {'a','c','e'})
        self.assertEqual(ids(graph.topic_candidates(subgroup, plan, expanded=True)), {'a','b','c','d','e'})

    def test_zero_outcome_match_recovers_only_declared_target_and_family(self):
        analysis = AnalysisDefinition(analysis_id='episode_rate', pooled_result_id='pooled',
            title='Episode rate', kind='primary', target_id='pfapa_episode_rate',
            outcome_group='PFAPA episode rate', outcome_terms=['PFAPA episode rate'],
            eligible_data_families=['event_counts_with_person_time'],
            accepted_effect_types=['IRR'], population_policy='any')
        plan = ResolvedAnalysisPlan(plan_id='p', profile_version='2.0', task_id='t',
            protocol_id='t', primary_analysis_id=analysis.analysis_id, analyses=[analysis])
        records = [EvidenceRecord(evidence_id=eid, article_id=eid, study_id=eid,
            target_id=target, outcome=outcome, data_family=family)
            for eid, target, family, outcome in [
                ('trial_a', 'pfapa_episode_rate', 'event_counts_with_person_time',
                 'PFAPA-compatible fever episodes'),
                ('trial_b', 'pfapa_episode_rate', 'event_counts_with_person_time',
                 'Total number of PFAPA episodes during randomized follow-up'),
                ('resolution', 'pfapa_symptom_resolution', 'binary_arm_counts',
                 'Disappearance of PFAPA symptoms'),
                ('wrong_family', 'pfapa_episode_rate', 'binary_arm_counts',
                 'Children with recurrent fever'),
            ]]
        graph = SemanticEvidenceGraph(records, plan=plan)
        candidates = graph.topic_candidates(analysis, plan)
        self.assertEqual({r.evidence_id for g in candidates.groups for r in g.records},
                         {'trial_a', 'trial_b'})
        audit = graph.topic_audit[analysis.analysis_id]
        self.assertEqual(audit['outcome_matched'], 0)
        self.assertEqual(audit['fallback'], 'zero_outcome_match_target_and_data_family')
        self.assertEqual(audit['reviewable_count'], 2)
        inventory = SemanticRetriever(graph, HashEmbeddingBackend()).inventory(plan)
        self.assertEqual(inventory[0].candidate_study_count, 2)
        decision = DeterministicAnalysisPlanner().plan_retrieval(plan, inventory)
        self.assertEqual(decision.actions[0].action, 'review_topic')

    def test_partial_outcome_match_recovers_0064_style_recurrence_counts(self):
        analysis = AnalysisDefinition(analysis_id='recurrence', pooled_result_id='pooled',
            title='Any recurrence', kind='primary', target_id='any_recurrence_low_volume_nodal_metastasis_vs_node_negative',
            outcome_group='Any recurrence in endometrial cancer',
            outcome_terms=['Any recurrence in endometrial cancer'],
            eligible_data_families=['binary_arm_counts'], accepted_effect_types=['RR'], population_policy='any',
            minimum_studies=2)
        plan = ResolvedAnalysisPlan(plan_id='p', profile_version='2.0', task_id='t',
            protocol_id='t', primary_analysis_id=analysis.analysis_id, analyses=[analysis])
        records = [EvidenceRecord(evidence_id=eid, article_id=eid, study_id=eid,
            target_id=target, outcome=outcome, data_family=family,
            raw_data={'intervention_events': 2, 'intervention_total': 23,
                      'control_events': 0, 'control_total': 18})
            for eid, target, family, outcome in [
                ('counts_003', analysis.target_id, 'binary_arm_counts', 'Any source-defined recurrence'),
                ('counts_010', analysis.target_id, 'binary_arm_counts', 'Any recurrence in endometrial cancer'),
                ('counts_012', analysis.target_id, 'binary_arm_counts', 'Any source-defined disease recurrence'),
                ('hr_only', 'source_reported_disease_event_hazard_low_volume_nodal_metastasis',
                 'adjusted_ratio_effect', 'Disease-free survival event'),
            ]]
        graph = SemanticEvidenceGraph(records, plan=plan)
        ids = {record.evidence_id for group in graph.topic_candidates(analysis, plan).groups
               for record in group.records}
        self.assertEqual(ids, {'counts_003', 'counts_010', 'counts_012'})
        audit = graph.topic_audit[analysis.analysis_id]
        self.assertEqual(audit['fallback'], 'partial_outcome_match_target_and_data_family')
        self.assertEqual(set(audit['typed_anchor_recovered_ids']), {'counts_003', 'counts_012'})
        inventory = SemanticRetriever(graph, HashEmbeddingBackend()).inventory(plan)
        self.assertEqual(inventory[0].candidate_study_count, 3)
        self.assertEqual(DeterministicAnalysisPlanner().plan_retrieval(plan, inventory).actions[0].action,
                         'review_topic')

    def test_shared_target_does_not_recover_a_clearly_other_outcome(self):
        base = AnalysisDefinition(analysis_id='version', pooled_result_id='v', title='Version',
            kind='primary', target_id='glenoid_summary', outcome_group='Version deviation',
            outcome_terms=['Glenoid version deviation'], eligible_data_families=['independent_continuous_arms'],
            accepted_effect_types=['MD_degrees'], population_policy='any')
        inclination = base.model_copy(update={'analysis_id': 'inclination', 'pooled_result_id': 'i',
            'outcome_group': 'Inclination deviation', 'outcome_terms': ['Glenoid inclination deviation']})
        plan = ResolvedAnalysisPlan(plan_id='p', profile_version='2.0', task_id='t',
            protocol_id='t', primary_analysis_id='version', analyses=[base, inclination])
        records = [EvidenceRecord(evidence_id=eid, article_id=eid, study_id=eid,
            target_id='glenoid_summary', outcome=outcome, data_family='independent_continuous_arms')
            for eid, outcome in [('v', 'Glenoid version deviation'),
                                 ('i', 'Glenoid inclination deviation')]]
        graph = SemanticEvidenceGraph(records, plan=plan)
        ids = {record.evidence_id for group in graph.topic_candidates(base, plan).groups
               for record in group.records}
        self.assertEqual(ids, {'v'})

    def test_zero_outcome_match_does_not_recover_unrelated_evidence(self):
        analysis = AnalysisDefinition(analysis_id='episode_rate', pooled_result_id='pooled',
            title='Episode rate', kind='primary', target_id='pfapa_episode_rate',
            outcome_group='PFAPA episode rate', outcome_terms=['PFAPA episode rate'],
            eligible_data_families=['event_counts_with_person_time'],
            accepted_effect_types=['IRR'], population_policy='any')
        plan = ResolvedAnalysisPlan(plan_id='p', profile_version='2.0', task_id='t',
            protocol_id='t', primary_analysis_id=analysis.analysis_id, analyses=[analysis])
        record = EvidenceRecord(evidence_id='resolution', article_id='a', study_id='s',
            target_id='pfapa_symptom_resolution', outcome='Disappearance of PFAPA symptoms',
            data_family='binary_arm_counts')
        graph = SemanticEvidenceGraph([record], plan=plan)
        self.assertEqual(graph.topic_candidates(analysis, plan).groups, [])
        self.assertIsNone(graph.topic_audit[analysis.analysis_id]['fallback'])


if __name__ == '__main__':
    unittest.main()
