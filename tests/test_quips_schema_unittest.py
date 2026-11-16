import unittest

from pydantic import ValidationError

from extensions.rob_schema import response_model


class QuipsSchemaTests(unittest.TestCase):
    def test_provider_schema_enforces_domain_specific_item_counts(self):
        schema = response_model('QUIPS', '2013').model_json_schema()
        domain_set = schema['$defs']['RoBDomainSet']['properties']
        for domain_id, minimum, maximum in (
            ('Q1', 7, 8), ('Q2', 5, 6), ('Q3', 6, 7),
            ('Q4', 3, 4), ('Q5', 7, 8), ('Q6', 4, 5),
        ):
            ref = domain_set[domain_id]['$ref'].split('/')[-1]
            items = schema['$defs'][ref]['properties']['assessment_items']
            self.assertEqual(minimum, items['minItems'])
            self.assertEqual(maximum, items['maxItems'])

    def test_three_item_domain_is_rejected_by_pydantic(self):
        item = {'item': 'one', 'response': 'NI', 'reasoning': 'Not reported.'}
        domain = {
            'judgment': 'unclear', 'rationale': 'Insufficient information.',
            'evidence': [], 'assessment_items': [item, {**item, 'item': 'two'}, {**item, 'item': 'three'}],
            'signalling_questions': [],
        }
        payload = {
            'domains': {domain_id: domain for domain_id in ('Q1','Q2','Q3','Q4','Q5','Q6')},
            'overall_judgment': 'unclear', 'overall_rationale': 'Insufficient information.',
        }
        with self.assertRaises(ValidationError):
            response_model('QUIPS', '2013').model_validate(payload)


if __name__ == '__main__':
    unittest.main()
