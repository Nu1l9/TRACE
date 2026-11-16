import sys
import types
import unittest
from unittest.mock import patch
from metapipeline.brain import draft,propose
from metapipeline.contracts import ContractError
class BrainTests(unittest.TestCase):
    def test_port_calls_plugin_and_preserves_input(self):
        request=draft('topic');mod=types.ModuleType('fixture_brain')
        def fill(r):
            r['topic']='edited'
            return {'schema_version':'brain_proposal/1.0','pipeline_request':{'schema_version':'meta_pipeline/1.0'},'missing_fields':['protocol'],'uncertainties':[]}
        mod.fill=fill
        with patch.dict(sys.modules,{'fixture_brain':mod}):result=propose(request,'fixture_brain:fill')
        self.assertEqual(request['topic'],'topic');self.assertEqual(result['missing_fields'],['protocol'])
    def test_plugin_requires_explicit_missing_fields(self):
        mod=types.ModuleType('fixture_brain');mod.fill=lambda _:{}
        with patch.dict(sys.modules,{'fixture_brain':mod}),self.assertRaises(ContractError):propose(draft('topic'),'fixture_brain:fill')
