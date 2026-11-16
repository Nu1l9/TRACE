"""Reserved Brain port. Implement fill(request) -> proposal; no model is bundled."""
from __future__ import annotations
from copy import deepcopy
from importlib import import_module
from typing import Protocol, Any
from .contracts import VERSION, normalize_request, require, digest

class Brain(Protocol):
    def fill(self, request: dict[str, Any]) -> dict[str, Any]: ...

def draft(topic):
    return {'schema_version':'brain_request/1.0','topic':topic,'known_fields':{},
        'required_fields':['case_id','protocol.protocol_id','protocol.pico.population','protocol.pico.intervention_or_exposure','protocol.pico.comparison','protocol.pico.outcome','protocol.targets','analysis_profile.primary_analysis_spec','documents'],
        'constraints':['Use protocol fields only; do not invent source evidence, source data, study identities or analysis results.','Report uncertain or missing fields explicitly.']}

def propose(request,plugin):
    module,sep,name=plugin.partition(':')
    require(bool(sep and module and name),'Brain plugin must be module:function')
    result=getattr(import_module(module),name)(deepcopy(request))
    require(isinstance(result,dict) and result.get('schema_version')=='brain_proposal/1.0','Brain must return brain_proposal/1.0')
    require(isinstance(result.get('pipeline_request'),dict),'Brain must return pipeline_request')
    require(isinstance(result.get('missing_fields'),list) and isinstance(result.get('uncertainties'),list),'Brain must expose missing_fields and uncertainties')
    # Proposal is saved for editing, never auto-executed by this port.
    return result
