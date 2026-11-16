"""Mandatory Graph sidecar policy; frozen component requests are never rewritten."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from .common import read, require, digest

DEFAULT = {
    'schema_version': 'meta_extensions/2.0',
    # Graph retrieval remains part of analysis, but an optional LLM alias loop
    # must not turn a completed evidence handoff into a new provider-dependent
    # stage.  It can be enabled explicitly for an exploratory run.
    'graph': {'brain_loop': False, 'max_rounds': 2},
    # Every Graph run routes RoB during screening, drafts the routed assessment
    # before extraction, and creates tool-specific high-risk sensitivities.
    'bias': {'mode': 'draft', 'sensitivity_policies': [], 'auto_sensitivity': True},
}

def load_settings(path=None):
    if path is None:
        return deepcopy(DEFAULT)
    path=Path(path).resolve(); value=read(path)
    require(isinstance(value,dict) and value.get('schema_version')=='meta_extensions/2.0','extensions require meta_extensions/2.0')
    require(set(value)<= {'schema_version','graph','bias'},'unknown extension setting')
    result=deepcopy(DEFAULT)
    for key in ('graph','bias'):
        require(isinstance(value.get(key,{}),dict),key+' must be an object')
        result[key].update(value.get(key,{}))
    require(set(result['graph'])<= {'patch_file','brain_loop','max_rounds'},'unknown graph setting')
    require(type(result['graph'].get('brain_loop',False)) is bool,'graph.brain_loop must be a boolean')
    rounds=result['graph'].get('max_rounds',2)
    require(type(rounds) is int and 0<=rounds<=2,'graph.max_rounds must be an integer from 0 to 2')
    require(set(result['bias'])<= {'mode','assessments_file','sensitivity_policies','auto_sensitivity'},'unknown bias setting')
    require(result['bias']['mode'] in ('off','import','draft'),'bias.mode must be off/import/draft')
    require(isinstance(result['bias']['sensitivity_policies'],list),'sensitivity_policies must be a list')
    require(type(result['bias'].get('auto_sensitivity',True)) is bool,'bias.auto_sensitivity must be boolean')
    if result['bias']['mode']=='off':
        require(not result['bias'].get('assessments_file') and not result['bias']['sensitivity_policies'],'bias off cannot have assessments or sensitivity policies')
    if result['bias']['mode']=='import':
        require(bool(result['bias'].get('assessments_file')),'import requires assessments_file')
    if result['bias']['mode']=='draft':
        require(not result['bias'].get('assessments_file'),'draft cannot also import assessments')
    for key,field in [('graph','patch_file'),('bias','assessments_file')]:
        if result[key].get(field):
            source=Path(result[key][field]).expanduser()
            if not source.is_absolute():source=path.parent/source
            source=source.resolve();require(source.is_file(),f'missing extension input: {source}')
            result[key][field]=str(source)
    return result

def settings_fingerprint(settings):
    external={}
    for key,field in [('graph','patch_file'),('bias','assessments_file')]:
        if settings[key].get(field):external[field]=read(settings[key][field])
    return digest({'settings':settings,'external_inputs':external})
