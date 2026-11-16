"""Strict provider response models; domain identity is fixed by the routed tool."""
from functools import lru_cache
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, create_model

class StrictModel(BaseModel):
    model_config=ConfigDict(strict=True,extra='forbid')

class Locator(StrictModel):
    line_start: int | None
    line_end: int | None
    page: int | None
    content_index: int | None

class Evidence(StrictModel):
    article_id: str = Field(min_length=1)
    source_type: Literal['markdown','content_list']
    locator: Locator | None
    quote: str | None

class Item(StrictModel):
    item: str = Field(min_length=1)
    response: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)

class Signal(StrictModel):
    question_id: str = Field(min_length=1)
    answer: Literal['Y','PY','PN','N','NI','NA']
    rationale: str = Field(min_length=1)

@lru_cache(maxsize=32)
def response_model(name,version):
    from .bias import ROB2_DOMAINS, QUIPS_DOMAINS, QUIPS_ITEM_COUNTS, ROUTED_TOOL_SPECS, QUIPS_JUDGMENTS, JUDGMENTS
    if name=='RoB 2': domains=ROB2_DOMAINS; judgments=JUDGMENTS|{'unknown'}
    elif name=='QUIPS': domains=QUIPS_DOMAINS; judgments=QUIPS_JUDGMENTS|{'unknown'}
    else:
        spec=ROUTED_TOOL_SPECS[(name,version)];domains=spec['domains'];judgments=spec['judgments']|{'unknown'}
    label=Literal[tuple(sorted(judgments))]
    def domain_model(domain_id):
        item_field = (Field(min_length=QUIPS_ITEM_COUNTS[domain_id]-1,
                            max_length=QUIPS_ITEM_COUNTS[domain_id])
                      if name == 'QUIPS' else Field())
        return create_model('RoBDomain_'+domain_id,__base__=StrictModel,
            judgment=(label,...),rationale=(str,Field(min_length=1)),
            evidence=(list[Evidence],...),assessment_items=(list[Item],item_field),
            signalling_questions=(list[Signal],...))
    domain_set=create_model('RoBDomainSet',__base__=StrictModel,
                            **{did:(domain_model(did),...) for did in domains})
    return create_model('RoBAssessment',__base__=StrictModel,domains=(domain_set,...),
                        overall_judgment=(label,...),overall_rationale=(str,Field(min_length=1)))

def provider_format(payload):
    tool=payload['assessment_template']['tool']
    model=response_model(tool['name'],tool['version'])
    return {'type':'json_schema','json_schema':{'name':'risk_of_bias_assessment','strict':True,'schema':model.model_json_schema()}}

def parse_response(content,payload):
    import json
    def unique_pairs(pairs):
        obj={}
        for key,value in pairs:
            if key in obj: raise ValueError('Duplicate JSON field: '+key)
            obj[key]=value
        return obj
    json.loads(content,object_pairs_hook=unique_pairs)
    tool=payload['assessment_template']['tool']
    result=response_model(tool['name'],tool['version']).model_validate_json(content,strict=True).model_dump()
    from .bias import ROB2_DOMAINS
    for did,domain in result['domains'].items():
        for evidence in domain['evidence']:
            if evidence['article_id']!=payload['candidate']['article_id']:
                raise ValueError('RoB evidence article differs from requested article')
        signals=domain['signalling_questions']
        ids=[s['question_id'] for s in signals]
        if tool['name']=='RoB 2':
            if len(ids)!=len(set(ids)) or set(ids)!=set(ROB2_DOMAINS[did][1]):
                raise ValueError('RoB 2 requires exactly the declared signalling questions for '+did)
        elif signals: raise ValueError('Signalling questions must be empty for this tool')
    result['domains']=[{'domain_id':did,**value} for did,value in result['domains'].items()]
    result['response_contract']='pydantic_strict_rob/1.0'
    return result
