"""Original generative baseline prompts and output validation.
No data selection or credentials are embedded in this helper.
"""
from __future__ import annotations
import json
import re
from decimal import Decimal, InvalidOperation
import hmi_config as _release
SCHEMA_VERSION = "1.2"

def llm(system, user):
    return _release.chat_text(system,user)

def schema(case:dict) -> dict:
    return {'schema_version':SCHEMA_VERSION,
            'source':{'speaker':case['speaker'], 'text':case['text'], 'domain':case['domain']},
            'shared_semantics':{'intent_id':None,'units':[],'links':[]},
            'evidence':[], 'mapping_status':'unmapped'}

def prompt_for(case:dict, method:str) -> tuple[str,str]:
    base = ('You map expressions to a shared semantic schema. JSON only, without Markdown. '
            'Both human and machine expressions use the same schema; neither speaker is the truth.\n'
            'Exact output structure: '+json.dumps(schema(case),ensure_ascii=False)+'\n'
            'Each unit contains unit_id, category, concept_id, surface, token_span, qualifiers. '
            'Links refer to unit_id, and evidence contains unit_id and text_span.\n')
    if case['dataset'] == 'dialect':
        base += ('Task: Bavarian dialect intent and slot parsing. Allowed labels: '+
                 json.dumps(case['catalog'],ensure_ascii=False)+'\n'
                 'intent_id="intent:<intent>". For each slot use category="slot", '
                 'concept_id="slot:<slot>" and token_span [start,end) in the token list. '
                 'For tokens not belonging to a slot, emit no unit. links=[].\n'
                 'Tokens: '+json.dumps(case['tokens'],ensure_ascii=False)+'\n')
    else:
        base += ('Task: map a lay medical mention to one candidate SNOMED concept. '
                 'Select concept_id exactly from the following list or return empty units. '
                 'intent_id=null, category="term", at most one unit, links=[].\n'
                 'Candidates: '+json.dumps(case['catalog'],ensure_ascii=False)+'\n')
    if method == 'ours_v1':
        base += (
            'Shared semantic mapping constraints: (1) same meaning must map to same canonical '
            'concept regardless of human/machine speaker; (2) preserve negation, time, and uncertainty '
            'in qualifiers when present; (3) no unsupported added facts; '
            ' (4) do not conflate hypernyms with exact synonyms; '
            ' (5) keep evidence text_span exactly present in input when supplied; '
            ' (6) use unmapped when not reliable.\n'
        )
    if method == 'ours':
        base += (
            'Shared-semantic fidelity constraints: treat human/machine as equally valid sources; '
            'preserve original meaning and do not hallucinate; use the given canonical IDs; '
            'preserve negation, time and uncertainty in qualifiers if present.\n'
        )
        if case['dataset'] == 'dialect':
            indexed = '\n'.join(f'[{i}] {tok}' for i, tok in enumerate(case['tokens']))
            base += (
                'Task-specific anchor rules:\n'
                'A. Decide the single intent label from allowed intents. Provide a label whenever '
                'the sentence supports one; do not leave intent null merely because slots are uncertain.\n'
                'B. Treat slot annotation as a dataset-specific subset of meaning: only mark '
                'explicit token spans with an allowed slot label. Do not add all semantically salient words.\n'
                'C. For every slot, choose its category first, then locate the smallest exact '
                'contiguous span in ORIGINAL indexed tokens. Preserve separate occurrences, '
                'including repeated slots of the same category; never merge them automatically.\n'
                'D. Output spans [start,end) with 0-based indices and end exclusive. '
                'The surface must be the exact original tokens joined by spaces.\n'
                'E. Recheck the chosen label against the chosen tokens, remove invented or duplicate '
                'spans, use links=[]. Do not infer slots from world knowledge.\n'
                'ORIGINAL TOKEN INDEX TABLE:\n' + indexed + '\n'
            )
        elif case['dataset'] == 'term':
            base += (
                'Task-specific concept anchor rules:\n'
                'A. Interpret the meaning of the original mention, not just its spelling.\n'
                'B. Compare that meaning with the aliases and IDs of ALL candidate concepts.\n'
                'C. Select exactly one candidate ID when there is an equivalent or directly '
                'supported meaning. Do not require verbatim alias equality.\n'
                'D. Do not turn symptoms into an unsupported diagnosis, or confuse related '
                'with equivalent concepts.\n'
                'E. Return empty units only if none of the candidates is supported; '
                'keep intent_id=null, token_span=null and links=[].\n'
            )
    if method in ('fewshot','ours','ours_v1'):
        base += ('Examples for training-free in-context mapping (not test answers): '+
                 json.dumps(case['examples'][:2],ensure_ascii=False)+'\n')
    user = ('Speaker: '+case['speaker']+'\nInput text: '+case['text'])
    if case['dataset'] == 'dialect':
        user += '\nTokens: '+json.dumps(case['tokens'],ensure_ascii=False)
    return base, user

def decode_json(raw:str) -> dict:
    m = re.search(r'```(?:json)?\s*(.*?)```',raw,re.I|re.S)
    if m:
        raw = m.group(1)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        st,en = raw.find('{'),raw.rfind('}')
        if st<0 or en<st:
            raise ValueError('模型没有返回JSON对象')
        obj = json.loads(raw[st:en+1])
    if not isinstance(obj,dict):
        raise ValueError('模型没有返回JSON对象')
    return obj

def validate_and_normalize(pred:dict,case:dict) -> dict:
    if not isinstance(pred.get('shared_semantics'),dict):
        raise ValueError('shared_semantics缺失')
    sem = pred['shared_semantics']
    if not isinstance(sem.get('units'),list):
        raise ValueError('units不是列表')
    target = schema(case)
    if case['dataset']=='dialect':
        valid_intents = {'intent:'+c for c in case['catalog']['intents']}
        valid_ids = {'slot:'+c for c in case['catalog']['slots']}
    else:
        valid_intents = {None}
        valid_ids = {c['id'] for c in case['catalog']}
    intent=sem.get('intent_id')
    # 方言空意图表示模型未确定，不属于HTTP/JSON等技术错误；仍保留可用槽位。
    if intent is not None and intent not in valid_intents:
        raise ValueError('非法意图标签: '+str(intent))
    target['shared_semantics']['intent_id']=intent
    mapping={}
    for i,unit in enumerate(sem['units']):
        if not isinstance(unit,dict):
            raise ValueError('unit应为对象')
        cid=unit.get('concept_id')
        if cid not in valid_ids:
            raise ValueError('非法概念ID: '+str(cid))
        category='slot' if case['dataset']=='dialect' else 'term'
        if unit.get('category') != category:
            raise ValueError('unit的category不正确')
        span=unit.get('token_span')
        if category=='slot':
            n=len(case['tokens'])
            if not(isinstance(span,list) and len(span)==2 and all(type(t)==int for t in span)
                   and 0<=span[0]<span[1]<=n):
                raise ValueError('token_span不合法')
            surface=' '.join(case['tokens'][span[0]:span[1]])
        else:
            if len(sem['units'])>1:
                raise ValueError('术语任务最多一个概念')
            span=None
            surface=str(unit.get('surface') or case['text'])
        new_id=f'u{i+1}'
        mapping[str(unit.get('unit_id',new_id))]=new_id
        target['shared_semantics']['units'].append({'unit_id':new_id,'category':category,
                'concept_id':cid,'surface':surface,'token_span':span,
                'qualifiers':unit.get('qualifiers') if isinstance(unit.get('qualifiers'),dict) else {}})
    # 公开数据集未提供关系标注：不把生成links作为有金标准的评价目标
    if isinstance(sem.get('links'),list):
        for link in sem['links']:
            if not isinstance(link,dict):
                continue
            h=mapping.get(str(link.get('head')))
            t=mapping.get(str(link.get('tail')))
            if h and t and isinstance(link.get('predicate'),str):
                target['shared_semantics']['links'].append({'head':h,'predicate':link['predicate'],'tail':t})
    if isinstance(pred.get('evidence'),list):
        for e in pred['evidence']:
            if not isinstance(e,dict):
                continue
            text=str(e.get('text_span') or '')
            uid=mapping.get(str(e.get('unit_id')))
            if uid and text and text in case['text']:
                target['evidence'].append({'unit_id':uid,'text_span':text})
    if case['dataset'] == 'dialect' and intent is None:
        target['mapping_status'] = 'partial' if target['shared_semantics']['units'] else 'unmapped'
    else:
        target['mapping_status'] = 'mapped' if intent is not None or target['shared_semantics']['units'] else 'unmapped'
    return target

def predict_with_raw(case:dict,method:str) -> tuple[dict,str]:
    """供实验保存原始LLM响应，避免失败后无法定位原因。"""
    if method == 'normalize':
        original=case['text']
        standard=llm('Rewrite in clear standard language without adding or deleting facts. Output only text.',original)
        system,user=prompt_for(case,method)
        user += '\nStandard language reference (original is authoritative): '+standard
    else:
        system,user=prompt_for(case,method)
    raw = llm(system,user)
    return validate_and_normalize(decode_json(raw),case), raw

def predict_one(case:dict,method:str) -> dict:
    pred, _ = predict_with_raw(case,method)
    return pred
