# -*- coding: utf-8 -*-
"""Publication version: shared configuration, no embedded credentials."""
from __future__ import annotations
import hmi_config as _release
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = _release.ROOT
BASE = _release.STAGE2
DETECTION_DIR = _release.STAGE2 / "cognitive_detection_v4"
OUTPUT = _release.STAGE2 / "internal_repair_baselines_v4_standalone"
DATASETS = ('meditod','kddres')
DEDUP_ACTIVE_BY_ATTRIBUTE = False
ONLY_FINAL_OPEN_ISSUES = False
RUN_MODE = _release.MODE
MODEL_NAME = _release.model_name()
CONTEXT_TURNS = 30
MAX_RETRIES = 3
JSON_REPAIR_ATTEMPTS = 2
LABELS = {'inconsistency','imprecision','incompleteness'}
METHODS = ('B1_direct','B2_context','B3_retrieval')
TOP_K = 5
VERSION = '7.2-standalone-part5-v4'

def jsonl_read(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f'找不到 {path}\n请先运行第二阶段V4并确认目录')
    with path.open('r', encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]

def jsonl_write_atomic(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.replace(path)

def json_write_atomic(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def prepare(ds: str) -> tuple[list[dict], dict]:
    """Construct case-level tasks from V4 attribute_uncertainties, never peek ahead."""
    detections = jsonl_read(DETECTION_DIR / f'{ds}_cognitive_differences.jsonl')
    dialogs = jsonl_read(BASE / ds / 'standard_dialogues.jsonl')
    turns_index = {str(d['dialogue_id']): d['turns'] for d in dialogs}
    mapping_file = BASE / ds / 'ours_v2_mappings.jsonl'
    mappings = jsonl_read(mapping_file)
    mindex = {(str(r.get('dialogue_id')), str(r.get('turn_id'))): r for r in mappings}
    out, skip = [], Counter()
    for rec in detections:
        res = rec.get('result')
        if rec.get('error') is not None or not isinstance(res, dict):
            skip['failed_v4'] += 1
            continue
        did = str(rec.get('dialogue_id'))
        turns = turns_index.get(did)
        if not turns:
            skip['missing_dialogue'] += 1
            continue
        rid = str(rec.get('response_turn_id'))
        indices = [i for i, t in enumerate(turns) if str(t.get('turn_id')) == rid]
        if len(indices) != 1:
            skip['turn_not_found_or_duplicated'] += 1
            continue
        i = indices[0]
        history = turns[max(0, i + 1 - CONTEXT_TURNS):i + 1]
        safe_history = []
        for t in history:
            mr = mindex.get((did, str(t.get('turn_id'))), {})
            mapping = mr.get('mapping') if mr.get('text') == t.get('text') else None
            safe_history.append({'turn_id': t.get('turn_id'), 'speaker': t.get('speaker'),
                                 'speaker_role': t.get('speaker_role'), 'text': t['text'],
                                 'shared_semantics': (mapping or {}).get('shared_semantics', {})})
        for attr in res.get('attribute_uncertainties', []):
            flags = attr.get('uncertainty') or {}
            active = [k for k in sorted(LABELS) if flags.get(k) is True]
            if not active:
                continue
            attribute = str(attr.get('attribute', '')).strip()
            if not attribute:
                skip['missing_attribute'] += 1
                continue
            scope = {'dataset': ds, 'dialogue_id': did, 'response_turn_id': rec['response_turn_id'],
                     'attribute': attribute}
            cid = 'repair_' + hashlib.sha256(json.dumps(scope, sort_keys=True,
                ensure_ascii=False).encode('utf-8')).hexdigest()[:16]
            out.append({'case_id': cid, **scope,
                        'source_turn_id': rec.get('source_turn_id'),
                        'source_text': rec.get('source_text'),
                        'response_text': rec.get('response_text'),
                        'labels': active, 'detected_attribute': attr,
                        'task_requirements': [x for x in res.get('task_requirements', [])
                                              if x.get('attribute') == attribute],
                        'history': safe_history,
                        'sample_ends_here': i == len(turns)-1,
                        'v4_detector_version': '4.0-multilabel-attribute'})
    if DEDUP_ACTIVE_BY_ATTRIBUTE:
        # 不能将逐轮累计的同一属性误认为多个独立修复任务。
        # 按首次出现时点生成任务，不读取后续真实回答。
        groups = {}
        for c in out:
            key = (c['dialogue_id'], c['attribute'])
            group = groups.get(key)
            if group is None:
                groups[key] = {'case': c, 'observations': []}
                group = groups[key]
            group['observations'].append({
                'response_turn_id': c['response_turn_id'],
                'labels': c['labels'],
                'observed_value': c['detected_attribute'].get('observed_value','')
            })
            # 固定首次检测时点，禁止后续轮次信息泄漏
        out = []
        for group in groups.values():
            c = group['case']
            c['observations'] = group['observations']
            c['first_detected_turn_id'] = group['observations'][0]['response_turn_id']
            c['observation_count'] = len(group['observations'])
            # case_id 以对话/属性为单位，而非以轮次为单位
            scope = {'dataset': ds, 'dialogue_id': c['dialogue_id'], 'attribute': c['attribute']}
            c['case_id'] = 'repair_' + hashlib.sha256(json.dumps(
                scope, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
            out.append(c)
        skip['repeated_observations_collapsed'] = sum(
            len(g['observations']) - 1 for g in groups.values())
    if ONLY_FINAL_OPEN_ISSUES:
        path = DETECTION_DIR / f'{ds}_open_issues.jsonl'
        if not path.exists():
            raise FileNotFoundError(f'需先生成V4最终未解决问题清单：{path}')
        open_issues = jsonl_read(path)
        open_index = {(str(x['dialogue_id']), str(x['attribute'])):x for x in open_issues
                      if any((x.get('uncertainty') or {}).values())}
        filtered = []
        for c in out:
            key = (c['dialogue_id'], c['attribute'])
            issue = open_index.get(key)
            if not issue:
                skip['already_resolved_or_not_finally_open'] += 1
                continue
            # 最终未关闭状态可能与最后一个检测结果不同，必须以状态文件为准。
            c['labels'] = [x for x in sorted(LABELS) if (issue.get('uncertainty') or {}).get(x) is True]
            c['final_open_issue'] = issue
            did = c['dialogue_id']
            turns = turns_index[did]
            # 在试验截断点规划下一步，不读取数据集中未采样的未来回答。
            final_idx = len(turns)-1
            c['response_turn_id'] = turns[final_idx].get('turn_id', final_idx)
            c['history'] = []
            for t in turns[max(0,final_idx+1-CONTEXT_TURNS):final_idx+1]:
                mr = mindex.get((did,str(t.get('turn_id'))),{})
                mapping = mr.get('mapping') if mr.get('text') == t.get('text') else None
                c['history'].append({'turn_id': t.get('turn_id'),
                    'speaker':t.get('speaker'), 'speaker_role':t.get('speaker_role'),
                    'text':t.get('text'),
                    'shared_semantics':(mapping or {}).get('shared_semantics',{})})
            c['sample_ends_here'] = True
            c['observation_truncated'] = True
            c['detected_attribute']['uncertainty'] = issue['uncertainty']
            c['detected_attribute']['interaction_status'] = issue.get('interaction_status','unknown')
            c['detected_attribute']['reasons'] = issue.get('reasons',{})
            filtered.append(c)
        skip['filtered_not_finally_open'] = len(out)-len(filtered)
        out = filtered
    expected = sum(
        1 for r in detections if r.get('error') is None and isinstance(r.get('result'), dict)
        for a in r['result'].get('attribute_uncertainties', [])
        if any((a.get('uncertainty') or {}).get(k) is True for k in LABELS)
        and str(a.get('attribute', '')).strip()
    )
    if expected != len(out):
        raise ValueError(f'Part5含标签属性记录={expected}，但可构建修复案例={len(out)}。'
                         f'请核对对话与映射文件是否属于同一次运行；skip={dict(skip)}')
    return out, dict(skip)

def model_call(system: str, user: str) -> dict:
    return _release.chat_json(system,user)

def _original_value(case):
    return case.get('detected_attribute', {}).get('observed_value', '')

def validate(raw:dict, case:dict)->dict:
    if not isinstance(raw,dict):
        raise ValueError('模型输出不是JSON对象')
    changes=raw.get('changes')
    if not isinstance(changes,list) or len(changes)>12:
        raise ValueError('changes必须是最多12项的数组')
    allowed=set(case['labels']); hist={str(t['turn_id']):str(t['text']) for t in case['history']}
    validated=[]
    for c in changes:
        if not isinstance(c,dict): raise ValueError('change不是对象')
        tid=str(c.get('evidence_turn_id',''))
        quote=c.get('evidence_quote','')
        if tid not in hist or not isinstance(quote,str) or not quote.strip() or quote not in hist[tid]:
            raise ValueError('修复引用未出现在对应历史发言的连续原文中')
        new=c.get('new_value')
        if new is None or not str(new).strip():
            raise ValueError('修复值为空')
        field=c.get('field')
        if not isinstance(field,str) or not field.strip():
            raise ValueError('缺少修复属性路径')
        dims=c.get('addresses_dimensions',[])
        if not isinstance(dims,list) or not dims or any(x not in allowed for x in dims):
            raise ValueError('修复维度必须是本案例原有标签')
        validated.append({'field':field.strip(),'old_value':c.get('old_value',''),
                          'new_value':new,'evidence_turn_id':tid,'evidence_quote':quote,
                          'reason':str(c.get('reason','')),'addresses_dimensions':sorted(set(dims))})
    resolved=raw.get('claimed_resolved_dimensions',[])
    remain=raw.get('remaining_dimensions',[])
    if not isinstance(resolved,list) or not isinstance(remain,list):
        raise ValueError('修复维度列表格式错误')
    if set(resolved)&set(remain) or set(resolved)|set(remain)!=allowed:
        raise ValueError('resolved和remaining必须不重叠且覆盖全部原始维度')
    justified=set(x for c in validated for x in c['addresses_dimensions'])
    if set(resolved)-justified:
        raise ValueError('声明解决的维度没有对应具体修复变更及证据')
    if not validated and resolved:
        raise ValueError('无具体修复时不能声称已解决')
    before=_original_value(case)
    after=raw.get('after_value',before)
    if after is None: after=before
    if not validated and after!=before:
        raise ValueError('无证据变更时不得改写原有语义')
    return {'before_value':before, 'after_value':after, 'changes':validated,
            'claimed_resolved_dimensions':sorted(set(resolved)),
            'remaining_dimensions':sorted(set(remain)),
            'repair_status':('full_candidate' if not remain else 'partial_candidate' if resolved else 'unresolved'),
            'rationale':str(raw.get('rationale','')),'limitations':str(raw.get('limitations','')),
            'requires_external_information':bool(remain),
            'validation_level':'quote_and_structure_only_not_semantic_gold'}

def terms(text):
    """Language-neutral-ish lexical matching (English words + Chinese ngrams)."""
    text = str(text).lower()
    en = re.findall(r'[a-z][a-z0-9_]*', text)
    cn = re.findall(r'[\u3400-\u9fff]+', text)
    grams = [chunk[i:i+2] for chunk in cn for i in range(max(0,len(chunk)-1))]
    return set(en + grams)


def retrieve(case, k=TOP_K):
    """Plain lexical retrieval; never reads any turn after the detection point."""
    hist = case['history']
    current_id = str(case['response_turn_id'])
    query = ' '.join([str(case['attribute']), str(case.get('source_text') or ''),
                      str(case.get('response_text') or '')])
    q = terms(query)
    scored = []
    for i, t in enumerate(hist):
        tok = terms(t['text'])
        score = len(tok & q) / max(1, len(q))
        scored.append((score, i, t))
    selected = sorted(scored, key=lambda x: (-x[0], -x[1]))[:k]
    # Include source/response turns if they are in history, within same k-plus-2 budget.
    picked = {str(x[2]['turn_id']): x[2] for x in selected}
    for t in hist:
        if str(t['turn_id']) in {current_id, str(case.get('source_turn_id'))}:
            picked[str(t['turn_id'])] = t
    order = {str(t['turn_id']):i for i,t in enumerate(hist)}
    return sorted(picked.values(), key=lambda t:order[str(t['turn_id'])])


def allowed_history(case, method):
    hist = case['history']
    if method == 'B1_direct':
        ids = {str(case['response_turn_id']), str(case.get('source_turn_id'))}
        return [t for t in hist if str(t['turn_id']) in ids]
    if method == 'B2_context':
        return hist
    if method == 'B3_retrieval':
        return retrieve(case)
    raise ValueError(method)


def prompt(case, method, allowed):
    intro = '''You are a baseline CLOSED-WORLD SEMANTIC REPAIR MODEL for authentic dialogue.
Part5 has ALREADY detected independent, possibly overlapping uncertainty labels.
Your task is to APPLY the maximum evidence-supported semantic improvements to the
TARGET attribute, not to select a conversational action. No clarification questions,
no imaginary user answers, no facts beyond the provided utterances.
For inconsistency: resolve only when specific correction/scope/time evidence supports it.
For imprecision: narrow only as much as explicit evidence warrants.
For incompleteness: restore only facts actually present in allowed evidence.
You may repair some dimensions and leave others unresolved.
Preserve approximation/negation and subject/time identity. No unjustified exact values.
For every change quote an EXACT contiguous substring of ONE supplied utterance and its turn_id.
No patch without valid quote. If nothing can be fixed, return changes=[].
Return exactly JSON of the form:
{
"before_value":"original value or empty",
"after_value":"evidence-supported improved value, or unchanged",
"changes":[{"field":"specific target field","old_value":"old or empty",
"new_value":"supported value","evidence_turn_id":"supplied turn id",
"evidence_quote":"exact contiguous quote","reason":"brief","addresses_dimensions":["imprecision"]}],
"claimed_resolved_dimensions":[], "remaining_dimensions":[],
"rationale":"brief", "limitations":"remaining uncertainty"
}
The dimension lists must partition the active labels, and each claimed resolution
must have at least one change linked to it. Do not output markdown.'''
    # All methods share case labels, original observed value and task requirements.
    payload = {'case_id':case['case_id'],'dataset':case['dataset'],
       'dialogue_id':case['dialogue_id'],'response_turn_id':case['response_turn_id'],
       'attribute':case['attribute'], 'labels':case['labels'],
       'detected_attribute':case['detected_attribute'],
       'task_requirements':case.get('task_requirements',[]),
       'source_text':case.get('source_text'), 'response_text':case.get('response_text')}
    if method == 'B1_direct':
        payload['current_exchange']=[{'turn_id':t['turn_id'],'speaker':t['speaker'],
                                      'text':t['text']} for t in allowed]
    elif method == 'B2_context':
        payload['dialogue_history']=[{'turn_id':t['turn_id'],'speaker':t['speaker'],
                                      'text':t['text']} for t in allowed]
    else:
        payload['retrieved_evidence']=[{'turn_id':t['turn_id'],'speaker':t['speaker'],
                                       'text':t['text']} for t in allowed]
    return intro, json.dumps(payload,ensure_ascii=False)


def run_one(case,method,allowed):
    # Standalone API client and matching structural validation; only baseline-visible evidence is admissible.
    restricted = dict(case, history=allowed)
    system,user=prompt(case,method,allowed)
    raw=model_call(system,user)
    for retry in range(JSON_REPAIR_ATTEMPTS+1):
        try:
            return validate(raw,restricted)
        except (ValueError,TypeError,KeyError) as exc:
            if retry==JSON_REPAIR_ATTEMPTS: raise
            repair_request={'validation_error':str(exc),'prior_output':raw,
                            'allowed_utterances':[{'turn_id':x['turn_id'],'text':x['text']} for x in allowed]}
            raw=model_call(system,user+'\nFix validation error and return full JSON:\n'+json.dumps(repair_request,ensure_ascii=False))


def main():
    _release.preflight('worker')
    by_ds={}
    for ds in DATASETS:
        cases,_=prepare(ds)
        by_ds[ds]=cases
        print(f'[{ds}] Part5 labeled attribute cases={len(cases)}; labels={sum(len(c["labels"]) for c in cases)}')
    # Standalone: Part5 is the only source of uncertainty cases.
    # Consistency manifest generated from these same fixed Part5 records.
    manifest=[{'dataset':c['dataset'],'case_id':c['case_id'],
               'dialogue_id':c['dialogue_id'],'response_turn_id':c['response_turn_id'],
               'attribute':c['attribute'],'labels':c['labels']}
              for cases in by_ds.values() for c in cases]
    if RUN_MODE=='real':
        jsonl_write_atomic(OUTPUT/'fixed_repair_cases.jsonl',manifest)
    if RUN_MODE=='inspect':
        for ds,cases in by_ds.items():
            for m in METHODS:
                lens=[len(allowed_history(c,m)) for c in cases]
                print(f'[{ds}] {m}: {len(cases)} cases, evidence utterances range={min(lens,default=0)}-{max(lens,default=0)}')
        return
    if RUN_MODE!='real':raise ValueError('RUN_MODE must be inspect or real')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    report={}
    for ds,cases in by_ds.items():
        for method in METHODS:
            dst=OUTPUT/f'{ds}_{method}_internal_repairs.jsonl'
            cached={r['case_id']:r for r in jsonl_read(dst)} if dst.exists() else {}
            rows=[]
            for i,case in enumerate(cases,1):
                evidence=allowed_history(case,method)
                system,user=prompt(case,method,evidence)
                fp=hashlib.sha256((VERSION+'\n'+MODEL_NAME+'\n'+system+'\n'+user).encode()).hexdigest()
                prev=cached.get(case['case_id'])
                if prev and prev.get('input_fingerprint')==fp and prev.get('error') is None:
                    result=prev['repair'];err=None;mode='cached'
                else:
                    mode='run'
                    try:result=run_one(case,method,evidence);err=None
                    except (ValueError,TypeError,KeyError,RuntimeError,json.JSONDecodeError) as exc:
                        result=None;err=str(exc)
                row={k:v for k,v in case.items() if k!='history'}
                row.update({'method':method,'repair':result,'error':err,
                            'input_fingerprint':fp,'model':MODEL_NAME,'version':VERSION,
                            'available_evidence_turn_ids':[x['turn_id'] for x in evidence],
                            'used_future_turns':False,'changed_original_files':False})
                rows.append(row)
                jsonl_write_atomic(dst,rows)
                print(f'[{ds}/{method}] {i}/{len(cases)} {mode}: '+
                      (f'ERROR {err[:110]}' if err else f'{result["repair_status"]}, changes={len(result["changes"])}'))
            good=[x for x in rows if x['error'] is None and x['repair'] is not None]
            report[f'{ds}/{method}']={'cases':len(cases),'evaluated':len(good),
              'errors':len(cases)-len(good),'statuses':dict(Counter(x['repair']['repair_status'] for x in good)),
              'proposed_changes_not_verified':sum(len(x['repair']['changes']) for x in good)}
    json_write_atomic(OUTPUT/'baseline_report.json',{
      'version':VERSION,'cases_source':'Part5 V4 cognitive_differences.jsonl; no Part6 dependency',
      'note':'Three internal semantic repair baselines on identical Part5 cases; proposals not independently verified.',
      'methods':report})
    print('Baseline results:', OUTPUT)

if __name__=='__main__':main()
