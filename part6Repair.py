# -*- coding: utf-8 -*-
"""Publication version: shared configuration, no embedded credentials."""
from __future__ import annotations
import hmi_config as _release

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = _release.ROOT
BASE = _release.STAGE2
DETECTION_DIR = _release.STAGE2 / "cognitive_detection_v4"
OUTPUT = _release.STAGE2 / "internal_repair_v4_direct"
DATASETS = ('meditod', 'kddres')
# 直接使用Part5每轮每属性的原始不确定性观察；不去重、不筛选最终未关闭问题
DEDUP_ACTIVE_BY_ATTRIBUTE = False
ONLY_FINAL_OPEN_ISSUES = False  # 禁止使用open_issues作样本筛选
RUN_MODE = _release.MODE
MODEL_NAME = _release.model_name()
CONTEXT_TURNS = 30  # past history ONLY, max 30 utterances
MAX_RETRIES = 3
JSON_REPAIR_ATTEMPTS = 2
VERSION = '4.1-direct-from-part5-observations'
ACTIONS = {'internal_repair', 'explain', 'clarify', 'confirm', 'ask', 'defer'}
LABELS = {'inconsistency', 'imprecision', 'incompleteness'}
GATE = {'sufficient', 'insufficient', 'conflicting', 'unknown'}


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


# 单轮封闭世界修复：每条属性观察只用对应检测时点及之前的历史文本。
def prompt(case: dict) -> tuple[str, str]:
    system = '''You perform CLOSED-WORLD INTERNAL SEMANTIC REPAIR on authentic dialogues.
Phase 2 ALREADY identified independent, non-exclusive inconsistency/imprecision/incompleteness labels.
Your task is NOT choosing an action or asking the user a question. Attempt maximum SAFE
partial or full semantic correction for EACH active dimension, using ONLY the provided
utterances available at the current detection time. NEVER infer later user replies.

For each original uncertainty dimension:
- inconsistency: reconcile ONLY with explicit scope/time/negation/correction evidence;
  if contradictory claims remain unreconciled, leave unresolved.
- imprecision: replace or narrow an ambiguous value ONLY with existing applicable evidence;
  do not fabricate exact dates, counts, identities or locations.
- incompleteness: recover required missing attribute/relationship ONLY if explicitly
  recorded in the available history for the same subject, event, time and conditions.
A single change may address multiple labels only if it logically does so.
Preserve correct original semantic content and uncertainty qualifiers such as 'about'.
Never claim a label resolved just by writing 'resolved'. A value unsupported by history
is not a valid repair. Do NOT generate clarification questions, simulations, or guesses.
If evidence supports no change, return no changes and all dimensions unresolved.
Use exact contiguous quoted text from a SINGLE historical utterance for every change;
identify turn_id. The output is a PROPOSED semantic patch on a COPY of the mapping,
not a verified true cognition state. Avoid replacing entire mapping structures.

Return STRICT JSON, no markdown:
{
 "before_value":"value of target semantic attribute according to supplied input",
 "after_value":"most conservative evidence-grounded repaired value, or same as before",
 "changes":[{
   "field":"specific semantic field/attribute path",
   "old_value":"old value or empty if missing",
   "new_value":"value supported by utterance",
   "evidence_turn_id":"turn id as provided",
   "evidence_quote":"exact contiguous substring from that utterance",
   "reason":"why evidence supports this change and target entity/time match",
   "addresses_dimensions":["inconsistency|imprecision|incompleteness"]
 }],
 "claimed_resolved_dimensions":["only dimensions actually resolved by changes"],
 "remaining_dimensions":["active dimensions not resolved"],
 "rationale":"brief explanation including why other labels cannot be resolved",
 "limitations":"explicit ambiguity/conflict/remaining missing information"
}'''
    payload = {k:v for k,v in case.items() if k not in ('observations','final_open_issue')}
    return system, json.dumps(payload, ensure_ascii=False)


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


def call_and_validate(case):
    system,user=prompt(case)
    raw=model_call(system,user)
    for attempt in range(JSON_REPAIR_ATTEMPTS+1):
        try:
            return validate(raw,case)
        except (ValueError,TypeError,KeyError) as exc:
            if attempt==JSON_REPAIR_ATTEMPTS:
                raise
            correction={'error':str(exc),'previous_output':raw,'case':case}
            raw=model_call(system,'Your previous output violated format/evidence rules. Correct it. Return full JSON.\n'+json.dumps(correction,ensure_ascii=False))


def main():
    _release.preflight('worker')
    cases_by_ds={}
    for ds in DATASETS:
        cases,skip=prepare(ds)
        cases_by_ds[ds]=cases
        print(f'[{ds}] Part5属性观察={len(cases)}; 标签数={sum(len(c["labels"]) for c in cases)}; 标签组合={dict(Counter("+".join(c["labels"]) for c in cases))}; skip={skip}')
    if RUN_MODE=='inspect':
        print('检查模式结束，设 RUN_MODE="real" 开始修复调用。')
        return
    if RUN_MODE!='real': raise ValueError('RUN_MODE必须是inspect或real')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest = [{'case_id': c['case_id'], 'dataset': c['dataset'],
                 'dialogue_id': c['dialogue_id'], 'response_turn_id': c['response_turn_id'],
                 'attribute': c['attribute'], 'labels': c['labels']}
                for cases in cases_by_ds.values() for c in cases]
    jsonl_write_atomic(OUTPUT / 'fixed_repair_cases.jsonl', manifest)
    report={}
    for ds,cases in cases_by_ds.items():
        outfile=OUTPUT/f'{ds}_internal_repairs.jsonl'
        cached={r.get('case_id'):r for r in jsonl_read(outfile)} if outfile.exists() else {}
        results=[]
        for i,case in enumerate(cases,1):
            sy,us=prompt(case)
            fp=hashlib.sha256((VERSION+'\n'+MODEL_NAME+'\n'+sy+'\n'+us).encode()).hexdigest()
            prev=cached.get(case['case_id'])
            if prev and prev.get('input_fingerprint')==fp and prev.get('error') is None:
                outcome=prev['repair']; err=None; origin='cached'
            else:
                origin='run'
                try: outcome=call_and_validate(case);err=None
                except (ValueError,TypeError,KeyError,RuntimeError,json.JSONDecodeError) as e:
                    outcome=None;err=str(e)
            row={k:v for k,v in case.items() if k!='history'}
            row.update({'method':'Ours','repair':outcome,'error':err,
                        'input_fingerprint':fp,'version':VERSION,'model':MODEL_NAME,
                        'used_future_turns':False,'changed_original_files':False})
            results.append(row);jsonl_write_atomic(outfile,results)
            print(f'[{ds}] {i}/{len(cases)} {origin}: '+(f'ERROR {err[:130]}' if err else f'{outcome["repair_status"]}, changes={len(outcome["changes"])}'))
        good=[r for r in results if r['repair'] is not None and r['error'] is None]
        report[ds]={'cases':len(cases),'successful_calls':len(good),'failed':len(cases)-len(good),
                    'repair_status_counts':dict(Counter(x['repair']['repair_status'] for x in good)),
                    'proposed_changes':sum(len(x['repair']['changes']) for x in good),
                    'claimed_resolutions_not_independently_verified':sum(len(x['repair']['claimed_resolved_dimensions']) for x in good)}
    json_write_atomic(OUTPUT/'repair_report.json',{'version':VERSION,'datasets':report,
       'note':'Each Part5 labeled attribute observation is a distinct evaluation case; no deduplication or final-state filtering. Offline closed-world repair proposals. Valid evidence quotes are necessary but not sufficient for semantic validity; independent evaluation required. No user simulation.'})
    print('结果保存于',OUTPUT)


if __name__=='__main__':
    main()
