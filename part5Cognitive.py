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
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = _release.ROOT
INPUT = _release.STAGE2
OUTPUT = _release.STAGE2 / "cognitive_detection_v4"
DATASETS = ('meditod', 'kddres')
RUN_MODE = _release.MODE
MODEL_NAME = _release.model_name("detector")
CONTEXT_TURNS = 10
DETECTOR_VERSION = '4.0-multilabel-attribute'
VALIDATION_REPAIR_ATTEMPTS = 2
MAX_RETRIES = 3
# True: a mapping error excludes that exchange rather than presenting missing mapping as cognitive incompleteness.
REQUIRE_BOTH_MAPPINGS = True

DIMENSIONS = {'inconsistency', 'imprecision', 'incompleteness'}
LEVELS = {'intent', 'concept', 'attribute', 'relation', 'context', 'evidence', 'response'}
STATUS = {'assessable', 'insufficient_evidence', 'not_applicable'}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f'缺少文件：{path}\n先运行 python run_experiments.py map-dialogues --execute 完成映射')
    with path.open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def build_exchanges(ds: str):
    folder = INPUT / ds
    dialogs = load_jsonl(folder / 'standard_dialogues.jsonl')
    mapping_path = folder / 'ours_v2_mappings.jsonl'
    mappings = load_jsonl(mapping_path) if mapping_path.exists() else []
    index = {}
    for row in mappings:
        key = (str(row.get('dialogue_id')), row.get('turn_id'))
        if key in index:
            raise ValueError(f'{ds} 映射存在重复键：{key}')
        index[key] = row
    exchanges, excluded = [], Counter()
    for d in dialogs:
        did = str(d['dialogue_id'])
        turns = d['turns']
        for i in range(1, len(turns)):
            prev, curr = turns[i-1], turns[i]
            # Do not assess a same-speaker continuation as an answer.
            if prev['speaker'] == curr['speaker']:
                excluded['same_speaker'] += 1
                continue
            pm = index.get((did, prev['turn_id']))
            cm = index.get((did, curr['turn_id']))
            if REQUIRE_BOTH_MAPPINGS and (not pm or not cm or not pm.get('mapping') or not cm.get('mapping')):
                excluded['missing_or_failed_mapping'] += 1
                continue
            # Prevent silent mixing of a different run's text.
            if pm and pm.get('text') != prev['text'] or cm and cm.get('text') != curr['text']:
                excluded['text_mismatch'] += 1
                continue
            exchange = {
                'dataset': ds, 'dialogue_id': did, 'source_turn_id': prev['turn_id'],
                'response_turn_id': curr['turn_id'],
                'source': {'speaker': prev['speaker'], 'speaker_role': prev['speaker_role'],
                           'text': prev['text'], 'mapping': pm.get('mapping') if pm else None},
                'response': {'speaker': curr['speaker'], 'speaker_role': curr['speaker_role'],
                             'text': curr['text'], 'mapping': cm.get('mapping') if cm else None},
                'history': [
                    {'turn_id': t['turn_id'], 'speaker': t['speaker'], 'text': t['text'],
                     'shared_semantics': (index.get((did, t['turn_id'])) or {}).get('mapping', {}).get('shared_semantics', {})
                       if isinstance((index.get((did, t['turn_id'])) or {}).get('mapping'), dict) else {}}
                    for t in turns[max(0, i-1-CONTEXT_TURNS):i-1]],
            }
            # Source and response roles are *contextual*, not semantic gold.
            exchanges.append(exchange)
    return exchanges, excluded, len(dialogs), len(mappings)


def make_prompt(ex: dict, state: dict):
    system = """You are an evidence-grounded auditor of THREE NON-EXCLUSIVE semantic uncertainty dimensions
in authentic conversations: inconsistency, imprecision, incompleteness. You are NOT making
claims about hidden mental states. First-stage semantic JSON is FALLIBLE; raw dialogue is primary evidence.

STAGE BOUNDARY: Detect and describe uncertainty ONLY. NEVER select repair/clarification/explanation/
confirmation actions. Dialogue management states (new question, awaiting answer, resolved) are NOT
uncertainty types, but a required semantic fact can remain incomplete while an answer is pending.
No need to accuse either speaker of misunderstanding. Do not invent unseen user turns or values.

EACH CURRENT TASK-RELEVANT ATTRIBUTE (concept/property/relation, including an item under a broader
attribute): independently test all three dimensions, even when another is already true:
- inconsistency: two *applicable*, simultaneous and unreconciled assertions about SAME entity,
  property, timeframe and conditions cannot both hold. A correction, changed state, hypothetical,
  and mere disagreement with a question's presupposition are not automatically conflicts.
- imprecision: a value EXISTS, but its ambiguity, range or conceptual granularity is insufficient
  for the CURRENT task requirement; name the specific precision gap and WHY required now.
- incompleteness: an explicitly needed semantic element/sub-element is not established by valid
  available evidence as of the response turn. This may coexist with imprecision when a broad
  value exists but another required sub-element is missing. Absence must not be inferred merely
  because an optional clinical or restaurant slot was never mentioned.

IMPORTANT: Evaluate ALL information available by the end of the response turn, including relevant
history, and distinguish not-yet-answered legitimate questions from communication failures.
An open question can have incomplete task information, but must be tagged as an awaiting response,
not a response failure. If an earlier requirement is fully supported by context, do not label it
missing merely because the CURRENT reply did not repeat the answer. If evidence insufficient to
know whether a requirement is due now, use undecidable, never fabricate task requirements.

MULTI-LABEL: for each attribute output all 3 booleans INDEPENDENTLY (any of 8 combinations).
For EVERY true flag give a DISTINCT causal reason and evidence references. Multiple flags may have
different sub-elements of the same parent attribute, list the precise sub-element in its reason.
A dimension=FALSE never implies all other dimensions=FALSE. Record no label if there is no task-
grounded basis. An unanswered question is not automatically a communicative error.

VERBATIM QUOTES: Each quote MUST be a contiguous exact substring of ONE raw utterance from source,
response or history (never concatenate speakers, modify punctuation or add ellipses). If no
verbatim evidence supports a true label, set that label false or mark undecidable. No fabricated
medical/restaurant ontology facts. Preserve approximate, negated, uncertain expressions.

TRACKING: You receive a list of OPEN ISSUE RECORDS from prior turns, along with an issue_id.
You may explicitly mark a previous issue resolved only if you cite a new or existing verbatim
utterance that establishes the missing information or settles the conflict. Missing information
at end of this sample is not a proven failure; sample may be truncated.

STRICT JSON only, no markdown, format:
{
 "status":"assessable|insufficient_evidence|not_applicable",
 "communicative_act":"question|answer|request|inform|confirm|clarify|other",
 "task_requirements":[{"attribute":"path","required_now":true,"target":"task condition",
   "basis_quote":"exact contiguous snippet","specificity":"required granularity or unspecified",
   "specificity_basis":"why needed now or unspecified"}],
 "attribute_uncertainties":[{
   "attribute":"semantic attribute path", "semantic_level":"intent|concept|attribute|relation|context|evidence|response",
   "observed_value":"verbatim or conservative description", "task_target":"required target",
   "uncertainty":{"inconsistency":false,"imprecision":false,"incompleteness":false},
   "reasons":{"inconsistency":"","imprecision":"","incompleteness":""},
   "evidence":{"source_quote":"","response_quote":"","context_quote":""},
   "source_type":"source_missing|mapping_omission|context_not_integrated|conflicting_evidence|underspecified_expression|unknown",
   "interaction_status":"awaiting_response|partially_answered|answered|unknown",
   "confidence":"high|medium|low"
 }],
 "resolved_issues":[{"issue_id":"EXACT previous issue id","reason":"evidence-based rationale","evidence_quote":"EXACT substring of ONE utterance"}],
 "unresolved":[{"attribute":"path","reason":"not enough evidence to classify"}],
 "brief_rationale":"..."
}

Only output attributes RELEVANT to explicit current task/utterance or continuing unresolved requirements.
Do not use interaction_status as one of the three uncertainty dimensions. Do not decide repair actions.
"""
    user = json.dumps({'exchange': ex, 'previous_open_issues': list(state.values())}, ensure_ascii=False)
    return system, user


def call_model(system: str, user: str) -> dict:
    return _release.chat_json(system,user,role="detector")


def validate(report: dict, ex: dict, state: dict):
    if not isinstance(report, dict) or report.get('status') not in STATUS:
        raise ValueError('status/JSON非法')
    for k in ('attribute_uncertainties', 'task_requirements', 'resolved_issues', 'unresolved'):
        if not isinstance(report.get(k, []), list):
            raise ValueError(k + '必须是列表')
    source = ex['source']['text']
    response = ex['response']['text']
    history = [x['text'] for x in ex['history']]
    all_text = [source, response, *history]

    def quote(v, texts, field):
        if v is None:
            return ''
        if not isinstance(v, str):
            raise ValueError(field + '必须是字符串')
        if v and not any(v in t for t in texts):
            raise ValueError(field + '引用不在单条原始发言中: ' + v[:100])
        return v

    reqs = []
    for v in report.get('task_requirements', []):
        if not isinstance(v, dict) or not str(v.get('attribute', '')).strip():
            raise ValueError('任务要求缺少属性')
        reqs.append({'attribute': str(v['attribute']).strip(),
                     'required_now': v.get('required_now') is True,
                     'target': str(v.get('target', '')),
                     'basis_quote': quote(v.get('basis_quote', ''), all_text, 'requirement.basis_quote'),
                     'specificity': str(v.get('specificity', '')),
                     'specificity_basis': str(v.get('specificity_basis', ''))})
    attr_rows = []
    seen = set()
    for a in report.get('attribute_uncertainties', []):
        if not isinstance(a, dict) or a.get('semantic_level') not in LEVELS:
            raise ValueError('属性语义层级非法')
        attr = str(a.get('attribute', '')).strip()
        if not attr or attr in seen:
            raise ValueError('属性缺失或重复: ' + attr)
        seen.add(attr)
        u = a.get('uncertainty')
        reasons = a.get('reasons')
        e = a.get('evidence')
        if not isinstance(u, dict) or not isinstance(reasons, dict) or not isinstance(e, dict):
            raise ValueError('不确定性/原因/证据对象缺失')
        if any(not isinstance(u.get(dim), bool) for dim in DIMENSIONS):
            raise ValueError('三维标签必须都是boolean')
        ev = {
            'source_quote': quote(e.get('source_quote', ''), [source], 'source_quote'),
            'response_quote': quote(e.get('response_quote', ''), [response], 'response_quote'),
            'context_quote': quote(e.get('context_quote', ''), history, 'context_quote')
        }
        if any(u.values()) and not any(ev.values()):
            raise ValueError('任一不确定性为true时必须提供真实原文证据: ' + attr)
        for dim in DIMENSIONS:
            if u[dim] and not str(reasons.get(dim, '')).strip():
                raise ValueError(attr + ' 缺少 ' + dim + ' 独立判断依据')
        if report['status'] != 'assessable' and any(u.values()):
            raise ValueError('非assessable不能确认不确定性标签')
        attr_rows.append({
            'attribute': attr, 'semantic_level': a['semantic_level'],
            'observed_value': str(a.get('observed_value', '')),
            'task_target': str(a.get('task_target', '')),
            'uncertainty': {dim: u[dim] for dim in sorted(DIMENSIONS)},
            'reasons': {dim: str(reasons.get(dim, '')) for dim in sorted(DIMENSIONS)},
            'evidence': ev,
            'source_type': str(a.get('source_type', 'unknown')),
            'interaction_status': str(a.get('interaction_status', 'unknown')),
            'confidence': str(a.get('confidence', 'low'))
        })
    resolved = []
    seen_r = set()
    for v in report.get('resolved_issues', []):
        if not isinstance(v, dict):
            raise ValueError('resolved_issues元素非法')
        iid = str(v.get('issue_id', ''))
        if iid not in state:
            raise ValueError('解决的issue_id不在既有open issues中: ' + iid)
        q = quote(v.get('evidence_quote', ''), all_text, 'resolved.evidence_quote')
        if not q:
            raise ValueError('关闭问题必须提供证据')
        if iid not in seen_r:
            resolved.append({'issue_id': iid, 'reason': str(v.get('reason', '')),
                             'evidence_quote': q})
            seen_r.add(iid)
    dims = sorted({d for a in attr_rows for d in DIMENSIONS if a['uncertainty'][d]})
    return {
        'status': report['status'],
        'communicative_act': str(report.get('communicative_act', 'other')),
        'task_requirements': reqs,
        'attribute_uncertainties': attr_rows,
        'has_uncertainty': bool(dims),
        'uncertain_attribute_count': sum(any(a['uncertainty'].values()) for a in attr_rows),
        'dimension_label_count': sum(sum(a['uncertainty'].values()) for a in attr_rows),
        'dimensions': dims,
        'resolved_issues': resolved,
        'unresolved': report.get('unresolved', []),
        'brief_rationale': str(report.get('brief_rationale', '')),
    }


def fingerprint(system:str,user:str):
    return hashlib.sha256((DETECTOR_VERSION+'\n'+MODEL_NAME+'\n'+system+'\n'+user).encode('utf-8')).hexdigest()


def repair_result(system:str,user:str,ex:dict,state:dict):
    raw=call_model(system,user)
    for n in range(VALIDATION_REPAIR_ATTEMPTS+1):
        try:return validate(raw,ex,state),None
        except (ValueError,TypeError,KeyError) as e:
            if n==VALIDATION_REPAIR_ATTEMPTS: raise
            correction=("Your JSON failed STRICT validation: "+str(e)+". Correct only invalid fields. "
                "Every quote must be a contiguous exact substring of ONE utterance. "
                "Do not concatenate or use ellipses, do not invent claims. Return ONLY corrected full JSON.\n"+
                json.dumps({'previous_json':raw,'exchange':ex,'previous_open_issues':list(state.values())},ensure_ascii=False))
            raw=call_model(system,correction)
    raise RuntimeError('验证修复失败')


def issue_key(ds,did,attribute):
    raw='|'.join((ds,did,attribute))
    return 'iss_'+hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]


def advance_state(state: dict, ex: dict, result: dict):
    """Maintain a conservative attribute-level uncertainty ledger, NOT repair decisions."""
    state = {k: dict(v) for k, v in state.items()}
    events = []
    explicitly_resolved = {r['issue_id']: r for r in result.get('resolved_issues', [])}
    flagged_attrs = {a['attribute'] for a in result.get('attribute_uncertainties', [])
                     if any(a['uncertainty'].values())}
    # A cached or fresh model answer may contradict itself. Keep the active uncertainty.
    conflicted = {iid for iid in explicitly_resolved
                  if iid in state and state[iid]['attribute'] in flagged_attrs}
    for iid in sorted(conflicted):
        events.append({'event': 'resolution_conflict_ignored', 'issue_id': iid,
                       'attribute': state[iid]['attribute'], 'turn_id': ex['response_turn_id'],
                       'reason': '同一属性被同时标为已解决和仍不确定；保留不确定性'})
        explicitly_resolved.pop(iid)
    for iid, r in explicitly_resolved.items():
        item = state.pop(iid, None)
        if item:
            events.append({'event': 'resolved', 'issue_id': iid,
                           'attribute': item['attribute'], 'turn_id': ex['response_turn_id'],
                           'reason': r['reason'], 'evidence_quote': r['evidence_quote']})
    for a in result.get('attribute_uncertainties', []):
        if not any(a['uncertainty'].values()):
            continue
        iid = issue_key(ex['dataset'], ex['dialogue_id'], a['attribute'])
        prior = state.get(iid)
        item = {'issue_id': iid, 'attribute': a['attribute'],
                'uncertainty': a['uncertainty'], 'reasons': a['reasons'],
                'evidence': a['evidence'], 'interaction_status': a['interaction_status'],
                'opened_turn_id': prior['opened_turn_id'] if prior else ex['response_turn_id'],
                'last_observed_turn_id': ex['response_turn_id']}
        if prior is None:
            events.append({'event': 'opened', 'issue_id': iid, 'attribute': a['attribute'],
                           'uncertainty': a['uncertainty'], 'turn_id': ex['response_turn_id']})
        elif prior['uncertainty'] != a['uncertainty']:
            events.append({'event': 'updated', 'issue_id': iid, 'attribute': a['attribute'],
                           'from_uncertainty': prior['uncertainty'], 'to_uncertainty': a['uncertainty'],
                           'turn_id': ex['response_turn_id']})
        state[iid] = item
    return state, events


def write_jsonl(path:Path,rows:list):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(path)


def main():
    _release.preflight('detector')
    by_ds={}
    for ds in DATASETS:
        exs,exc,nd,nm=build_exchanges(ds)
        by_ds[ds]=exs
        print(f'[{ds}] 对话={nd}, 映射记录={nm}, 可评价跨角色问答={len(exs)}, 跳过={dict(exc)}')
    if RUN_MODE=='inspect':
        for ds,exs in by_ds.items():
            for ex in exs[:2]:
                print(ds,repr(ex['source']['text'][:70]),'->',repr(ex['response']['text'][:70]))
        print('inspect不调用模型；改RUN_MODE="real"进行V4三维多标签检测')
        return
    if RUN_MODE!='real':raise ValueError('RUN_MODE只能是inspect或real')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    overall={}
    for ds,exs in by_ds.items():
        path=OUTPUT/f'{ds}_cognitive_differences.jsonl'
        previously={}
        if path.exists():
            for row in load_jsonl(path):
                k=(str(row['dialogue_id']),row['response_turn_id'])
                previously[k]=row
        states=defaultdict(dict); rows=[];events=[]
        for idx,ex in enumerate(exs,1):
            did=ex['dialogue_id']; state=states[did]
            system,user=make_prompt(ex,state)
            sig=fingerprint(system,user)
            key=(did,ex['response_turn_id'])
            prior=previously.get(key)
            reused=bool(prior and prior.get('input_fingerprint')==sig and prior.get('error') is None and prior.get('result'))
            if reused:
                result=prior['result'];err=None
            else:
                try:result,err=repair_result(system,user,ex,state)
                except (RuntimeError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:
                    result=None;err=str(e)
            event_list=[]
            if result is not None:
                states[did],event_list=advance_state(state,ex,result)
                events.extend([{'dataset':ds,'dialogue_id':did,**e} for e in event_list])
            row={'dataset':ds,'dialogue_id':did,'source_turn_id':ex['source_turn_id'],
                'response_turn_id':ex['response_turn_id'],'source_speaker':ex['source']['speaker'],
                'response_speaker':ex['response']['speaker'],'source_text':ex['source']['text'],
                'response_text':ex['response']['text'],'result':result,'state_events':event_list,
                'error':err,'input_fingerprint':sig,'evaluator_model':MODEL_NAME}
            rows.append(row)
            # Preserve completed sequential rows on interruptions; resume only fingerprint-matched prefix.
            write_jsonl(path,rows)
            print(f'[{ds}] {idx}/{len(exs)} '+('cached' if reused else 'run')+' -> '+
                (('ERROR '+err[:110]) if err else f'{result["uncertain_attribute_count"]} uncertain attrs, '+
                 f'{result["dimension_label_count"]} labels; open={len(states[did])}'))
        write_jsonl(OUTPUT/f'{ds}_issue_events.jsonl',events)
        write_jsonl(OUTPUT/f'{ds}_open_issues.jsonl',[
            {'dataset':ds,'dialogue_id':did,**item} for did,state in states.items() for item in state.values()])
        valid = [r for r in rows if r['result'] is not None and r['error'] is None]
        labels = Counter(); combinations = Counter(); levels = Counter(); statuses = Counter()
        req_total = 0; unresolved_total = 0
        for row in valid:
            res = row['result']
            statuses[res['status']] += 1
            req_total += sum(v['required_now'] for v in res['task_requirements'])
            unresolved_total += len(res['unresolved'])
            for a in res['attribute_uncertainties']:
                active = tuple(sorted(k for k, v in a['uncertainty'].items() if v))
                if active:
                    combinations['+'.join(active)] += 1
                    levels[a['semantic_level']] += 1
                for d in active:
                    labels[d] += 1
        final_open = Counter('+'.join(sorted(k for k, v in x['uncertainty'].items() if v))
                             for state in states.values() for x in state.values())
        overall[ds] = {
            'candidate_exchanges': len(exs), 'successfully_evaluated': len(valid),
            'failed_or_pending': len(exs) - len(valid),
            'cases_with_uncertainty': sum(r['result']['has_uncertainty'] for r in valid),
            'uncertain_attribute_observations_by_dimension': dict(labels),
            'attribute_multilabel_combinations': dict(combinations),
            'by_semantic_level': dict(levels), 'statuses': dict(statuses),
            'final_open_issue_combinations': dict(final_open),
            'unique_issues_opened': sum(e['event']=='opened' for e in events),
            'issues_resolved': sum(e['event']=='resolved' for e in events),
            'required_now_attribute_records': req_total,
            'unresolved_records': unresolved_total,
        }

    dump_json(OUTPUT/'cognitive_report.json',{'detector_version':DETECTOR_VERSION,
        'note':'Provisional LLM independent per-dimension judgments; no repair decisions; open questions and uncertainty types recorded separately. Not gold.',
        'datasets':overall,'dimensions':sorted(DIMENSIONS)})
    print('V4输出目录:',OUTPUT)

if __name__=='__main__':
    main()
