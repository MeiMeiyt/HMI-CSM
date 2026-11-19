# -*- coding: utf-8 -*-
"""Publication version: shared configuration, no embedded credentials."""
from __future__ import annotations
import hmi_config as _release
import csv
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
DETECT = _release.STAGE2 / "cognitive_detection_v4"
OURS = _release.STAGE2 / "internal_repair_v4_direct"
BASELINES = _release.STAGE2 / "internal_repair_baselines_v4_standalone"
OUT = _release.STAGE2 / "repair_evaluation_v4"
DATASETS=('meditod','kddres')
METHODS=('Ours','B1_direct','B2_context','B3_retrieval')
DIMS=('inconsistency','imprecision','incompleteness')
RUN_MODE = _release.MODE
MODEL = _release.model_name("judge")
VERSION='1.0-independent-semantic-repair-judge'
MAX_RETRIES=3


def read_jsonl(p):
    if not p.exists(): raise FileNotFoundError(f'找不到输入文件: {p}')
    with p.open(encoding='utf-8-sig') as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(p)


def write_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(p)


def make_case_index(ds):
    """Independently reconstruct immutable Part5 cases; no Part6/7 imports."""
    rows=read_jsonl(DETECT/f'{ds}_cognitive_differences.jsonl')
    dialogs=read_jsonl(BASE/ds/'standard_dialogues.jsonl')
    mappings=read_jsonl(BASE/ds/'ours_v2_mappings.jsonl')
    dialog_idx={str(d['dialogue_id']):d['turns'] for d in dialogs}
    map_idx={(str(m.get('dialogue_id')),str(m.get('turn_id'))):m for m in mappings}
    cases={}
    for rec in rows:
        if rec.get('error') is not None or not isinstance(rec.get('result'),dict):continue
        did=str(rec['dialogue_id']); rid=str(rec['response_turn_id']); turns=dialog_idx[did]
        found=[i for i,t in enumerate(turns) if str(t.get('turn_id'))==rid]
        if len(found)!=1:raise ValueError(f'无法唯一匹配轮次: {ds}, {did}, {rid}')
        index=found[0]
        hist=[]
        for t in turns[max(0,index-29):index+1]:
            m=map_idx.get((did,str(t.get('turn_id'))),{})
            mp=m.get('mapping') if m.get('text')==t.get('text') else None
            hist.append({'turn_id':t.get('turn_id'),'text':t['text'],'speaker':t.get('speaker'),
                         'shared_semantics':(mp or {}).get('shared_semantics',{})})
        for a in rec['result'].get('attribute_uncertainties',[]):
            flags=a.get('uncertainty') or {}
            labels=sorted(k for k in DIMS if flags.get(k) is True)
            attr=str(a.get('attribute','')).strip()
            if not labels or not attr:continue
            scope={'dataset':ds,'dialogue_id':did,'response_turn_id':rec['response_turn_id'],'attribute':attr}
            cid='repair_'+hashlib.sha256(json.dumps(scope,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:16]
            if cid in cases:raise ValueError('Part5重复属性案例: '+cid)
            cases[cid]={**scope,'case_id':cid,'labels':labels,'detected_attribute':a,
                        'task_requirements':[x for x in rec['result'].get('task_requirements',[]) if x.get('attribute')==attr],
                        'history':hist,'source_text':rec.get('source_text'), 'response_text':rec.get('response_text')}
    return cases


def repair_file(ds,method):
    if method=='Ours':return OURS/f'{ds}_internal_repairs.jsonl'
    return BASELINES/f'{ds}_{method}_internal_repairs.jsonl'


def load_all():
    cases={}; outputs={}; problems=[]
    for ds in DATASETS:
        d=make_case_index(ds)
        cases.update(d)
        for method in METHODS:
            rows=read_jsonl(repair_file(ds,method))
            local={}
            for r in rows:
                cid=r.get('case_id')
                if cid in local:raise ValueError(f'重复结果: {method} {cid}')
                if cid not in d:raise ValueError(f'输出含Part5未定义的case: {method} {cid}')
                base=d[cid]
                if str(r.get('response_turn_id'))!=str(base['response_turn_id']) or r.get('attribute')!=base['attribute'] or sorted(r.get('labels',[]))!=base['labels']:
                    raise ValueError(f'输入不匹配: {method} {cid}')
                local[cid]=r
            for cid in d:
                if cid not in local:problems.append(f'{method} 缺少 {cid}')
                else:outputs[(cid,method)]=local[cid]
    if problems:raise ValueError('案例未对齐：\n'+'\n'.join(problems[:30]))
    return cases,outputs


def evidence_checks(case, output):
    repair=output.get('repair') or {}
    changes=repair.get('changes') or []
    if not isinstance(changes,list):changes=[]
    history={str(t['turn_id']):str(t['text']) for t in case['history']}
    allowed=set(str(x) for x in output.get('available_evidence_turn_ids',history.keys()))
    checks=[]
    for c in changes:
        tid=str(c.get('evidence_turn_id',''))
        q=c.get('evidence_quote','')
        ok=isinstance(q,str) and bool(q.strip()) and tid in history and tid in allowed and q in history[tid]
        checks.append({'quote_valid':ok,'evidence_turn_id':tid,'quote':q,'field':c.get('field',''),
                       'new_value':c.get('new_value'),'addresses_dimensions':c.get('addresses_dimensions',[])})
    return checks


def llm_call(system,user):
    return _release.chat_json(system,user,role="judge")


SYSTEM_PROMPT='''You are an INDEPENDENT, METHOD-BLIND evaluator of CLOSED-WORLD semantic repair.
You are evaluating an authentic dialogue at a fixed time point. Historical utterances after
that point are NOT available. Do not use outside medical, geographic, cultural, or restaurant
knowledge to infer missing facts. First-hand text and original unified semantics are evidence.
The Part5 uncertainty labels are the initial problems, NOT proof that any repair worked.
Independently check the proposed patched semantic value against the historical dialogue,
object identity, time, negation, task requirements and attribution.
For EACH original dimension, mark resolved=true ONLY if the proposed actual semantic change
removes that specific uncertainty, is supported by context and preserves other semantics.
A change can address more than one dimension; each must be justified separately.
No changes -> ALL original dimensions must be unresolved.
A quotation that exists in the dialogue is NOT automatically semantic support.
Invalid quotation -> changes relying on it cannot be considered supported.
Determine if each proposed change is correct and supported. Distinguish
"correct semantic change", "source text supports value", "new erroneous content".
Do not trust claimed_resolved_dimensions or repair_status. They are hypotheses.
If evidence is ambiguous, insufficient, or the before-state lacks usable detail, prefer
not verified to guessing. Do not hallucinate missing target values.

Return EXACT JSON:
{
 "dimension_judgments": {
  "inconsistency": {"resolved":false,"reason":""},
  "imprecision": {"resolved":false,"reason":""},
  "incompleteness": {"resolved":false,"reason":""}
 },
 "change_judgments":[
   {"change_index":0,"semantically_correct":false,"supported_by_history":false,
    "introduces_new_error":false,"reason":""}
 ],
 "new_error_in_case":false,
 "reason":"short explanation grounded in quoted source facts"
}
Only active original dimensions matter. Do not output markdown.'''


def judge_prompt(case,output,checks):
    repair=output.get('repair') or {}
    payload={'case':{'attribute':case['attribute'],'labels':case['labels'],
                     'detected_attribute':case['detected_attribute'],
                     'task_requirements':case['task_requirements'],
                     'source_text':case['source_text'],'response_text':case['response_text'],
                     'history':case['history']},
             'proposed_repair':{'before_value':repair.get('before_value'),
                     'after_value':repair.get('after_value'), 'changes':repair.get('changes',[])},
             'deterministic_quote_checks':checks}
    return json.dumps(payload,ensure_ascii=False)


def validate_judgment(raw,case,output,checks):
    if not isinstance(raw,dict):raise ValueError('judge输出不是JSON对象')
    dc=raw.get('dimension_judgments')
    cj=raw.get('change_judgments')
    if not isinstance(dc,dict) or not isinstance(cj,list):raise ValueError('缺少维度/变更判定')
    labels=set(case['labels'])
    changes=(output.get('repair') or {}).get('changes') or []
    result_dims={}
    for k in DIMS:
        item=dc.get(k,{})
        if not isinstance(item,dict) or type(item.get('resolved')) is not bool:raise ValueError(f'维度 {k} 必须有resolved布尔值')
        resolved=item['resolved'] if k in labels else False
        result_dims[k]={'resolved':resolved,'reason':str(item.get('reason',''))}
    if len(cj)!=len(changes):raise ValueError('change_judgments数量与changes不一致')
    evaluations={}
    for x in cj:
        if type(x.get('change_index')) is not int:raise ValueError('change_index类型不合法')
        j=x['change_index']
        if j in evaluations or j<0 or j>=len(changes):raise ValueError('change_index无效或重复')
        for k in ('semantically_correct','supported_by_history','introduces_new_error'):
            if type(x.get(k)) is not bool:raise ValueError('change_judgment布尔字段缺失 '+k)
        # invalid quote overrides model statement; also cannot be 'correct supported repair'
        supported=x['supported_by_history'] and checks[j]['quote_valid']
        correct=x['semantically_correct'] and supported and not x['introduces_new_error']
        evaluations[j]={'correct':correct,'supported':supported,
                        'new_error':x['introduces_new_error'],
                        'reason':str(x.get('reason',''))}
    if set(evaluations)!=set(range(len(changes))):raise ValueError('change_indexes未覆盖全部changes')
    good_dims=set()
    for j,c in enumerate(changes):
        if evaluations[j]['correct']:
            good_dims.update(k for k in c.get('addresses_dimensions',[]) if k in labels)
    for k in labels:
        if result_dims[k]['resolved'] and k not in good_dims:
            # do not count dimensions resolved without correct, justified change
            result_dims[k]['resolved']=False
            result_dims[k]['reason']+=' [程序复核：缺乏对应正确且有证据的变更]'
    error=bool(raw.get('new_error_in_case',False)) or any(x['new_error'] for x in evaluations.values())
    return {'dimension_judgments':result_dims,'change_judgments':[{'change_index':i,**evaluations[i]} for i in range(len(changes))],
            'new_error_in_case':error,'reason':str(raw.get('reason',''))}


def judgment_fp(case,output,checks):
    sys=SYSTEM_PROMPT
    user=judge_prompt(case,output,checks)
    return hashlib.sha256((VERSION+'\n'+MODEL+'\n'+sys+'\n'+user).encode()).hexdigest()


def evaluate(cases,outputs):
    OUT.mkdir(parents=True,exist_ok=True)
    judge_file=OUT/'blind_judgments.jsonl'
    cached={}
    if judge_file.exists():
        for r in read_jsonl(judge_file):
            cached[(r.get('case_id'),r.get('method'))]=r
    judgment_rows=[]
    for idx,(cid,method) in enumerate(((c,m) for c in cases for m in METHODS),1):
        case=cases[cid]; out=outputs[(cid,method)]
        checks=evidence_checks(case,out); fp=judgment_fp(case,out,checks)
        prev=cached.get((cid,method))
        if prev and prev.get('fingerprint')==fp and prev.get('error') is None:
            verdict=prev['verdict'];error=None;src='cached'
        else:
            src='run';verdict=None;error=None
            # No edits: deterministic zero-dimension-resolution; no LLM needed.
            if out.get('repair') is not None and not (out['repair'].get('changes') or []):
                verdict={'dimension_judgments':{k:{'resolved':False,'reason':'无语义修改'} for k in DIMS},
                         'change_judgments':[],'new_error_in_case':False,'reason':'无修复候选'}
                src='deterministic'
            elif out.get('error') is not None or out.get('repair') is None:
                error='修复模型输出失败，无法评价'
            else:
                try:
                    raw=llm_call(SYSTEM_PROMPT,judge_prompt(case,out,checks))
                    verdict=validate_judgment(raw,case,out,checks)
                except (RuntimeError,ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:error=str(exc)
        r={'case_id':cid,'dataset':case['dataset'],'method':method,'labels':case['labels'],
           'verdict':verdict,'error':error,'fingerprint':fp,'judge_model':MODEL}
        judgment_rows.append(r)
        write_jsonl(judge_file,judgment_rows+[
            c for k,c in cached.items() if k not in {(x['case_id'],x['method']) for x in judgment_rows}])
        print(f'{idx}/{len(cases)*len(METHODS)} {method} {cid}: {src}, '+('ERROR '+error[:90] if error else 'ok'))
    return judgment_rows


def summarize(cases,outputs,judgments):
    jidx={(x['case_id'],x['method']):x for x in judgments}
    lines=[];report={}
    for method in METHODS:
        entries=[];dim_den=Counter();dim_num=Counter();counts=Counter()
        for cid,case in cases.items():
            out=outputs[(cid,method)]
            jr=jidx.get((cid,method))
            verdict=(jr or {}).get('verdict')
            valid=verdict is not None and (jr or {}).get('error') is None
            labels=case['labels']
            changes=(out.get('repair') or {}).get('changes') or []
            if out.get('error') is not None or out.get('repair') is None:counts['model_failures']+=1
            elif valid:counts['judged_cases']+=1
            else:counts['judge_failures']+=1
            if not valid:
                resolved=[];correct=0;supported=0;new_err=False
            else:
                resolved=[k for k in labels if verdict['dimension_judgments'][k]['resolved']]
                correct=sum(x['correct'] for x in verdict['change_judgments'])
                supported=sum(x['supported'] for x in verdict['change_judgments'])
                new_err=verdict['new_error_in_case']
            for k in labels:dim_den[k]+=1
            for k in resolved:dim_num[k]+=1
            counts['initial_labels']+=len(labels)
            counts['resolved_labels']+=len(resolved)
            counts['fully_repaired']+=int(valid and len(resolved)==len(labels) and not new_err)
            counts['partial_repaired']+=int(valid and 0<len(resolved)<len(labels) and not new_err)
            counts['proposed_changes']+=len(changes)
            counts['correct_changes']+=correct
            counts['supported_changes']+=supported
            counts['new_error_cases']+=int(new_err)
            counts['remaining_labels']+=len(labels)-len(resolved)
            entries.append(len(resolved)/len(labels))
            lines.append({'dataset':case['dataset'],'case_id':cid,'method':method,
                          'dialogue_id':case['dialogue_id'],'response_turn_id':case['response_turn_id'],
                          'attribute':case['attribute'],'initial_dimensions':'+'.join(labels),
                          'resolved_dimensions':'+'.join(resolved),'initial_label_count':len(labels),
                          'resolved_label_count':len(resolved),'proposed_changes':len(changes),
                          'correct_changes':correct,'supported_changes':supported,
                          'new_error':new_err,'evaluation_valid':valid,
                          'repair_gain':len(resolved)/len(labels),'fully_repaired':valid and len(resolved)==len(labels) and not new_err})
        n=len(cases)
        def ratio(a,b):return round(a/b,4) if b else None
        report[method]={
          'cases':n,'judged_cases':counts['judged_cases'], 'model_failures':counts['model_failures'],
          'judge_failures':counts['judge_failures'], 'FRR':ratio(counts['fully_repaired'],n),
          'DRR':{k:ratio(dim_num[k],dim_den[k]) for k in DIMS},
          'DRR_denominator':dict(dim_den),'DRR_numerator':dict(dim_num),
          'ARG':ratio(sum(entries),n),'SRA':ratio(counts['correct_changes'],counts['proposed_changes']),
          'ESR':ratio(counts['supported_changes'],counts['proposed_changes']),
          'EIR':ratio(counts['new_error_cases'],n),'residual_uncertainty_rate':ratio(counts['remaining_labels'],counts['initial_labels']),
          'partially_repaired_cases':counts['partial_repaired'],
          'proposed_changes':counts['proposed_changes'],
          'correct_changes':counts['correct_changes'],'supported_changes':counts['supported_changes'],
          'new_error_cases':counts['new_error_cases'],
          'note':'All available cases included. Missing judgment counted as unresolved, not success. SRA/ESR null if no changes.'}
    csvpath=OUT/'per_case_evaluation.csv'
    with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(lines[0].keys()) if lines else ['dataset','case_id','method'])
        writer.writeheader();writer.writerows(lines)
    write_json(OUT/'evaluation_summary.json',{'judge_model':MODEL,'version':VERSION,'case_count':len(cases),
       'note':'LLM-assisted provisional repair evaluation, not independent gold. Deterministic quote validity gates judge outputs. Repeated observations are retained; for publication, analyze clustered confidence intervals by dialogue/issue and audit judge performance.',
       'methods':report})
    print('\n===== Evaluation Summary =====')
    for m,v in report.items():
        print(f'{m}: FRR={v["FRR"]} ARG={v["ARG"]} DRR={v["DRR"]} SRA={v["SRA"]} ESR={v["ESR"]} EIR={v["EIR"]}, judged={v["judged_cases"]}/{v["cases"]}')
    print('结果目录:',OUT)


def main():
    _release.preflight('judge')
    cases,outputs=load_all()
    ds_count=Counter(c['dataset'] for c in cases.values())
    print('Part5属性级案例:',len(cases),'按数据集:',dict(ds_count),'各方法结果均已匹配')
    if RUN_MODE=='inspect':
        for m in METHODS:
            n=sum(len((outputs[(cid,m)].get('repair') or {}).get('changes') or []) for cid in cases)
            errors=sum(outputs[(cid,m)].get('error') is not None for cid in cases)
            print(f'{m}: {len(cases)}条；提出修改={n}项；修复调用错误={errors}')
        return
    if RUN_MODE=='real':
        judgments=evaluate(cases,outputs)
    elif RUN_MODE=='report':
        judgments=read_jsonl(OUT/'blind_judgments.jsonl')
        if len({(x['case_id'],x['method']) for x in judgments})!=len(cases)*len(METHODS):
            raise ValueError('评价结果不完整或重复，请先运行real')
    else:raise ValueError('RUN_MODE需为inspect、real或report')
    summarize(cases,outputs,judgments)

if __name__=='__main__':main()
