# -*- coding: utf-8 -*-
"""Publication version: shared configuration, no embedded credentials."""
from __future__ import annotations
import hmi_config as _release
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = _release.ROOT
DATA = _release.ROOT / "data"
OUT = _release.STAGE2
SEED = _release.SEED
N_DIALOGUES = int(_release.PAPER["stage2"]["dialogues_per_dataset"])
MAX_TURNS_PER_DIALOGUE = int(_release.PAPER["stage2"]["max_turns_per_dialogue"])
RUN_MODE = _release.MODE
MED_DIR = _release.data_path("stage2","meditod_directory")
KDD_DIR = _release.data_path("stage2","kddres_directory")
KDD_SCRIPT = _release.PAPER["stage2"]["kddres_script"]
MODEL_NAME = _release.model_name()
CONTEXT_TURNS = 6
RETRIES = 3


def save_jsonl(p: Path, records: list[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    temp.replace(p)


def load_json(p: Path):
    if p.suffix == '.jsonl':
        with p.open(encoding='utf-8-sig') as f:
            return [json.loads(x) for x in f if x.strip()]
    return json.loads(p.read_text(encoding='utf-8-sig'))


def find_first(root: Path, names: list[str]) -> Path:
    if not root.exists():
        raise FileNotFoundError(f'目录不存在: {root}\n请将官方仓库下载并解压到此处')
    for name in names:
        files = sorted(root.rglob(name))
        if files:
            return files[0]
    raise FileNotFoundError(f'在 {root} 未找到: {names}')


def preview(obj):
    return json.dumps(obj, ensure_ascii=False, default=str)[:650]


def get_turns(d: dict):
    for key in ('turns', 'utterances', 'dialogue', 'dialog', 'messages', 'conversation', 'log', 'content'):
        v = d.get(key)
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return v
    return None


def convert_turn(t: dict) -> dict | None:
    text = next((t[k] for k in ('utterance', 'text', 'sentence', 'transcript', 'content', 'message', 'sent')
                 if isinstance(t.get(k), str) and t[k].strip()), None)
    role = next((t[k] for k in ('speaker', 'role', 'speaker_id', 'from', 'agent', 'participant')
                 if isinstance(t.get(k), (str, int)) and str(t[k]).strip()), None)
    if text is None or role is None:
        return None
    r = str(role).lower()
    if r in ('doctor', 'physician', 'clinician', 'system', 'agent', 'assistant', '客服', '服務員'):
        normalized = 'professional'
    elif r in ('patient', 'user', 'human', 'customer', '客戶', '用户', '用戶'):
        normalized = 'client'
    else:
        normalized = str(role)
    return {'speaker': str(role), 'speaker_role': normalized, 'text': text.strip()}


def extract_collection(raw, dataset: str, filename: str) -> list[dict]:
    if isinstance(raw, dict):
        for name in ('dialogues', 'dialogs', 'data', 'sessions', 'conversations'):
            if isinstance(raw.get(name), (dict, list)):
                raw = raw[name]
                break
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [(str(i), x) for i, x in enumerate(raw)]
    else:
        raise ValueError(f'{dataset} 数据顶层结构不支持: {preview(raw)}')
    results = []
    rejects = Counter()
    for idx, doc in items:
        if isinstance(doc, list):
            doc = {'turns': doc}
        if not isinstance(doc, dict):
            rejects['not_dict'] += 1
            continue
        turns_raw = get_turns(doc)
        if turns_raw is None:
            rejects['missing_turn_list'] += 1
            continue
        turns = [convert_turn(t) for t in turns_raw]
        if any(x is None for x in turns):
            rejects['unknown_turn_structure'] += 1
            continue
        if len(turns) < 4 or len(set(t['speaker'] for t in turns)) < 2:
            rejects['not_multiturn_two_speakers'] += 1
            continue
        did = str(doc.get('dialogue_id') or doc.get('dialog_id') or doc.get('id') or idx)
        uid = hashlib.sha256((dataset+'|'+filename+'|'+did).encode()).hexdigest()[:14]
        results.append({'dialogue_id': f'{dataset}_{uid}', 'original_id': did, 'dataset': dataset,
                        'source_file': filename, 'turns': [{**t, 'turn_id': i} for i,t in enumerate(turns)]})
    print(f'[{dataset}] 读取候选={len(items)} 合格对话={len(results)} 跳过={dict(rejects)}')
    if not results:
        raise ValueError(f'{dataset} 无法解析成多轮对话；原始片段示例: {preview(items[0] if items else raw)}')
    return results


def read_meditod() -> list[dict]:
    p = find_first(MED_DIR, ['dialogs.json'])
    return extract_collection(load_json(p), 'meditod', _release.source_label(p))


def read_kddres() -> list[dict]:
    folder = KDD_DIR / KDD_SCRIPT
    if not folder.exists():
        raise FileNotFoundError(f'未找到粤语目录 {folder}。请下载KddRES仓库并保留 data_zh_hant')
    # Collect available splits in a stable order; preparation samples eligible dialogues from this pool.
    result = []
    for name in ('test.json', 'val.json', 'train.json'):
        files = sorted(folder.rglob(name))
        if not files:
            continue
        for p in files:
            result.extend(extract_collection(load_json(p), 'kddres', _release.source_label(p)))
    if not result:
        raise FileNotFoundError(f'未找到 {folder}/test.json、val.json 或 train.json')
    seen = set()
    unique = []
    for d in result:
        key = tuple((t['speaker'],t['text']) for t in d['turns'])
        if key not in seen:
            unique.append(d)
            seen.add(key)
    return unique


def select(dialogues: list[dict], ds: str) -> list[dict]:
    if len(dialogues) < N_DIALOGUES:
        raise ValueError(f'{ds} 只有 {len(dialogues)} 段合格对话，不足 {N_DIALOGUES} 段。不会重复抽样。')
    rng = random.Random(SEED + (0 if ds == 'meditod' else 1))
    chosen = rng.sample(sorted(dialogues, key=lambda x:x['dialogue_id']), N_DIALOGUES)
    chosen.sort(key=lambda x: x['dialogue_id'])
    return chosen


def empty_schema():
    return {'intent_id': None, 'units': [], 'links': []}


def map_prompt(dialogue: dict, ix: int, catalog: dict):
    turns = dialogue['turns']
    target = turns[ix]
    history = [{'speaker':x['speaker'], 'text':x['text']} for x in turns[max(0,ix-CONTEXT_TURNS):ix]]
    system = (
        'You perform the first-stage shared-semantic mapping (Ours V2 schema) on an authentic dialogue turn. '
        'Return JSON only. Do NOT treat the other participant as necessarily wrong or correct. '
        'Only map information supported by the current utterance and dialogue history. '
        'Dialogue history helps resolve references, but do not assert unspoken facts. '
        'Keep negation, uncertainty, temporal and quantitative qualifiers if grounded. '
        'Do not invent clinical diagnoses or standard concept identifiers. '
        'Use an ID only if present in the provided candidate catalog; otherwise concept_id=null. '
        'Do not copy labels/Gold from the dataset as if they are predictions. '
        'Response must be: {"shared_semantics":{"intent_id":null,"units":[],"links":[]},'
        '"evidence":[],"mapping_status":"mapped|unmapped"}. '
        'Each unit: {"unit_id":"u1","category":"...","concept_id":null,'
        '"surface":"exact substring from current utterance","token_span":null,"qualifiers":{}}. '
        'Links: {"head":"u1","predicate":"...","tail":"u2"}; '
        'evidence: {"unit_id":"u1","text_span":"..."}. '
        'Provide no unrelated explanation.'
    )
    user = json.dumps({'dataset': dialogue['dataset'], 'domain': ('medical' if dialogue['dataset']=='meditod' else 'restaurant'),
                       'speaker':target['speaker'], 'history':history, 'utterance':target['text'],
                       'candidate_catalog':catalog}, ensure_ascii=False)
    return system, user


def call_llm(system: str, user: str):
    return _release.chat_json(system,user)


def validate_mapping(pred: dict, utterance: str):
    if not isinstance(pred,dict): raise ValueError('映射不是JSON对象')
    sem = pred.get('shared_semantics')
    if not isinstance(sem,dict) or not isinstance(sem.get('units'),list) or not isinstance(sem.get('links'),list):
        raise ValueError('shared_semantics/units/links格式不正确')
    if len(sem['units']) > 30: raise ValueError('语义单元过多')
    uids = set()
    for u in sem['units']:
        if not isinstance(u,dict): raise ValueError('unit不是对象')
        if not isinstance(u.get('unit_id'),str) or u['unit_id'] in uids: raise ValueError('unit_id无效或重复')
        uids.add(u['unit_id'])
        if not isinstance(u.get('category'),str) or not isinstance(u.get('qualifiers'),dict):
            raise ValueError('unit category/qualifiers不合法')
        if not isinstance(u.get('surface'),str): raise ValueError('surface缺失')
        if u['surface'] and u['surface'] not in utterance: raise ValueError('surface不是原句片段')
        if u.get('token_span') is not None: raise ValueError('未提供统一tokens，此处token_span必须为null')
    for link in sem['links']:
        if not isinstance(link,dict) or link.get('head') not in uids or link.get('tail') not in uids:
            raise ValueError('link引用不存在的unit')
    evidence = pred.get('evidence',[])
    if not isinstance(evidence,list): raise ValueError('evidence不是列表')
    for e in evidence:
        if not isinstance(e,dict) or e.get('unit_id') not in uids or not isinstance(e.get('text_span'),str) or e['text_span'] not in utterance:
            raise ValueError('evidence未锚定原句')
    status = pred.get('mapping_status')
    if status not in ('mapped','unmapped'): raise ValueError('mapping_status不合法')
    return {'shared_semantics': {'intent_id':sem.get('intent_id'),'units':sem['units'],'links':sem['links']},
            'evidence':evidence,'mapping_status':status}


def get_catalog(ds: str) -> dict:
    """Optional catalog must be built from permitted non-evaluation sources; never the selected Gold."""
    p = DATA / f'{ds}_candidate_catalog.json'
    if p.is_file():
        obj = load_json(p)
        if not isinstance(obj,(dict,list)): raise ValueError(f'候选目录格式错误: {p}')
        return obj
    print(f'[提示] 未提供 {p.name}，只能保留原文概念及属性；不生成虚构的标准ID。')
    return {}


def main():
    _release.preflight()
    from prepare_data import load_phase2_selection
    selected = load_phase2_selection()
    for ds, dialogues in selected.items():
        folder=OUT/ds
        folder.mkdir(parents=True,exist_ok=True)
        save_jsonl(folder/'standard_dialogues.jsonl',dialogues)
        manifest={'dataset':ds,'seed':SEED,'dialogues':len(dialogues), 'turns':sum(len(d['turns']) for d in dialogues),
                  'sources':dict(Counter(d['source_file'] for d in dialogues)),
                  'sample_scope':'available files, full dialogues; not necessarily official held-out test',
                  'run_mode':RUN_MODE, 'max_turns_per_dialogue':MAX_TURNS_PER_DIALOGUE, 'paper_limits_applied':True}
        (folder/'selection_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'[{ds}] selected={manifest["dialogues"]} dialogues, {manifest["turns"]} turns; sources={manifest["sources"]}')
        if RUN_MODE=='inspect':
            print('  Validated the shared dialogue selection; no API call.')
            continue
        if RUN_MODE!='real': raise ValueError('RUN_MODE只能为inspect或real')
        catalog=get_catalog(ds)
        # checkpoint per turn to resume and avoid paying twice
        path=folder/'ours_v2_mappings.jsonl'
        done = {}
        if path.exists():
            with path.open(encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        r=json.loads(line)
                        done[(r['dialogue_id'],r['turn_id'])]=r
        for d in dialogues:
            for ix,t in enumerate(d['turns']):
                key=(d['dialogue_id'],ix)
                sys,usr=map_prompt(d,ix,catalog)
                sig = _release.digest({"model":MODEL_NAME,"system":sys,"user":usr})
                old = done.get(key)
                if (old and old.get("error") is None and old.get("mapping")
                        and old.get("text")==t["text"] and old.get("input_fingerprint")==sig):
                    continue
                try:
                    pred=validate_mapping(call_llm(sys,usr),t['text'])
                    error=None
                except (ValueError,RuntimeError,KeyError,TypeError) as e:
                    pred=None
                    error=str(e)
                row={'dialogue_id':d['dialogue_id'],'turn_id':ix,'speaker':t['speaker'],
                     'speaker_role':t['speaker_role'],'text':t['text'],'mapping':pred,
                     'error':error,'model':MODEL_NAME,
                     'input_fingerprint':sig}
                with path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                done[key]=row
                print(f'[{ds}] {len(done)}/{manifest["turns"]} -> {"ok" if pred else "ERROR: "+str(error)[:80]}')
        print('saved:',path)


def dialogue_excerpt(d):
    return ' | '.join(f'{t["speaker"]}: {t["text"][:55]}' for t in d['turns'][:4])

if __name__ == '__main__':
    main()
