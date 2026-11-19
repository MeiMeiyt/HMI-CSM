"""Prepare the manuscript-size, shared evaluation samples without API calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import hmi_config as config
from dataset_helpers import read_dialect, normalize_sct, biospans, dialect_example


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_exact(records, count, seed, dataset):
    if count <= 0:
        raise ValueError(f"{dataset}: sample count must be positive.")
    if len(records) < count:
        raise ValueError(f"{dataset}: need {count} eligible records; found only {len(records)}. No replacement or silent downsampling is allowed.")
    indices = sorted(random.Random(seed).sample(range(len(records)), count))
    return [(index, records[index]) for index in indices]


def validate_phase1(cases):
    counts = Counter(c.get("dataset") for c in cases)
    expected = {"dialect":int(config.PAPER["stage1"]["dialect_samples"]),
                "term":int(config.PAPER["stage1"]["mednorm_samples"])}
    if dict(counts) != expected:
        raise ValueError(f"Stage 1 counts differ from configuration: {dict(counts)} != {expected}")
    ids = [c.get("id") for c in cases]
    if any(not isinstance(x,str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Stage 1 case IDs must be nonempty and unique.")
    for case in cases:
        if not isinstance(case.get("text"),str) or not isinstance(case.get("gold"),dict):
            raise ValueError("Each case must contain original text and reference annotations.")
        if case["dataset"] == "dialect":
            if not isinstance(case.get("tokens"),list) or not isinstance(case.get("catalog"),dict):
                raise ValueError("Invalid dialect case.")
        elif not isinstance(case.get("catalog"),list):
            raise ValueError("Invalid terminology candidate catalog.")
    return dict(counts)


def dialect_cases():
    cfg = config.PAPER["stage1"]
    test_file = config.data_path("stage1","dialect_test")
    dev_file = config.data_path("stage1","dialect_validation")
    test, dev = read_dialect(test_file), read_dialect(dev_file)
    expected = int(cfg["dialect_samples"])
    if len(test) != expected:
        raise ValueError(f"NaLiBaSID: the complete chosen test split must contain {expected} examples; parsed {len(test)}. Check that the correct split/file is supplied.")
    catalog = {"intents":sorted({r["intent"] for r in dev}),
               "slots":sorted({t[2:] for r in dev for t in r["tags"] if t.startswith(("B-","I-"))})}
    missing_intents = {r["intent"] for r in test} - set(catalog["intents"])
    missing_slots = {t[2:] for r in test for t in r["tags"] if t.startswith(("B-","I-"))} - set(catalog["slots"])
    if missing_intents or missing_slots:
        raise ValueError("The reference label inventory does not cover the test split. Supply a suitable non-test reference split; test answers are not used as examples.")
    examples = [dialect_example(r) for r in dev[:2]]
    cases = []
    for i, row in enumerate(test):
        cases.append({
            "id":f"dialect_{i:05d}","dataset":"dialect","speaker":"human","domain":"assistant",
            "text":row["text"],"tokens":row["tokens"],"catalog":catalog,"examples":examples,
            "source_row":i,
            "gold":{"intent_id":"intent:"+row["intent"],
                    "slots":[list(x) for x in biospans(row["tags"])]},
        })
    return cases, {"test_sha256":sha256_file(test_file),"reference_sha256":sha256_file(dev_file),
                   "selection":"complete test split in original order"}


def read_mednorm_all(path):
    """Read valid SNOMED-target rows across all original MedNorm sources.

    Unlike the small experiment helper, this does not restrict to CADEC and
    does not reduce each concept group to a single evaluation sample.
    """
    records, skipped = [], Counter()
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream,delimiter="\t" if Path(path).suffix.lower()==".tsv" else ",")
        fields=set(reader.fieldnames or [])
        text_field=next((x for x in ("phrase","text","term") if x in fields),None)
        code_field=next((x for x in ("single_mapped_sct_ids","concept_id","sct_id") if x in fields),None)
        if not text_field or not code_field:
            raise ValueError("MedNorm requires phrase/text/term and single_mapped_sct_ids/concept_id/sct_id columns.")
        for line,row in enumerate(reader,2):
            text=str(row.get(text_field) or "").strip()
            raw=str(row.get(code_field) or "").strip()
            if raw.startswith("sct:"):
                raw=raw[4:]
            code=normalize_sct(raw)
            if not text or not code:
                skipped["missing_text_or_single_snomed_id"]+=1
                continue
            records.append({"text":text,"code":"sct:"+code,"source_row":line,
                            "original_dataset":str(row.get("original_dataset") or "")})
    return records,dict(skipped)


def normalized_phrase(text):
    return " ".join(text.casefold().split())


def read_reference_catalog(path):
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw,dict):
        raw=[{"id":key,"aliases":value} for key,value in raw.items()]
    if not isinstance(raw,list):
        raise ValueError("A term reference catalog must be a list or ID-to-aliases dictionary.")
    result=defaultdict(set)
    for item in raw:
        code=str(item.get("id") or item.get("concept_id") or "").strip()
        if code and not code.startswith("sct:"):
            number=normalize_sct(code)
            code="sct:"+number if number else ""
        aliases=item.get("aliases",[])
        if isinstance(aliases,str):
            aliases=[aliases]
        if code and isinstance(aliases,list):
            result[code].update(str(x).strip() for x in aliases if str(x).strip())
    return result


def terminology_cases():
    cfg=config.PAPER["stage1"]
    source=config.data_path("stage1","mednorm_file")
    rows,skipped=read_mednorm_all(source)
    chosen=sample_exact(rows,int(cfg["mednorm_samples"]),config.SEED+10,"MedNorm")
    chosen_indices={i for i,_ in chosen}
    test_phrases={normalized_phrase(r["text"]) for _,r in chosen}
    references=defaultdict(set)
    external=cfg.get("term_reference_catalog")
    reference_sha=None
    if external:
        path=config.data_path("stage1","term_reference_catalog")
        references=read_reference_catalog(path)
        reference_sha=sha256_file(path)
        reference_mode="supplied non-evaluation reference catalog"
    else:
        reference_mode="unselected MedNorm rows, excluding every evaluated surface form"
        for i,row in enumerate(rows):
            if i not in chosen_indices and normalized_phrase(row["text"]) not in test_phrases:
                references[row["code"]].add(row["text"])
    # Do not allow a reference alias to reproduce any evaluated surface form.
    references={code:sorted(a for a in aliases if normalized_phrase(a) not in test_phrases)
                for code,aliases in references.items()}
    references={code:aliases for code,aliases in references.items() if aliases}
    if not references:
        raise ValueError("No independent reference aliases remain. Supply a non-evaluation term_reference_catalog; test gold labels will not be injected into the candidate pool.")
    reference_examples=[(name,code) for code,names in sorted(references.items()) for name in names][:2]
    examples=[{"input":name,"output":{"intent_id":None,"units":[
        {"unit_id":"u1","category":"term","concept_id":code,"surface":name,
         "token_span":None,"qualifiers":{}}],"links":[]}} for name,code in reference_examples]
    limit=int(cfg["candidate_limit"])
    if limit <= 0:
        raise ValueError("candidate_limit must be positive.")
    cache={}
    cases=[]
    for _,row in chosen:
        phrase=normalized_phrase(row["text"])
        if phrase not in cache:
            scored=[]
            for code,aliases in references.items():
                similarity=max(SequenceMatcher(None,phrase,normalized_phrase(a)).ratio() for a in aliases)
                scored.append((similarity,code))
            scored.sort(key=lambda x:(-x[0],x[1]))
            cache[phrase]=[{"id":code,"aliases":references[code][:4]} for _,code in scored[:limit]]
        cases.append({
            "id":f"term_{row['source_row']:07d}","dataset":"term","speaker":"human","domain":"medical",
            "text":row["text"],"tokens":[],"catalog":cache[phrase],"examples":examples,
            "source_row":row["source_row"],"original_dataset":row["original_dataset"],
            "gold":{"concept_id":row["code"]},
        })
    info={"source_sha256":sha256_file(source),"eligible_rows":len(rows),"skipped":skipped,
          "seed":config.SEED+10,"source_row_numbers":[r["source_row"] for _,r in chosen],
          "sources":dict(Counter(r["original_dataset"] for _,r in chosen)),
          "candidate_reference_mode":reference_mode,"reference_sha256":reference_sha,
          "reference_concepts":len(references),"candidate_limit":limit,
          "candidate_gold_recall":sum(c["gold"]["concept_id"] in {x["id"] for x in c["catalog"]} for c in cases)/len(cases)}
    return cases,info


def write_if_consistent(path,rows,overwrite=False):
    if path.exists():
        old=config.jsonl_read(path)
        if config.digest(old)!=config.digest(rows) and not overwrite:
            raise ValueError(f"A different prepared selection exists at {path}. Use --overwrite only to replace generated selections, then choose new model run IDs.")
    config.jsonl_write(path,rows)


def prepare_phase1(overwrite=False):
    old=config.PAPER["stage1"].get("existing_manifest")
    if old:
        source=config.data_path("stage1","existing_manifest")
        cases=config.jsonl_read(source)
        info={"existing_manifest_sha256":sha256_file(source),"selection":"retained existing evaluation cases"}
    else:
        dialect,dialect_info=dialect_cases()
        terms,term_info=terminology_cases()
        cases=dialect+terms
        info={"dialect":dialect_info,"term":term_info}
    counts=validate_phase1(cases)
    destination=config.PREPARED/"fixed_test_cases.jsonl"
    write_if_consistent(destination,cases,overwrite)
    info.update({"counts":counts,"case_ids":[c["id"] for c in cases],
                 "case_manifest_fingerprint":config.digest(cases)})
    (config.PREPARED/"stage1_selection.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Prepared stage 1:",counts)
    return cases


def prepare_phase2(overwrite=False):
    import part4CognitiveMapping as mapping
    pools={"meditod":mapping.read_meditod(),"kddres":mapping.read_kddres()}
    count=int(config.PAPER["stage2"]["dialogues_per_dataset"])
    limit=int(config.PAPER["stage2"]["max_turns_per_dialogue"])
    if limit < 4:
        raise ValueError("The per-dialogue limit must retain at least four turns.")
    selected={}
    for offset,(dataset,pool) in enumerate(pools.items()):
        candidates=[]; seen=set()
        for dialogue in sorted(pool,key=lambda x:x["dialogue_id"]):
            turns=dialogue["turns"][:limit]
            if len(turns)<4 or len({t["speaker"] for t in turns})<2:
                continue
            content=tuple((t["speaker"],t["text"]) for t in turns)
            if content in seen:
                continue
            seen.add(content)
            candidates.append({**dialogue,"turns":turns})
        chosen=[d for _,d in sample_exact(candidates,count,config.SEED+offset,dataset)]
        selected[dataset]=chosen
        destination=config.PREPARED/"stage2"/dataset/"standard_dialogues.jsonl"
        write_if_consistent(destination,chosen,overwrite)
        manifest={"dataset":dataset,"seed":config.SEED+offset,"eligible_dialogues":len(candidates),
                  "dialogues":len(chosen),"turns":sum(len(d["turns"]) for d in chosen),
                  "max_turns_per_dialogue":limit,
                  "selected_dialogue_ids":[d["dialogue_id"] for d in chosen],
                  "original_ids":[d["original_id"] for d in chosen],
                  "source_files":sorted({d["source_file"] for d in chosen}),
                  "fingerprint":config.digest(chosen),
                  "eligibility":"at least four retained turns and two speakers; duplicate dialogue texts excluded",
                  "sample_scope":"random eligible dialogues from available source files; not necessarily an official held-out split"}
        destination.with_name("selection_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        print("Prepared stage 2:",dataset,manifest["dialogues"],"dialogues;",manifest["turns"],"turns")
    return selected


def load_phase2_selection():
    result={}
    expected=int(config.PAPER["stage2"]["dialogues_per_dataset"])
    limit=int(config.PAPER["stage2"]["max_turns_per_dialogue"])
    for dataset in ("meditod","kddres"):
        folder=config.PREPARED/"stage2"/dataset
        rows=config.jsonl_read(folder/"standard_dialogues.jsonl")
        manifest=json.loads((folder/"selection_manifest.json").read_text())
        if len(rows)!=expected or any(not 4<=len(d["turns"])<=limit for d in rows):
            raise ValueError("Prepared dialogue counts/turn limits differ from the paper configuration.")
        if manifest["fingerprint"]!=config.digest(rows):
            raise ValueError("Prepared dialogue fingerprint mismatch.")
        result[dataset]=rows
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("1","2","all"),default="all")
    parser.add_argument("--overwrite",action="store_true",help="Replace generated selections only; original data files are never changed.")
    args=parser.parse_args()
    if args.stage in ("1","all"):prepare_phase1(args.overwrite)
    if args.stage in ("2","all"):prepare_phase2(args.overwrite)


if __name__=="__main__":
    main()
