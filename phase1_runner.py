"""Shared, resumable stage-1 runner; defaults to inspection without API calls."""
from __future__ import annotations
import json
from collections import Counter

import hmi_config as config
import part1Mapping as ours
import mapping_baselines as baselines
from prepare_data import validate_phase1


def run_methods(methods):
    cases=config.jsonl_read(config.PREPARED/"fixed_test_cases.jsonl")
    counts=validate_phase1(cases)
    ours.check_cases(cases)
    print("Stage 1 samples:",counts,"model:",config.model_name() or "(not configured)")
    if config.MODE=="inspect":
        print("Inspection complete; no API calls and no predictions generated.")
        return
    if config.MODE!="real":
        raise ValueError("Use inspect or real mode.")
    config.preflight()
    config.record_run_configuration()
    config.STAGE1.mkdir(parents=True,exist_ok=True)
    for method in methods:
        if method not in ("ours","direct","normalize","fewshot"):
            raise ValueError("Unknown generative mapping method.")
        path=config.STAGE1/f"{method}_predictions.jsonl"
        existing=config.jsonl_read(path) if path.exists() else []
        cached={}
        for row in existing:
            prior=cached.get(row["id"])
            if row.get("method") != method:
                raise ValueError("Checkpoint method does not match the requested method.")
            if prior and (prior.get("case_fingerprint"),prior.get("model")) != (row.get("case_fingerprint"),row.get("model")):
                raise ValueError("Checkpoint contains incompatible records for the same ID.")
            # A retried failure can append another record before compaction.
            cached[row["id"]]=row
        # Preserve all completed cases if the process is interrupted.
        if path.exists():
            stale=set(cached)-{c["id"] for c in cases}
            if stale:
                raise ValueError("The checkpoint belongs to a different selection. Use a new run ID.")
        completed={}
        for i,case in enumerate(cases,1):
            fp=ours.fingerprint(case)
            old=cached.get(case["id"])
            if (old and old.get("case_fingerprint")==fp and old.get("model")==config.model_name()
                    and old.get("error") is None and isinstance(old.get("prediction"),dict)):
                completed[case["id"]]=old
                continue
            try:
                prediction=ours.predict_one(case) if method=="ours" else baselines.predict_one(case,method)
                error=None
            except (ValueError,RuntimeError,KeyError,TypeError,json.JSONDecodeError) as exc:
                prediction=None;error=config.redact_credentials(str(exc))
            row={"id":case["id"],"dataset":case["dataset"],"method":method,
                 "case_fingerprint":fp,"prediction":prediction,"error":error,
                 "model":config.model_name()}
            completed[case["id"]]=row
            with path.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(row,ensure_ascii=False)+"\n")
            print(f"{method}: {i}/{len(cases)} -> {'ok' if error is None else 'ERROR'}")
        # Collapse retried rows into one row per input sample in original order.
        config.jsonl_write(path,[completed[c["id"]] for c in cases])


if __name__=="__main__":
    run_methods(("ours","direct","normalize","fewshot"))
