"""Explicit stage dispatcher. API execution requires --execute."""
from __future__ import annotations
import argparse
import importlib
import json
import os

STAGES = {
    "traditional":"part2Traditional", "map":"part1Mapping", "baselines1":"part2Baseline",
    "evaluate1":"part3compared", "map-dialogues":"part4CognitiveMapping",
    "detect":"part5Cognitive", "repair":"part6Repair", "baselines2":"part7Baseline",
    "evaluate2":"part8Evaluate",
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("inspect","prepare","prepare1","prepare2","all",*STAGES))
    parser.add_argument("--execute",action="store_true",help="Actually execute the chosen stage; generative stages can incur provider charges.")
    parser.add_argument("--run-id",help="Separate each base model/configuration into its own output directory.")
    parser.add_argument("--overwrite",action="store_true",help="Replace generated sample selections when preparing data.")
    args=parser.parse_args()
    os.environ["HMI_RUN_MODE"]="real" if args.execute else "inspect"
    if args.run_id:
        os.environ["HMI_RUN_ID"]=args.run_id
    import hmi_config as config
    import prepare_data
    if args.stage in ("inspect","all") and (args.stage=="inspect" or not args.execute):
        print(json.dumps({
            "api_calls":False,"stage1":config.PAPER["stage1"],
            "stage2":config.PAPER["stage2"],"seed":config.SEED,
            "stage_order":list(STAGES),
            "prepared_stage1_exists":(config.PREPARED/"fixed_test_cases.jsonl").is_file(),
            "prepared_stage2_exists":all((config.PREPARED/"stage2"/d/"standard_dialogues.jsonl").is_file()
                                         for d in ("meditod","kddres")),
            "note":"Use prepare first. Each generative stage requires --execute and environment-only credentials."
        },ensure_ascii=False,indent=2))
        return
    if args.stage in ("prepare","prepare1","prepare2"):
        if args.stage in ("prepare","prepare1"):prepare_data.prepare_phase1(args.overwrite)
        if args.stage in ("prepare","prepare2"):prepare_data.prepare_phase2(args.overwrite)
        return
    if args.stage=="all":
        config.preflight()
        prepare_data.prepare_phase1(args.overwrite)
        prepare_data.prepare_phase2(args.overwrite)
        stages=list(STAGES)
    else:
        stages=[args.stage]
    for stage in stages:
        module=importlib.import_module(STAGES[stage])
        print("Running stage:",stage,"mode:",config.MODE)
        module.main()


if __name__=="__main__":
    main()
