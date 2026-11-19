"""Offline tests. Fixtures are synthetic; no provider API is contacted."""
from __future__ import annotations
import contextlib
import copy
import csv
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hmi_config as config
import prepare_data
import part1Mapping
import phase1_runner


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.paper=copy.deepcopy(config.PAPER)
        self.stack=contextlib.ExitStack()
        for key,value in {
            "ROOT":self.root,"PREPARED":self.root/"prepared",
            "RUN_DIR":self.root/"results"/"test",
            "STAGE1":self.root/"results"/"test"/"stage1",
            "STAGE2":self.root/"results"/"test"/"stage2",
            "PAPER":self.paper,"MODE":"inspect",
        }.items():
            self.stack.enter_context(mock.patch.object(config,key,value))
        self.output=self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def make_stage1_data(self):
        data=self.root/"data"
        data.mkdir()
        def conll(n):
            return "".join(f"# intent: test_intent\n# text: fixture {i}\n0\tfixture\t_\tO\n\n" for i in range(n))
        (data/"de-ba.MAS.test").write_text(conll(1212))
        (data/"de-ba.MAS.valid").write_text(conll(2))
        with (data/"mednorm_full.tsv").open("w",newline="") as stream:
            writer=csv.writer(stream,delimiter="\t")
            writer.writerow(["original_dataset","phrase","single_mapped_sct_ids"])
            for i in range(10030):
                # The 10k target deliberately exceeds the CADEC-only subset.
                writer.writerow(["CADEC" if i%2 else "OTHER",f"synthetic mention {i}",str(100000+i%10)])

    def test_paper_size_stage1_and_no_test_alias_leak(self):
        self.make_stage1_data()
        cases=prepare_data.prepare_phase1()
        self.assertEqual(len(cases),11212)
        self.assertEqual(sum(c["dataset"]=="dialect" for c in cases),1212)
        terms=[c for c in cases if c["dataset"]=="term"]
        self.assertEqual(len(terms),10000)
        self.assertTrue(any(c["original_dataset"]=="OTHER" for c in terms))
        test_phrases={prepare_data.normalized_phrase(c["text"]) for c in terms}
        for case in terms:
            for candidate in case["catalog"]:
                self.assertTrue(all(prepare_data.normalized_phrase(a) not in test_phrases for a in candidate["aliases"]))
        selected_before=[c["source_row"] for c in terms]
        repeated=prepare_data.prepare_phase1()
        self.assertEqual(selected_before,[c["source_row"] for c in repeated if c["dataset"]=="term"])
        case=copy.deepcopy(cases[0])
        prompt1=part1Mapping.build_prompt(case)
        case["gold"]={"intent_id":"FORBIDDEN_TEST_ANSWER","slots":[["FORBIDDEN",0,1]]}
        self.assertEqual(prompt1,part1Mapping.build_prompt(case))
        self.assertNotIn("FORBIDDEN_TEST_ANSWER","".join(prompt1))

    def test_no_replacement_or_silent_short_sampling(self):
        with self.assertRaisesRegex(ValueError,"No replacement"):
            prepare_data.sample_exact([1,2],3,1,"test")
        self.make_stage1_data()
        (self.root/"data"/"de-ba.MAS.test").write_text("# intent: test_intent\n0\tfixture\t_\tO\n\n")
        with self.assertRaisesRegex(ValueError,"complete chosen test split"):
            prepare_data.dialect_cases()

    def test_stage2_200_each_30_turns_deterministic_and_tamper_detected(self):
        mapping=importlib.import_module("part4CognitiveMapping")
        med=self.root/"data"/"MediTOD"
        kdd=self.root/"data"/"KddRES"
        med.mkdir(parents=True)
        (kdd/"data_zh_hant").mkdir(parents=True)
        def dialogues(prefix):
            return [{"dialogue_id":f"{prefix}_{i}","turns":[
                {"speaker":"doctor" if j%2==0 else "patient",
                 "text":f"synthetic {prefix} dialogue {i}, turn {j}"}
                for j in range(35)]} for i in range(215)]
        (med/"dialogs.json").write_text(json.dumps(dialogues("med")))
        (kdd/"data_zh_hant"/"test.json").write_text(json.dumps(dialogues("kdd")))
        with mock.patch.multiple(mapping,ROOT=self.root,MED_DIR=med,KDD_DIR=kdd):
            selected=prepare_data.prepare_phase2()
            repeated=prepare_data.prepare_phase2()
        for ds in ("meditod","kddres"):
            self.assertEqual(len(selected[ds]),200)
            self.assertTrue(all(len(d["turns"])==30 for d in selected[ds]))
            self.assertEqual(selected[ds],repeated[ds])
        self.assertEqual(prepare_data.load_phase2_selection(),selected)
        path=config.PREPARED/"stage2"/"meditod"/"standard_dialogues.jsonl"
        changed=config.jsonl_read(path)
        changed[0]["turns"][0]["text"]="modified fixture"
        config.jsonl_write(path,changed)
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            prepare_data.load_phase2_selection()

    def test_credentials_not_in_repr_or_api_errors(self):
        secret="UNIT_TEST_VALUE_NOT_A_CREDENTIAL"
        variables={"HMI_API_KEY":secret,"HMI_MODEL":"unit-test-model",
                   "HMI_BASE_URL":"https://example.invalid/v1","HMI_RETRIES":"1"}
        with mock.patch.dict(os.environ,variables):
            settings=config.ModelSettings.from_env()
            self.assertNotIn(secret,repr(settings))
            error=HTTPError("https://example.invalid",401,"unauthorized",{},io.BytesIO(secret.encode()))
            with mock.patch.object(config,"urlopen",side_effect=error):
                with self.assertRaises(RuntimeError) as caught:
                    config.chat_text("system","user")
            self.assertNotIn(secret,str(caught.exception))
            self.assertIn("HTTP 401",str(caught.exception))

    def test_append_checkpoint_resumes_after_failed_retry_and_interruptions(self):
        self.paper["stage1"]["dialect_samples"]=1
        self.paper["stage1"]["mednorm_samples"]=2
        cases=[
            {"id":"d","dataset":"dialect","speaker":"human","domain":"assistant","text":"fixture",
             "tokens":["fixture"],"catalog":{"intents":["x"],"slots":[]},"examples":[],
             "gold":{"intent_id":"intent:x","slots":[]}},
            *[{"id":f"t{i}","dataset":"term","speaker":"human","domain":"medical","text":f"fixture{i}",
               "tokens":[],"catalog":[],"examples":[],"gold":{"concept_id":"sct:100"}} for i in range(2)]
        ]
        config.jsonl_write(config.PREPARED/"fixed_test_cases.jsonl",cases)
        seen={}
        def predict(case,method="ours"):
            key=case["id"];seen[key]=seen.get(key,0)+1
            if key=="d" and seen[key]==1:raise RuntimeError("synthetic transient failure")
            if key=="t1" and seen[key]<3:raise KeyboardInterrupt()
            return {"mapping_status":"unmapped"}
        env={"HMI_API_KEY":"UNIT_TEST_VALUE_NOT_A_CREDENTIAL",
             "HMI_MODEL":"unit-test-model","HMI_BASE_URL":"https://example.invalid/v1"}
        with mock.patch.dict(os.environ,env),mock.patch.object(config,"MODE","real"),mock.patch.object(part1Mapping,"predict_one",side_effect=predict):
            with self.assertRaises(KeyboardInterrupt):phase1_runner.run_methods(("ours",))
            with self.assertRaises(KeyboardInterrupt):phase1_runner.run_methods(("ours",))
            phase1_runner.run_methods(("ours",))
        rows=config.jsonl_read(config.STAGE1/"ours_predictions.jsonl")
        self.assertEqual([r["id"] for r in rows],["d","t0","t1"])
        self.assertTrue(all(r["error"] is None for r in rows))
        self.assertEqual(seen["t0"],1)

    def test_all_stage_imports_and_default_cli_do_not_call_api(self):
        with mock.patch.object(config,"urlopen",side_effect=AssertionError("Unexpected network request")):
            for name in ("part1Mapping","part2Baseline","part2Traditional","part3compared",
                         "part4CognitiveMapping","part5Cognitive","part6Repair","part7Baseline","part8Evaluate"):
                importlib.import_module(name)
        p=subprocess.run([sys.executable,str(ROOT/"run_experiments.py"),"all"],
                         text=True,capture_output=True,cwd=ROOT,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr)
        plan=json.loads(p.stdout)
        self.assertFalse(plan["api_calls"])
        self.assertEqual(plan["stage1"]["mednorm_samples"],10000)
        self.assertEqual(plan["stage2"]["dialogues_per_dataset"],200)

    def test_all_stage1_methods_feed_the_same_offline_evaluator(self):
        traditional=importlib.import_module("part2Traditional")
        baseline=importlib.import_module("mapping_baselines")
        evaluator=importlib.import_module("part3compared")
        self.paper["stage1"]["dialect_samples"]=1
        self.paper["stage1"]["mednorm_samples"]=2
        data=self.root/"data";data.mkdir()
        dev=data/"de-ba.MAS.valid"
        dev.write_text("# intent: x\n0\tfixture\t_\tO\n\n")
        cases=[
            {"id":"dialect_0","dataset":"dialect","speaker":"human","domain":"assistant","text":"fixture",
             "tokens":["fixture"],"catalog":{"intents":["x"],"slots":[]},"examples":[],
             "gold":{"intent_id":"intent:x","slots":[]}},
            *[{"id":f"term_{i}","dataset":"term","speaker":"human","domain":"medical","text":f"fixture {i}",
               "tokens":[],"catalog":[{"id":"sct:100","aliases":["reference"]}],"examples":[],
               "gold":{"concept_id":"sct:100"}} for i in range(2)]
        ]
        config.jsonl_write(config.PREPARED/"fixed_test_cases.jsonl",cases)
        # A legal abstention is sufficient to verify all six output/evaluation paths.
        response=json.dumps({"shared_semantics":{"intent_id":None,"units":[],"links":[]},"evidence":[]})
        env={"HMI_API_KEY":"UNIT_TEST_VALUE_NOT_A_CREDENTIAL","HMI_MODEL":"unit-test-model",
             "HMI_BASE_URL":"https://example.invalid/v1"}
        with mock.patch.multiple(traditional,OUT=config.STAGE1,DEV_FILE=dev,RUN_MODE="run"), \
             mock.patch.multiple(evaluator,RESULTS_DIR=config.STAGE1,DRAW_FIGURES=False), \
             mock.patch.object(config,"MODE","real"),mock.patch.dict(os.environ,env), \
             mock.patch.object(part1Mapping,"call_llm",return_value=response), \
             mock.patch.object(baseline,"llm",return_value=response):
            config.STAGE1.mkdir(parents=True,exist_ok=True)
            traditional.main()
            phase1_runner.run_methods(("ours","direct","normalize","fewshot"))
            evaluator.main()
        for method in ("lexicon","tfidf","ours","direct","normalize","fewshot"):
            rows=config.jsonl_read(config.STAGE1/f"{method}_predictions.jsonl")
            self.assertEqual([r["id"] for r in rows],[c["id"] for c in cases])
            self.assertTrue(all(r["error"] is None for r in rows))
        with (config.STAGE1/"summary.csv").open(encoding="utf-8-sig",newline="") as stream:
            summary=list(csv.DictReader(stream))
        self.assertEqual(len(summary),12)
        self.assertTrue((config.STAGE1/"pairwise_vs_ours.csv").is_file())


if __name__=="__main__":
    unittest.main()
