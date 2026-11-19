# -*- coding: utf-8 -*-
"""Publication version: shared configuration, no embedded credentials."""
from __future__ import annotations
import hmi_config as _release

import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ==================== 只修改这里 ====================
RUN_MODE = _release.MODE
MODEL_NAME = _release.model_name()
MAX_FORMAT_REPAIRS = 0              # 0=每条只调用一次语义映射（推荐公平对比）
                                    # 1=非法JSON/格式错误时最多额外调用一次LLM
USE_JSON_MODE = False               # 服务商支持 response_format=json_object 时才改True
TIMEOUT_SECONDS = 90
REQUEST_RETRIES = 3                 # 仅网络/限流等技术异常重试
BACKUP_OLD_RESULT = True            # 覆盖旧Ours前自动备份
# ====================================================

ROOT = _release.ROOT
RESULTS = _release.STAGE1
INPUT_FILE = _release.PREPARED / "fixed_test_cases.jsonl"
OUTPUT_FILE = RESULTS / "ours_predictions.jsonl"
DIAGNOSTICS_FILE = RESULTS / "ours_mapping_diagnostics.jsonl"
SCHEMA_VERSION = "1.2"


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到固定测试集：{path}")
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as ex:
                raise ValueError(f"{path.name} 第{line_number}行JSON错误：{ex}") from ex
            if not isinstance(item, dict):
                raise ValueError(f"{path.name} 第{line_number}行必须为JSON对象")
            rows.append(item)
    return rows


def fingerprint(case: dict) -> str:
    """必须与旧版Part1/Part3计算方式一致，不能修改。"""
    b = json.dumps(case, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def check_cases(cases: list[dict]) -> None:
    if not cases:
        raise ValueError("fixed_test_cases.jsonl是空文件")
    seen = set()
    for case in cases:
        sid = case.get("id")
        ds = case.get("dataset")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError(f"样本ID为空或重复：{sid}")
        seen.add(sid)
        if ds not in ("dialect", "term"):
            raise ValueError(f"{sid}的dataset错误：{ds}")
        if not isinstance(case.get("text"), str):
            raise ValueError(f"{sid}缺少text")
        if not isinstance(case.get("gold"), dict):
            raise ValueError(f"{sid}缺少gold（仅用于保持清单完整，不提供给LLM）")
        if ds == "dialect":
            if not isinstance(case.get("catalog"), dict) or not isinstance(case.get("tokens"), list):
                raise ValueError(f"{sid}缺少方言catalog或tokens")
        else:
            if not isinstance(case.get("catalog"), list):
                raise ValueError(f"{sid}缺少术语catalog")
    print("固定测试集：", len(cases), "条；分布：", dict(Counter(c["dataset"] for c in cases)))
    print("注意：原文件保留，预测时不传递任何gold字段。")


def make_empty_prediction(case: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "speaker": case.get("speaker", "human"),
            "text": case["text"],
            "domain": case.get("domain", "general"),
        },
        "shared_semantics": {"intent_id": None, "units": [], "links": []},
        "evidence": [],
        "mapping_status": "unmapped",
    }


def lexical_hints(case: dict) -> list[dict]:
    """概念锚定提示：仅由原文和候选别名计算，绝不使用Gold。
    只对术语任务做字面候选提示；并非最终语义判断。
    """
    if case["dataset"] != "term":
        return []
    from difflib import SequenceMatcher
    mention = case["text"].casefold().strip()
    hints = []
    for cand in case["catalog"]:
        aliases = [str(a) for a in cand.get("aliases", []) if a]
        if not aliases:
            continue
        score = max(SequenceMatcher(None, mention, a.casefold()).ratio() for a in aliases)
        hints.append((score, cand.get("id")))
    hints.sort(key=lambda x: (-x[0], str(x[1])))
    return [{"concept_id": cid, "lexical_similarity": round(score, 3)}
            for score, cid in hints[:3] if isinstance(cid, str)]


def build_prompt(case: dict) -> tuple[str, str]:
    """两个任务采用同一种DASSR JSON结构；提示与候选全部来自固定测试清单。
    此函数完全不访问 case['gold']。
    """
    template = make_empty_prediction(case)
    base = (
        "You are a shared-semantic mapping system, not a free-form chatbot.\n"
        "Human and machine are source labels only; neither is the reference truth.\n"
        "Return exactly ONE JSON object, no Markdown or explanations.\n"
        "Use the following object structure:\n"
        + json.dumps(template, ensure_ascii=False)
        + "\nEvery item in shared_semantics.units has: "
          "unit_id, category, concept_id, surface, token_span, qualifiers.\n"
        "Every evidence entry has: unit_id, text_span.\n"
        "Do not infer unmentioned facts. Use only canonical labels listed below.\n"
    )

    if case["dataset"] == "dialect":
        token_lines = "\n".join(f"{i}: {word}" for i, word in enumerate(case["tokens"]))
        base += (
            "\nTASK: Bavarian dialect intent detection and slot annotation.\n"
            "Allowed intent labels (without 'intent:' prefix): "
            + json.dumps(case["catalog"]["intents"], ensure_ascii=False)
            + "\nAllowed slot labels (without 'slot:' prefix): "
            + json.dumps(case["catalog"]["slots"], ensure_ascii=False)
            + "\nRules:\n"
            "1. Determine intent from the full ORIGINAL utterance. Format: intent:<label>.\n"
            "2. Extract only spans expected for this dataset's intent/slot annotation task, "
              "not every meaningful word. Use category='slot' and concept_id='slot:<label>'.\n"
            "3. Use original token indices, zero-based [start,end), end excluded.\n"
            "4. The entire annotated phrase can cover multiple tokens. "
              "Do NOT artificially shorten it to a minimum phrase.\n"
            "5. Do NOT merge separate occurrences of the same slot type.\n"
            "6. Verify that token_span and surface refer to the same ORIGINAL token sequence.\n"
            "7. Distinguish similar slot classes from context, e.g. weather/attribute vs condition_description, "
              "reference vs item name. Do not make up annotations.\n"
            "8. Set links=[]; qualifiers={} if unneeded. "
              "Use null intent only if truly indeterminate.\n"
            "Indexed ORIGINAL tokens:\n" + token_lines + "\n"
        )
    else:
        base += (
            "\nTASK: map a patient's informal medical mention to ONE allowed SNOMED concept.\n"
            "Candidate concepts and their known aliases:\n"
            + json.dumps(case["catalog"], ensure_ascii=False)
            + "\nRules:\n"
            "1. Read the entire original mention and compare its meaning with candidate aliases.\n"
            "2. Choose ONE best-supported equivalent candidate ID; exact wording is not required.\n"
            "3. A patient phrase may express a symptom rather than a diagnosis; never add a diagnosis.\n"
            "4. Avoid conflating merely related, broader, or narrower concepts.\n"
            "5. Return empty units only if no candidate concept is adequately supported.\n"
            "6. Use category='term', token_span=null, intent_id=null, links=[].\n"
            "7. Lexical hints are POSSIBLE anchors, not labels or ground truth. "
              "They may be wrong.\n"
            "Lexical anchor suggestions:\n"
            + json.dumps(lexical_hints(case), ensure_ascii=False) + "\n"
        )

    examples = case.get("examples") or []
    if examples:
        # 示例已由原有数据准备程序从非测试数据中生成；不暴露当前样本gold。
        base += "\nTwo annotated in-context examples:\n"
        base += json.dumps(examples[:2], ensure_ascii=False) + "\n"

    user = (
        f"Speaker: {case.get('speaker', 'human')}\n"
        f"Original text: {case['text']}\n"
        "Return only the shared semantic JSON for this input."
    )
    return base, user


def call_llm(system: str, user: str) -> str:
    return _release.chat_text(system, user, json_mode=USE_JSON_MODE)


def parse_model_json(answer: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", answer, flags=re.I | re.S)
    if fenced:
        answer = fenced.group(1)
    try:
        obj = json.loads(answer.strip())
    except json.JSONDecodeError:
        first, last = answer.find("{"), answer.rfind("}")
        if first == -1 or last < first:
            raise ValueError("模型未输出JSON对象")
        try:
            obj = json.loads(answer[first:last + 1])
        except json.JSONDecodeError as ex:
            raise ValueError(f"模型JSON格式错误：{ex}") from ex
    if not isinstance(obj, dict):
        raise ValueError("模型JSON根节点不是对象")
    return obj


def infer_unique_span(tokens: list[str], surface: str) -> list[int] | None:
    """仅当surface与原始tokens的某个连续片段唯一一致时，修正模型的索引错位。
    不改变槽位类别，不使用Gold，也不推测新槽位。
    """
    words = surface.split()
    if not words:
        return None
    norm = lambda s: s.casefold().strip()
    options = []
    for a in range(len(tokens) - len(words) + 1):
        if all(norm(tokens[a + j]) == norm(words[j]) for j in range(len(words))):
            options.append([a, a + len(words)])
    return options[0] if len(options) == 1 else None


def validate_prediction(raw: dict, case: dict) -> tuple[dict, list[str]]:
    """校验并轻度修正格式；绝不查Gold来补答案或改标签。"""
    sem = raw.get("shared_semantics")
    if not isinstance(sem, dict) or not isinstance(sem.get("units"), list):
        raise ValueError("缺少合法shared_semantics.units")

    result = make_empty_prediction(case)
    notes: list[str] = []
    task = case["dataset"]
    intent = sem.get("intent_id")

    if task == "dialect":
        valid_intents = {"intent:" + v for v in case["catalog"]["intents"]}
        valid_concepts = {"slot:" + v for v in case["catalog"]["slots"]}
        if intent not in valid_intents:
            if intent is not None:
                notes.append(f"illegal_intent:{intent}")
            intent = None
    else:
        valid_concepts = {v["id"] for v in case["catalog"]}
        intent = None  # term任务没有Gold意图，不应虚构
    result["shared_semantics"]["intent_id"] = intent

    mapping: dict[str, str] = {}
    unit_seen = set()
    skipped = 0
    for obj in sem["units"]:
        if not isinstance(obj, dict):
            skipped += 1
            notes.append("non_object_unit")
            continue
        cid = obj.get("concept_id")
        if cid not in valid_concepts:
            skipped += 1
            notes.append(f"out_of_catalog:{cid}")
            continue
        category = "slot" if task == "dialect" else "term"
        if obj.get("category") != category:
            skipped += 1
            notes.append(f"wrong_category:{cid}")
            continue

        span = obj.get("token_span")
        surface = str(obj.get("surface") or "")
        if task == "dialect":
            tokens = case["tokens"]
            valid = (
                isinstance(span, list) and len(span) == 2
                and all(type(v) is int for v in span)
                and 0 <= span[0] < span[1] <= len(tokens)
            )
            canonical_text = " ".join(tokens[span[0]:span[1]]) if valid else ""
            if surface and (not valid or surface.casefold() != canonical_text.casefold()):
                repaired = infer_unique_span(tokens, surface)
                if repaired is not None and repaired != span:
                    notes.append(f"span_index_repair:{span}->{repaired}")
                    span, valid = repaired, True
            if not valid:
                skipped += 1
                notes.append(f"invalid_token_span:{span}")
                continue
            surface = " ".join(tokens[span[0]:span[1]])
            key = (cid, span[0], span[1])
        else:
            span = None
            surface = surface or case["text"]
            key = (cid,)

        if key in unit_seen:
            notes.append(f"duplicate_unit:{key}")
            continue
        if task == "term" and result["shared_semantics"]["units"]:
            skipped += 1
            notes.append("extra_term_unit")
            continue

        unit_seen.add(key)
        uid = "u" + str(len(result["shared_semantics"]["units"]) + 1)
        mapping[str(obj.get("unit_id", uid))] = uid
        qualifiers = obj.get("qualifiers")
        result["shared_semantics"]["units"].append({
            "unit_id": uid, "category": category,
            "concept_id": cid, "surface": surface,
            "token_span": span,
            "qualifiers": qualifiers if isinstance(qualifiers, dict) else {},
        })

    # 这两个公开语料无统一语义关系Gold，本实验任务统一不输出links。
    result["shared_semantics"]["links"] = []

    for ev in raw.get("evidence", []):
        if not isinstance(ev, dict):
            continue
        text_span = str(ev.get("text_span") or "")
        uid = mapping.get(str(ev.get("unit_id")))
        if uid and text_span and text_span in case["text"]:
            new_ev = {"unit_id": uid, "text_span": text_span}
            if new_ev not in result["evidence"]:
                result["evidence"].append(new_ev)

    # 没有证据时，仅用原文中确实出现的surface生成可追溯证据，不补新事实。
    for unit in result["shared_semantics"]["units"]:
        if (unit["surface"] in case["text"] and
                not any(x["unit_id"] == unit["unit_id"] for x in result["evidence"])):
            result["evidence"].append({
                "unit_id": unit["unit_id"], "text_span": unit["surface"]
            })

    units = result["shared_semantics"]["units"]
    if task == "dialect":
        if intent is None and not units:
            status = "unmapped"
        elif intent is None or skipped:
            status = "partial"
        else:
            status = "mapped"
    else:
        status = "partial" if skipped and units else ("mapped" if units else "unmapped")
    result["mapping_status"] = status
    return result, notes


def predict_one(case: dict, method: str = "ours") -> dict:
    """仅实现本文方法；旧Baseline预测文件保持不变。"""
    if method != "ours":
        raise ValueError("本文件只负责Ours映射；三种Baseline请使用原来的第二部分脚本")
    system, user = build_prompt(case)
    answer = call_llm(system, user)
    obj, _ = validate_prediction(parse_model_json(answer), case)
    return obj


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dst = path.with_name(f"{path.stem}_backup_{stamp}{path.suffix}")
    shutil.copy2(path, dst)
    return dst


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    from phase1_runner import run_methods
    run_methods(("ours",))


if __name__ == "__main__":
    main()
