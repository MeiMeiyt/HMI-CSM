"""Publication version: shared configuration, no embedded credentials."""
from __future__ import annotations
import hmi_config as _release

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

# ================= PyCharm 顶部设置：直接修改这里 =================
ROOT = _release.ROOT
RESULTS_DIR = _release.STAGE1

METHODS = ("lexicon","tfidf","direct","normalize","fewshot","ours")

# True：所有六种方法必须有结果文件，严格保证同一批样本对比。
# False：仅比较现有文件（试运行用，不建议用于论文正式结果）。
REQUIRE_ALL_METHODS = True
PRIMARY_METHOD = "ours"
# True：自动生成柱状图（需要 matplotlib；没有安装仍能正常生成CSV）。
DRAW_FIGURES = False
# =================================================================


def load_jsonl(path: Path) -> list[dict]:
    """读取JSONL并报告准确的文件行号，便于排查损坏文件。"""
    result = []
    with path.open('r', encoding='utf-8-sig') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as ex:
                raise ValueError(f'{path.name} 第{lineno}行不是有效JSON: {ex}') from ex
            if not isinstance(item, dict):
                raise ValueError(f'{path.name} 第{lineno}行必须是JSON对象')
            result.append(item)
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(case: dict) -> str:
    """完全复用前两部分的样本指纹算法，防止跨轮结果混用。"""
    encoded = json.dumps(case, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def read_all_data() -> tuple[list[dict], dict[str, dict[str, dict]]]:
    manifest = _release.PREPARED / "fixed_test_cases.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f'未找到 {manifest}。请先运行第一部分生成固定测试样本。')
    cases = load_jsonl(manifest)
    if not cases:
        raise ValueError('固定测试样本为空')

    case_ids = [str(c.get('id', '')) for c in cases]
    if any(not x for x in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError('fixed_test_cases.jsonl 的样本ID为空或重复')
    for case in cases:
        if case.get('dataset') not in ('dialect', 'term'):
            raise ValueError(f'样本 {case["id"]} 的dataset必须为dialect或term')
        if not isinstance(case.get('gold'), dict):
            raise ValueError(f'样本 {case["id"]} 缺少Gold标注')
        if case['dataset'] == 'dialect':
            if 'intent_id' not in case['gold'] or 'slots' not in case['gold']:
                raise ValueError(f'方言样本 {case["id"]} 缺少Gold意图或槽位')
        else:
            if 'concept_id' not in case['gold']:
                raise ValueError(f'术语样本 {case["id"]} 缺少Gold概念')

    all_predictions: dict[str, dict[str, dict]] = {}
    for method in METHODS:
        path = RESULTS_DIR / f'{method}_predictions.jsonl'
        if not path.is_file():
            if REQUIRE_ALL_METHODS:
                raise FileNotFoundError(f'缺少 {path.name}；请先运行对应方法。')
            print(f'提示：{path.name} 不存在，跳过该方法。')
            continue
        rows = load_jsonl(path)
        pred_ids = [str(row.get('id', '')) for row in rows]
        if len(pred_ids) != len(set(pred_ids)):
            raise ValueError(f'{method} 的预测结果存在重复ID')
        if set(pred_ids) != set(case_ids):
            missing = sorted(set(case_ids) - set(pred_ids))[:5]
            extra = sorted(set(pred_ids) - set(case_ids))[:5]
            raise ValueError(f'{method} 的结果与固定测试集不一致；缺失ID:{missing}，多余ID:{extra}')
        by_id = {str(row['id']): row for row in rows}
        for case in cases:
            rec = by_id[case['id']]
            if rec.get('case_fingerprint') != fingerprint(case):
                raise ValueError(
                    f'{method} 样本 {case["id"]} 指纹不匹配：可能重新生成过测试集，'
                    '请在同一份fixed_test_cases.jsonl上重新运行六种方法。'
                )
            if rec.get('method') not in (None, method):
                raise ValueError(f'{method} 的样本 {case["id"]} 包含错误的方法标签')
            if rec.get('dataset') != case['dataset']:
                raise ValueError(f'{method} 的样本 {case["id"]} 包含错误的数据集标签')
        all_predictions[method] = by_id

    if not all_predictions:
        raise ValueError('没有找到任何预测结果文件')
    return cases, all_predictions


def gold_slots(case: dict) -> set[tuple[str, int, int]]:
    """Gold格式为 [[slot:类型, 起点, 终点], ...]，终点不包含。"""
    result = set()
    for s in case['gold']['slots']:
        if not isinstance(s, (list, tuple)) or len(s) != 3:
            raise ValueError(f'样本 {case["id"]} 的Gold槽位格式错误: {s}')
        result.add((str(s[0]), int(s[1]), int(s[2])))
    return result


def decode_record(case: dict, record: dict) -> dict:
    """将模型预测解析为可比较字段；缺失、非法和弃权均保留。"""
    raw = record.get('prediction')
    api_error = record.get('error')
    decoded = {
        'status': 'ok', 'error': '', 'valid_prediction': False,
        'intent_id': None, 'slots': set(), 'concept_id': None,
        'unmapped': False, 'technical_error': False,
    }

    if api_error or raw is None:
        decoded['status'] = 'technical_error'
        decoded['error'] = str(api_error or 'prediction为null')
        decoded['technical_error'] = True
        return decoded

    if not isinstance(raw, dict) or not isinstance(raw.get('shared_semantics'), dict):
        decoded['status'] = 'invalid_structure'
        decoded['error'] = '缺少合法的shared_semantics对象'
        return decoded

    sem = raw['shared_semantics']
    units = sem.get('units')
    if not isinstance(units, list):
        decoded['status'] = 'invalid_structure'
        decoded['error'] = 'units不是列表'
        return decoded

    try:
        if case['dataset'] == 'dialect':
            intent = sem.get('intent_id')
            if intent is not None and not isinstance(intent, str):
                raise ValueError('intent_id既不是字符串也不是null')
            allowed_intents = {'intent:' + x for x in case.get('catalog', {}).get('intents', [])}
            allowed_slots = {'slot:' + x for x in case.get('catalog', {}).get('slots', [])}
            if intent is not None and intent not in allowed_intents:
                raise ValueError('intent_id不在当前任务允许的标签集合中')
            slots = set()
            for u in units:
                if not isinstance(u, dict) or u.get('category') != 'slot':
                    raise ValueError('方言units的category不为slot')
                span = u.get('token_span')
                if not (isinstance(span, list) and len(span) == 2
                        and all(type(v) is int for v in span)
                        and 0 <= span[0] < span[1] <= len(case.get('tokens', []))):
                    raise ValueError('非法slot token_span')
                if not isinstance(u.get('concept_id'), str):
                    raise ValueError('非法slot concept_id')
                if u['concept_id'] not in allowed_slots:
                    raise ValueError('slot concept_id不在当前任务允许的标签集合中')
                slots.add((u['concept_id'], span[0], span[1]))
            decoded['intent_id'] = intent
            decoded['slots'] = slots
        else:
            if len(units) > 1:
                raise ValueError('术语结果包含多于1个概念')
            if units:
                u = units[0]
                if not isinstance(u, dict) or u.get('category') != 'term':
                    raise ValueError('术语unit的category不为term')
                if not isinstance(u.get('concept_id'), str):
                    raise ValueError('非法术语concept_id')
                allowed_concepts = {c['id'] for c in case.get('catalog', []) if isinstance(c, dict) and 'id' in c}
                if u['concept_id'] not in allowed_concepts:
                    raise ValueError('术语concept_id不在当前测试样本的候选集合中')
                decoded['concept_id'] = u['concept_id']
    except (TypeError, ValueError, KeyError) as ex:
        decoded['status'] = 'invalid_structure'
        decoded['error'] = str(ex)
        return decoded

    decoded['valid_prediction'] = True
    explicit_abstain = raw.get('mapping_status') in ('unmapped', 'partial')
    no_semantics = (decoded['intent_id'] is None and not decoded['slots']) if case['dataset'] == 'dialect' else decoded['concept_id'] is None
    decoded['unmapped'] = bool(explicit_abstain or no_semantics)
    if decoded['unmapped']:
        decoded['status'] = 'unmapped'
    return decoded


def evaluate(cases: list[dict], by_id: dict[str, dict], dataset: str, method: str):
    """计算该方法在该数据集的指标。所有数据都放入分母。"""
    n = len(cases)
    counts = defaultdict(int)
    tp = fp = fn = 0
    detail = []
    errors = []

    for case in cases:
        d = decode_record(case, by_id[case['id']])
        counts['valid'] += int(d['valid_prediction'])
        counts['technical'] += int(d['technical_error'])
        counts['invalid'] += int(d['status'] == 'invalid_structure')
        counts['unmapped'] += int(d['unmapped'])

        row = {
            'id': case['id'], 'dataset': dataset, 'method': method,
            'text': case.get('text', ''), 'status': d['status'],
            'error': d['error'], 'correct': False,
            'runtime_ms': by_id[case['id']].get('runtime_ms'),
        }
        if dataset == 'dialect':
            gold_intent = case['gold']['intent_id']
            expected = gold_slots(case)
            actual = d['slots']
            got_intent = d['intent_id']
            intent_ok = d['valid_prediction'] and (got_intent == gold_intent)
            slots_ok = d['valid_prediction'] and (actual == expected)
            row.update({
                'gold_intent': gold_intent, 'pred_intent': got_intent,
                'gold_slots': sorted(expected), 'pred_slots': sorted(actual),
                'intent_correct': bool(intent_ok),
                'slots_exact': bool(slots_ok),
                'correct': bool(intent_ok and slots_ok),
            })
            counts['intent_correct'] += int(intent_ok)
            counts['slots_exact'] += int(slots_ok)
            counts['joint_correct'] += int(intent_ok and slots_ok)
            tp += len(actual & expected)
            fp += len(actual - expected)
            fn += len(expected - actual)
            if not intent_ok and not slots_ok:
                row['error_type'] = 'intent_and_slots'
            elif not intent_ok:
                row['error_type'] = 'intent'
            elif not slots_ok:
                row['error_type'] = 'slots'
        else:
            gold = case['gold']['concept_id']
            predicted = d['concept_id']
            candidate_ids = {c['id'] for c in case.get('catalog', []) if isinstance(c, dict) and 'id' in c}
            retrieved = gold in candidate_ids
            correct = d['valid_prediction'] and predicted == gold
            counts['concept_correct'] += int(correct)
            counts['covered'] += int(d['valid_prediction'] and predicted is not None)
            counts['candidate_present'] += int(retrieved)
            counts['candidate_present_correct'] += int(correct and retrieved)
            row.update({
                'gold_concept': gold, 'pred_concept': predicted,
                'gold_in_candidates': retrieved, 'correct': bool(correct),
            })
            if not retrieved:
                row['error_type'] = 'gold_not_in_candidates'
            elif predicted is None:
                row['error_type'] = 'abstained_or_failed'
            elif not correct:
                row['error_type'] = 'wrong_concept'
        if d['technical_error']:
            row['error_type'] = 'technical_error'
        elif d['status'] == 'invalid_structure':
            row['error_type'] = 'invalid_structure'

        detail.append(row)
        if not row['correct']:
            errors.append(row)

    report = {
        'dataset': dataset, 'method': method, 'samples': n,
        # 是「结构可被程序使用的预测比例」，不是模型生成的原始JSON有效率。
        'usable_output_rate': counts['valid'] / n,
        'technical_error_rate': counts['technical'] / n,
        'invalid_structure_rate': counts['invalid'] / n,
        'unmapped_rate': counts['unmapped'] / n,
        'errors': len(errors),
    }

    if dataset == 'dialect':
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        report.update({
            'intent_accuracy': counts['intent_correct'] / n,
            'slot_precision': p, 'slot_recall': r, 'slot_micro_f1': f1,
            'slot_exact_accuracy': counts['slots_exact'] / n,
            'joint_exact_match': counts['joint_correct'] / n,
        })
    else:
        covered = counts['covered']
        present = counts['candidate_present']
        report.update({
            'concept_accuracy': counts['concept_correct'] / n,
            'mapping_coverage': covered / n,
            # 仅在已输出概念的样本上计算的准确率，不替代全样本Accuracy。
            'accuracy_on_mapped': counts['concept_correct'] / covered if covered else None,
            # 检索阶段的正确概念召回（对同一固定测试集各方法应一致）。
            'candidate_gold_recall': present / n,
            'accuracy_when_candidate_present': (
                counts['candidate_present_correct'] / present if present else None
            ),
        })
    return report, detail, errors


def percent(x) -> str:
    return 'N/A' if x is None else f'{100*x:.1f}%'


def make_plots(summary: list[dict]) -> None:
    if not DRAW_FIGURES:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('没有安装matplotlib：已生成全部CSV和JSONL，跳过PNG图表。')
        print('可以在PyCharm的 Python Packages 中安装 matplotlib。')
        return

    figure_specs = {
        'dialect': [
            ('intent_accuracy', 'Intent Accuracy'),
            ('slot_micro_f1', 'Slot Micro-F1'),
            ('joint_exact_match', 'Joint Exact Match'),
        ],
        'term': [
            ('concept_accuracy', 'Concept Accuracy'),
            ('mapping_coverage', 'Mapping Coverage'),
        ],
    }
    for ds, metrics in figure_specs.items():
        rows = [r for r in summary if r['dataset'] == ds]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(11, 5.3))
        width = 0.75 / len(metrics)
        for j, (field, label) in enumerate(metrics):
            positions = [i - .375 + width * (j + .5) for i in range(len(rows))]
            values = [r.get(field) or 0 for r in rows]
            ax.bar(positions, values, width=width, label=label)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r['method'] for r in rows], rotation=15)
        ax.set_ylim(0, 1.06)
        ax.set_ylabel('Score (0 to 1)')
        ax.set_title(f'{ds.capitalize()} dataset: methods on identical test cases')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()
        outfile = RESULTS_DIR / f'{ds}_comparison.png'
        fig.savefig(outfile, dpi=190)
        plt.close(fig)
        print('已生成图表：', outfile)

    # 额外：比较结构可用率、弃权率、技术错误率，而不只呈现准确率。
    rows = summary
    if rows:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        names = [f'{r["dataset"]}-{r["method"]}' for r in rows]
        fields = [('usable_output_rate', 'Usable Output'),
                  ('unmapped_rate', 'Unmapped'),
                  ('technical_error_rate', 'Technical Error')]
        width = 0.24
        for j, (field, label) in enumerate(fields):
            ax.bar([i + (j - 1) * width for i in range(len(rows))],
                   [r[field] for r in rows], width=width, label=label)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, rotation=35, ha='right')
        ax.set_ylim(0, 1.06)
        ax.set_ylabel('Rate')
        ax.set_title('Output status comparison')
        ax.legend()
        fig.tight_layout()
        outfile = RESULTS_DIR / 'output_status_comparison.png'
        fig.savefig(outfile, dpi=190)
        plt.close(fig)
        print('已生成图表：', outfile)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases, predictions = read_all_data()
    cases_by_dataset = defaultdict(list)
    for case in cases:
        cases_by_dataset[case['dataset']].append(case)

    summary = []
    detailed = defaultdict(dict)  # (dataset,id) -> method -> one record
    methods_loaded = [m for m in METHODS if m in predictions]

    for dataset, ds_cases in cases_by_dataset.items():
        for method in methods_loaded:
            report, detail, errors = evaluate(ds_cases, predictions[method], dataset, method)
            summary.append(report)
            write_jsonl(RESULTS_DIR / f'errors_{dataset}_{method}.jsonl', errors)
            for item in detail:
                detailed[(dataset, item['id'])][method] = item

    columns = [
        'dataset', 'method', 'samples',
        'intent_accuracy', 'slot_precision', 'slot_recall', 'slot_micro_f1',
        'slot_exact_accuracy', 'joint_exact_match',
        'concept_accuracy', 'mapping_coverage', 'accuracy_on_mapped',
        'candidate_gold_recall', 'accuracy_when_candidate_present',
        'usable_output_rate', 'technical_error_rate', 'invalid_structure_rate',
        'unmapped_rate', 'errors',
    ]
    write_csv(RESULTS_DIR / 'summary.csv', summary, columns)

    # 每条测试样本横向对齐六个方法，方便直接打开Excel检查。
    per_case = []
    comparison_cols = ['dataset', 'id', 'text', 'gold']
    for method in methods_loaded:
        comparison_cols += [f'{method}_correct', f'{method}_status',
                            f'{method}_prediction', f'{method}_error']
    for case in cases:
        row = {'dataset': case['dataset'], 'id': case['id'],
               'text': case.get('text', ''),
               'gold': json.dumps(case['gold'], ensure_ascii=False)}
        for method in methods_loaded:
            d = detailed[(case['dataset'], case['id'])][method]
            row[f'{method}_correct'] = int(d['correct'])
            row[f'{method}_status'] = d['status']
            value = (d.get('pred_intent') if case['dataset'] == 'dialect'
                     else d.get('pred_concept'))
            if case['dataset'] == 'dialect':
                value = {'intent': value, 'slots': d.get('pred_slots', [])}
            row[f'{method}_prediction'] = json.dumps(value, ensure_ascii=False)
            row[f'{method}_error'] = d['error']
        per_case.append(row)
    write_csv(RESULTS_DIR / 'per_sample_comparison.csv', per_case, comparison_cols)

    # 主比较：Ours相对其他方法在每条样本上的赢/输/平。
    pairwise = []
    if PRIMARY_METHOD in methods_loaded:
        for dataset, ds_cases in cases_by_dataset.items():
            for other in methods_loaded:
                if other == PRIMARY_METHOD:
                    continue
                won = lost = both_correct = both_wrong = 0
                for case in ds_cases:
                    d = detailed[(dataset, case['id'])]
                    a, b = d[PRIMARY_METHOD]['correct'], d[other]['correct']
                    if a and not b:
                        won += 1
                    elif not a and b:
                        lost += 1
                    elif a and b:
                        both_correct += 1
                    else:
                        both_wrong += 1
                pairwise.append({'dataset': dataset, 'compare_to': other,
                    'samples': len(ds_cases), 'ours_win': won, 'ours_lose': lost,
                    'both_correct': both_correct, 'both_wrong': both_wrong,
                    'accuracy_difference': (won-lost) / len(ds_cases)})
    write_csv(RESULTS_DIR / f'pairwise_vs_{PRIMARY_METHOD}.csv', pairwise,
              ['dataset', 'compare_to', 'samples', 'ours_win', 'ours_lose',
               'both_correct', 'both_wrong', 'accuracy_difference'])

    print('\n==== 六种方法统一评价（所有失败样本均计入分母） ====')
    for ds in ('dialect', 'term'):
        sub = [r for r in summary if r['dataset'] == ds]
        if not sub:
            continue
        print(f'\n[{ds}] 共 {sub[0]["samples"]} 条')
        for row in sub:
            if ds == 'dialect':
                main_score = f'意图Acc={percent(row["intent_accuracy"])} / 槽位F1={percent(row["slot_micro_f1"])} / 联合EM={percent(row["joint_exact_match"])}'
            else:
                main_score = f'术语Acc={percent(row["concept_accuracy"])} / 覆盖率={percent(row["mapping_coverage"])}'
            print(f'{row["method"]:10s} {main_score} | 技术错误={percent(row["technical_error_rate"])} / 未映射={percent(row["unmapped_rate"])}')
    print('\n已生成：summary.csv、per_sample_comparison.csv、pairwise_vs_ours.csv 和各方法错误明细。')
    if any(c['dataset'] == 'term' for c in cases):
        print('注意：术语Acc的上限受候选集正确概念召回率影响；见summary.csv的candidate_gold_recall。')
    print('注意：样本数很小时仅适合作为流程试运行，不宜据此作显著性或泛化结论。')
    make_plots(summary)


if __name__ == '__main__':
    main()
