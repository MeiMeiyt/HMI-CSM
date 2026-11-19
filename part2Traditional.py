"""Publication version: shared configuration, no embedded credentials."""
from __future__ import annotations
import hmi_config as _release

import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# ===================== PyCharm修改区域 =====================
RUN_MODE = "run" if _release.MODE == "real" else "inspect"
METHODS = ('lexicon', 'tfidf')

# 以下阈值只是试运行默认值；正式实验应在验证集上选择，不可使用测试集调参。
LEXICON_TERM_MIN_SIM = 0.40
TFIDF_TERM_MIN_SIM = 0.25
TFIDF_SLOT_MIN_SIM = 0.75
TFIDF_MAX_SLOT_TOKENS = 5
# ============================================================

ROOT = _release.ROOT
OUT = _release.STAGE1
DEV_FILE = _release.data_path("stage1", "dialect_validation")
SCHEMA_VERSION = '1.2'
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def case_fingerprint(case: dict) -> str:
    """与第一部分的指纹算法一致，供第三部分检查是否同一批数据。"""
    payload = json.dumps(case, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def schema(case: dict) -> dict:
    """与本文方法相同的JSON结构；只建立空模板，不读取测试答案。"""
    return {
        'schema_version': SCHEMA_VERSION,
        'source': {
            'speaker': case['speaker'],
            'text': case['text'],
            'domain': case['domain'],
        },
        'shared_semantics': {'intent_id': None, 'units': [], 'links': []},
        'evidence': [],
        'mapping_status': 'unmapped',
    }


def read_dialect(path: Path) -> list[dict]:
    """读取非测试方言验证集，支持NaLiBaSID的常见CoNLL文本格式。"""
    if not path.is_file():
        raise FileNotFoundError(f'缺少方言验证集：{path}')
    text = path.read_text(encoding='utf-8-sig')
    records = []
    for chunk in re.split(r'\n\s*\n', text.strip()):
        meta, tokens, tags = {}, [], []
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                match = re.match(r'#\s*([\w-]+)\s*[:=]\s*(.+)', line)
                if match:
                    meta[match.group(1).lower()] = match.group(2).strip()
                continue
            fields = line.split('\t')
            if len(fields) >= 4:
                token, tag = fields[1], fields[3]
            elif len(fields) == 2:
                token, tag = fields
            else:
                continue
            if tag == 'O' or re.fullmatch(r'[BI]-.+', tag):
                tokens.append(token)
                tags.append(tag)
        if tokens and meta.get('intent'):
            records.append({
                'text': meta.get('text', ' '.join(tokens)),
                'tokens': tokens,
                'tags': tags,
                'intent': meta['intent'],
            })
    if not records:
        raise ValueError('无法解析方言验证集：请检查文件名、分列和intent/BIO标签。')
    return records


def biospans(tags: list[str]) -> list[tuple[str, int, int]]:
    """将BIO标签转换为(slot:类别, 起始token, 结束token)列表。"""
    spans = []
    begin, active = None, None
    for i, tag in enumerate(tags + ['O']):
        label = tag[2:] if tag.startswith(('B-', 'I-')) else None
        if begin is not None and (tag == 'O' or tag.startswith('B-') or label != active):
            spans.append(('slot:' + active, begin, i))
            begin, active = None, None
        if tag.startswith('B-') or (tag.startswith('I-') and begin is None):
            begin, active = i, label
    return spans


def norm(text: str) -> str:
    """Unicode/大小写/标点/空格归一化，不翻译，不引入任何gold。"""
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    text = re.sub(r'[^\w\s]+', ' ', text, flags=re.UNICODE)
    return ' '.join(text.split())


def str_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def char_ngrams(text: str) -> Counter[str]:
    s = norm(text)
    if not s:
        return Counter()
    padded = f' {s} '
    grams = Counter(padded[i:i+n] for n in (2, 3, 4)
                    for i in range(max(0, len(padded)-n+1)))
    return grams if grams else Counter({s: 1})


class CharTfidfIndex:
    """标准TF-IDF权重 + L2规范化 + 余弦相似度，纯Python实现。"""

    def __init__(self, texts: list[str]):
        self.texts = texts
        bag = [char_ngrams(x) for x in texts]
        n = len(bag)
        df: Counter[str] = Counter()
        for item in bag:
            df.update(item.keys())
        self.idf = {term: math.log((n + 1) / (freq + 1)) + 1
                    for term, freq in df.items()}
        self.vectors = [self._vector(x) for x in bag]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        vec = {t: (1 + math.log(f)) * self.idf[t]
               for t, f in counts.items() if t in self.idf and f > 0}
        length = math.sqrt(sum(v*v for v in vec.values()))
        return {t: v/length for t, v in vec.items()} if length else {}

    def similarities(self, query: str) -> list[float]:
        q = self._vector(char_ngrams(query))
        if not q:
            return [0.0] * len(self.vectors)
        return [sum(val * doc.get(token, 0.0) for token, val in q.items())
                for doc in self.vectors]


def read_reference_data(cases: list[dict]) -> list[dict]:
    """方言只读取NaLiBaSID验证集作为参考库，绝不读取测试标签。"""
    if not any(case['dataset'] == 'dialect' for case in cases):
        return []
    dev = read_dialect(DEV_FILE)
    if not dev:
        raise ValueError('方言验证集没有有效标注样本。')
    return dev


def make_slot_dictionary(dev: list[dict]):
    lexicon: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in dev:
        for code, start, end in biospans(row['tags']):
            key = tuple(norm(t) for t in row['tokens'][start:end])
            if key and all(key):
                lexicon[key][code] += 1
    return lexicon


def exact_slot_match(case: dict, lexicon: dict) -> list[tuple[str, int, int]]:
    """长词优先的词典槽位匹配；歧义标签票数相同时弃权。"""
    tokens = [norm(t) for t in case['tokens']]
    max_len = max((len(k) for k in lexicon), default=0)
    spans = []
    i = 0
    while i < len(tokens):
        matched = None
        for k in range(min(max_len, len(tokens)-i), 0, -1):
            key = tuple(tokens[i:i+k])
            counts = lexicon.get(key)
            if not counts:
                continue
            ranking = counts.most_common()
            if len(ranking) > 1 and ranking[0][1] == ranking[1][1]:
                continue
            matched = (ranking[0][0], i, i+k)
            break
        if matched:
            spans.append(matched)
            i = matched[2]
        else:
            i += 1
    return spans


def tfidf_slot_match(case: dict, labels: list[str], index: CharTfidfIndex):
    """在验证集标注的槽位片段中做字符TF-IDF检索，阈值以下弃权。"""
    tokens = case['tokens']
    candidates = []
    for start in range(len(tokens)):
        for end in range(start+1, min(len(tokens), start+TFIDF_MAX_SLOT_TOKENS)+1):
            query = ' '.join(tokens[start:end])
            scores = index.similarities(query)
            if not scores:
                continue
            best = max(range(len(scores)), key=lambda k: scores[k])
            if scores[best] >= TFIDF_SLOT_MIN_SIM:
                candidates.append((scores[best], end-start, start, end, labels[best]))
    # 先取高置信度跨度，相同分数优先较长跨度；不允许重叠
    candidates.sort(key=lambda r: (-r[0], -r[1], r[2]))
    occupied = set()
    result = []
    for _, _, start, end, code in candidates:
        if any(pos in occupied for pos in range(start, end)):
            continue
        result.append((code, start, end))
        occupied.update(range(start, end))
    return sorted(result, key=lambda row: row[1])


def make_output(case: dict, intent: str | None, spans: list[tuple] | None = None,
                concept: str | None = None) -> dict:
    """与LLM方法完全相同的目标JSON。不得传入或读取case['gold']。"""
    obj = schema(case)
    sem = obj['shared_semantics']
    sem['intent_id'] = intent
    if case['dataset'] == 'dialect':
        allowed = {'slot:' + s for s in case['catalog']['slots']}
        for code, start, end in (spans or []):
            if code not in allowed:
                continue
            uid = f'u{len(sem["units"])+1}'
            surface = ' '.join(case['tokens'][start:end])
            sem['units'].append({
                'unit_id': uid, 'category': 'slot', 'concept_id': code,
                'surface': surface, 'token_span': [start, end], 'qualifiers': {}
            })
            if surface in case['text']:
                obj['evidence'].append({'unit_id': uid, 'text_span': surface})
    elif concept is not None:
        if concept not in {x['id'] for x in case['catalog']}:
            raise ValueError('传统方法预测了候选表之外的概念。')
        sem['units'].append({
            'unit_id': 'u1', 'category': 'term', 'concept_id': concept,
            'surface': case['text'], 'token_span': None, 'qualifiers': {}
        })
        obj['evidence'].append({'unit_id': 'u1', 'text_span': case['text']})
    obj['mapping_status'] = 'mapped' if sem['intent_id'] is not None or sem['units'] else 'unmapped'
    return obj


def term_aliases(case: dict) -> list[tuple[str, str]]:
    """仅从第一部分锁定的候选表提取非测试来源的别名，不读取测试gold。"""
    aliases = []
    for candidate in case['catalog']:
        for phrase in candidate.get('aliases', []):
            if norm(phrase):
                aliases.append((phrase, candidate['id']))
    return aliases


class TraditionalMapper:
    def __init__(self, dev: list[dict]):
        self.dev = dev
        self.dev_texts = [row['text'] for row in dev]
        self.intent_index = CharTfidfIndex(self.dev_texts) if dev else None
        self.slot_dict = make_slot_dictionary(dev)
        slots = []
        for row in dev:
            for code, start, end in biospans(row['tags']):
                slots.append((' '.join(row['tokens'][start:end]), code))
        self.slot_labels = [code for _, code in slots]
        self.slot_index = CharTfidfIndex([s for s, _ in slots]) if slots else None

    def lexicon(self, case: dict) -> dict:
        if case['dataset'] == 'dialect':
            if not self.dev:
                raise ValueError('缺少方言验证集参考语料。')
            # 字符串相似度检索近邻意图；词典精确匹配槽位
            best = max(range(len(self.dev)),
                       key=lambda i: str_sim(case['text'], self.dev[i]['text']))
            intent = 'intent:' + self.dev[best]['intent']
            spans = exact_slot_match(case, self.slot_dict)
            return make_output(case, intent, spans=spans)
        aliases = term_aliases(case)
        if not aliases:
            return make_output(case, None)
        scores = [(str_sim(case['text'], phrase), cid) for phrase, cid in aliases]
        best_score = max(s for s, _ in scores)
        best_codes = {cid for s, cid in scores if abs(s-best_score) < 1e-10}
        concept = next(iter(best_codes)) if best_score >= LEXICON_TERM_MIN_SIM and len(best_codes) == 1 else None
        return make_output(case, None, concept=concept)

    def tfidf(self, case: dict) -> dict:
        if case['dataset'] == 'dialect':
            if not self.dev or self.intent_index is None:
                raise ValueError('缺少方言验证集参考语料。')
            scores = self.intent_index.similarities(case['text'])
            best = max(range(len(scores)), key=lambda i: scores[i])
            intent = 'intent:' + self.dev[best]['intent']
            spans = (tfidf_slot_match(case, self.slot_labels, self.slot_index)
                     if self.slot_index else [])
            return make_output(case, intent, spans=spans)
        aliases = term_aliases(case)
        if not aliases:
            return make_output(case, None)
        idx = CharTfidfIndex([s for s, _ in aliases])
        scores = idx.similarities(case['text'])
        best_score = max(scores)
        codes = {cid for (_, cid), sc in zip(aliases, scores) if abs(sc-best_score) < 1e-10}
        concept = (next(iter(codes)) if best_score >= TFIDF_TERM_MIN_SIM and len(codes) == 1
                   else None)
        return make_output(case, None, concept=concept)


def main():
    manifest = _release.PREPARED / "fixed_test_cases.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError('没有找到fixed_test_cases.jsonl，请先运行第一部分。')
    cases = load_jsonl(manifest)
    if not cases:
        raise ValueError('固定测试集为空。')
    if len(set(case['id'] for case in cases)) != len(cases):
        raise ValueError('测试集包含重复样本ID。')
    dev = read_reference_data(cases)
    model = TraditionalMapper(dev)
    print(f'固定测试集：{len(cases)}条；方言非测试参考句：{len(dev)}条')
    print('传统方法不会调用大语言模型，也不会使用API Key。')
    if RUN_MODE == 'inspect':
        for dataset in ('dialect', 'term'):
            print(dataset, sum(x['dataset'] == dataset for x in cases), '条')
        return
    if RUN_MODE != 'run':
        raise ValueError('RUN_MODE必须为inspect或run。')
    for method in METHODS:
        if method not in ('lexicon', 'tfidf'):
            raise ValueError('未知方法：' + method)
        runner = getattr(model, method)
        outputs = []
        for idx, case in enumerate(cases, 1):
            start = time.perf_counter()
            try:
                pred, error = runner(case), None
            except (KeyError, ValueError, TypeError, IndexError) as ex:
                pred, error = None, f'{type(ex).__name__}: {ex}'
            outputs.append({
                'id': case['id'],
                'case_fingerprint': case_fingerprint(case),
                'dataset': case['dataset'], 'method': method,
                'prediction': pred, 'error': error,
                'runtime_ms': round((time.perf_counter()-start)*1000, 3),
            })
            status = ('ERROR' if pred is None else pred['mapping_status'].upper())
            print(f'{method}: {idx}/{len(cases)} -> {status}')
        output_file = OUT/f'{method}_predictions.jsonl'
        save_jsonl(output_file, outputs)
        print('文件已保存：', output_file)


if __name__ == '__main__':
    main()
