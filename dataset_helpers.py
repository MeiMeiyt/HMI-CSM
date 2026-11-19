"""Original dataset parsing helpers, without sample limits or credentials."""
import re
from pathlib import Path
from decimal import Decimal, InvalidOperation

def read_dialect(path: Path) -> list[dict]:
    """NaLiBaSID/xSID类CoNLL：句间空行，# intent注释及BIO token标签。
    使用数据原有Gold，模型提示中不会出现测试Gold。
    """
    if not path.is_file():
        raise FileNotFoundError(f'缺少数据文件：{path}；从NaLiBaSID下载解压后放入data目录。')
    contents = path.read_text(encoding='utf-8-sig')
    result = []
    for section in re.split(r'\n\s*\n', contents.strip()):
        meta, tokens, tags = {}, [], []
        for line in section.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                m = re.match(r'#\s*([\w-]+)\s*[:=]\s*(.+)', line)
                if m:
                    meta[m.group(1).lower()] = m.group(2).strip()
                continue
            cols = line.split('\t')
            # 已有项目中支持两类导出：4+列token位于第二列、BIO位于第四列；或token\tBIO
            if len(cols) >= 4:
                token, tag = cols[1], cols[3]
            elif len(cols) == 2:
                token, tag = cols
            else:
                continue
            if tag == 'O' or re.fullmatch(r'[BI]-.+', tag):
                tokens.append(token)
                tags.append(tag)
        if tokens and meta.get('intent'):
            result.append({'text': meta.get('text', ' '.join(tokens)),
                           'tokens': tokens, 'tags': tags, 'intent': meta['intent']})
    if not result:
        raise ValueError(f'未成功解析NaLiBaSID数据：{path}，请核对实际文件列格式。')
    return result

def normalize_sct(value) -> str | None:
    val = str(value or '').strip()
    if not val or val.lower() in ('nan', 'null', 'none', '0', '-1'):
        return None
    if val.isdecimal():
        return val
    try:
        d = Decimal(val)
        if d.is_finite() and d > 0 and d == d.to_integral_value():
            return str(int(d))
    except InvalidOperation:
        pass
    return None

def biospans(tags: list[str]) -> list[tuple[str, int, int]]:
    result, start, active = [], None, None
    for i, tag in enumerate(tags + ['O']):
        label = tag[2:] if tag.startswith(('B-', 'I-')) else None
        if start is not None and (tag == 'O' or tag.startswith('B-') or label != active):
            result.append(('slot:' + active, start, i))
            start, active = None, None
        if tag.startswith('B-') or (tag.startswith('I-') and start is None):
            start, active = i, label
    return result

def dialect_example(row):
    return {'input': row['text'], 'output': {'intent_id': 'intent:' + row['intent'],
            'units': [{'unit_id': f'u{i}', 'category': 'slot', 'concept_id': cid,
                       'surface': ' '.join(row['tokens'][a:b]), 'token_span': [a,b], 'qualifiers': {}}
                      for i, (cid, a, b) in enumerate(biospans(row['tags']), 1)], 'links': []}}
