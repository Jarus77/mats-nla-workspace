"""Natural-text generators for the exp 19 bank. Each takes [(name, text)] source files and an rng and returns
items WITHOUT token ids: {id, feature, cluster, value, text, meta}. Both values of a row are read points inside
the SAME source file (minimal pairs), so the two classes share vocabulary. The read position is the last
character of ``text``; ``text`` is the file prefix up to that point, cut to the last CONTEXT chars at a line
boundary. No model, no tokenizer; unit-tested on small snippets."""
from __future__ import annotations

import ast
import io
import random
import re
import tokenize

CONTEXT = 1200          # characters of preceding context kept
MIN_CONTEXT = 150
SEED = 2222


def _prefix(text: str, end: int) -> str | None:
    if end < MIN_CONTEXT:
        return None
    p = text[:end]
    if len(p) > CONTEXT:
        cut = p.rfind("\n", 0, len(p) - CONTEXT)
        p = p[cut + 1:] if cut >= 0 else p[-CONTEXT:]
    return p if not p.endswith((" ", "\t", "\n")) else None


def _emit(out: list, feature: str, name: str, k: int, value: str, text: str, meta: dict) -> None:
    out.append({"id": f"{feature}_{re.sub(r'[^A-Za-z0-9]', '', name)[-24:]}_{k}", "feature": feature,
                "cluster": name, "value": value, "text": text, "meta": meta})


def _word_end_inside(s: str, lo: int, hi: int) -> list[int]:
    """Offsets in s[lo:hi] that end a word of 3+ letters followed by a space."""
    return [m.end() for m in re.finditer(r"[A-Za-z]{3,}(?= )", s[lo:hi])]


# ------------------------------------------------------------------ py_string
def _py_tokens(text: str):  # noqa: ANN202
    try:
        return list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return []


def _line_offsets(text: str) -> list[int]:
    offs, pos = [0], 0
    for line in text.splitlines(keepends=True):
        pos += len(line)
        offs.append(pos)
    return offs


def build_py_string(files, rng: random.Random, per_file: int = 2) -> list[dict]:  # noqa: ANN001
    out = []
    for name, text in files:
        offs = _line_offsets(text)
        toks = _py_tokens(text)
        k = 0
        for i, t in enumerate(toks):
            if t.type != tokenize.STRING or t.start[0] != t.end[0]:
                continue
            s = t.string
            q = s[-1]
            if q not in "\"'" or s[:3] in ('"""', "'''") or len(s) < 20 or s[0] not in "\"'rRbBuU":
                continue
            start = offs[t.start[0] - 1] + t.start[1]
            body_lo = start + (s.index(q) + 1)
            body_hi = start + len(s) - 1
            ins = [body_lo + e for e in _word_end_inside(text, body_lo, body_hi) if e >= 8]
            # outside: the end of a NAME token at least 2 tokens after this string, on the next 2 lines, no string between
            outs = []
            for u in toks[i + 1: i + 12]:
                if u.type == tokenize.STRING:
                    break
                if u.type == tokenize.NAME and u.start[0] <= t.end[0] + 2 and u is not toks[i + 1]:
                    outs.append(offs[u.start[0] - 1] + u.end[1])
            if ins and outs and k < per_file:
                pi, po = _prefix(text, rng.choice(ins)), _prefix(text, rng.choice(outs))
                if pi and po:
                    _emit(out, "py_string", name, k, "inside", pi, {"quote": q})
                    _emit(out, "py_string", name, k + 1, "outside", po, {"quote": q})
                    k += 2
    return out


# ------------------------------------------------------------------ py_arg
def build_py_arg(files, rng: random.Random, per_file: int = 2) -> list[dict]:  # noqa: ANN001
    out = []
    for name, text in files:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        offs = _line_offsets(text)
        cands = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and len(node.args) >= 3 and not any(isinstance(a, ast.Starred) for a in node.args):
                a0, a2 = node.args[0], node.args[2]
                if a0.end_lineno != a2.end_lineno or any(isinstance(a, (ast.Call, ast.Constant)) and isinstance(getattr(a, "value", None), str) for a in (a0, a2)):
                    continue
                e0 = offs[a0.end_lineno - 1] + a0.end_col_offset
                e2 = offs[a2.end_lineno - 1] + a2.end_col_offset
                if text[e0 - 1].isalnum() and text[e2 - 1].isalnum():
                    cands.append((e0, e2))
        rng.shuffle(cands)
        k = 0
        for e0, e2 in cands:
            if k >= per_file:
                break
            p0, p2 = _prefix(text, e0), _prefix(text, e2)
            if p0 and p2:
                _emit(out, "py_arg", name, k, "first", p0, {})
                _emit(out, "py_arg", name, k + 1, "third", p2, {})
                k += 2
    return out


# ------------------------------------------------------------------ js_string
_JS_STR = re.compile(r"""(["'])((?:(?!\1)[^\\\n])*)\1""")


def build_js_string(files, rng: random.Random, per_file: int = 2) -> list[dict]:  # noqa: ANN001
    out = []
    for name, text in files:
        k = 0
        for m in _JS_STR.finditer(text):
            body = m.group(2)
            if len(body) < 18 or len(body.split()) < 3:
                continue
            q = m.group(1)
            ins = [m.start(2) + e for e in _word_end_inside(text, m.start(2), m.end(2)) if e >= 8]
            after = text[m.end(): m.end() + 160]
            nxt = _JS_STR.search(after)
            lim = nxt.start() if nxt else len(after)
            outs = [m.end() + w.end() for w in re.finditer(r"[A-Za-z_]{3,}", after[:lim]) if w.end() > 3][1:]
            if ins and outs and k < per_file:
                pi, po = _prefix(text, rng.choice(ins)), _prefix(text, rng.choice(outs))
                if pi and po:
                    _emit(out, "js_string", name, k, "inside", pi, {"quote": q})
                    _emit(out, "js_string", name, k + 1, "outside", po, {"quote": q})
                    k += 2
    return out


# ------------------------------------------------------------------ md_heading / md_list
def _lines_with_offsets(text: str):  # noqa: ANN202
    pos = 0
    for line in text.splitlines(keepends=True):
        yield pos, line
        pos += len(line)


def build_md_heading(files, rng: random.Random, per_file: int = 2) -> list[dict]:  # noqa: ANN001
    out = []
    for name, text in files:
        sh, dp = [], []
        for pos, line in _lines_with_offsets(text):
            m = re.match(r"(#{1,4}) (.+)", line.rstrip("\n"))
            if not m:
                continue
            words = _word_end_inside(line, m.start(2), len(line.rstrip("\n")))
            words = [pos + m.start(2) + e for e in words]
            if len(words) < 2:
                continue
            (sh if len(m.group(1)) <= 2 else dp).append(words[min(len(words) - 1, rng.randint(1, 3))])
        rng.shuffle(sh), rng.shuffle(dp)
        k = 0
        for a, b in zip(sh, dp):
            if k >= per_file:
                break
            pa, pb = _prefix(text, a), _prefix(text, b)
            if pa and pb:
                _emit(out, "md_heading", name, k, "shallow", pa, {})
                _emit(out, "md_heading", name, k + 1, "deep", pb, {})
                k += 2
    return out


def build_md_list(files, rng: random.Random, per_file: int = 2) -> list[dict]:  # noqa: ANN001
    out = []
    for name, text in files:
        top, nested = [], []
        for pos, line in _lines_with_offsets(text):
            m = re.match(r"( *)[-*] (.+)", line.rstrip("\n"))
            if not m:
                continue
            words = [pos + m.start(2) + e for e in _word_end_inside(line, m.start(2), len(line.rstrip("\n")))]
            if len(words) < 2:
                continue
            (top if len(m.group(1)) == 0 else nested).append(rng.choice(words[1:]))
        rng.shuffle(top), rng.shuffle(nested)
        k = 0
        for a, b in zip(top, nested):
            if k >= per_file:
                break
            pa, pb = _prefix(text, a), _prefix(text, b)
            if pa and pb:
                _emit(out, "md_list", name, k, "top", pa, {})
                _emit(out, "md_list", name, k + 1, "nested", pb, {})
                k += 2
    return out


# ------------------------------------------------------------------ json_depth
def _json_string_values(text: str) -> list[tuple[int, int]]:
    """(end offset of the closing quote, depth) for every string VALUE (not key) in raw JSON text."""
    out, depth, i, n = [], 0, 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            k = j + 1
            while k < n and text[k] in " \t\r\n":
                k += 1
            is_key = k < n and text[k] == ":"
            if not is_key and j - i > 8:
                out.append((j + 1, depth))
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return out


def build_json_depth(files, rng: random.Random, per_file: int = 2) -> list[dict]:  # noqa: ANN001
    out = []
    for name, text in files:
        vals = _json_string_values(text)
        sh = [e for e, d in vals if d == 1]
        dp = [e for e, d in vals if d >= 3]
        rng.shuffle(sh), rng.shuffle(dp)
        k = 0
        for a, b in zip(sh, dp):
            if k >= per_file:
                break
            pa, pb = _prefix(text, a), _prefix(text, b)
            if pa and pb:
                _emit(out, "json_depth", name, k, "shallow", pa, {"depth": 1})
                _emit(out, "json_depth", name, k + 1, "deep", pb, {"depth": 3})
                k += 2
    return out


# ------------------------------------------------------------------ csv_field
_CSV_FIELD = r"[A-Za-z0-9][A-Za-z0-9 ._:-]{1,}[A-Za-z0-9]"


def build_csv_field(files, rng: random.Random, per_file: int = 2) -> list[dict]:  # noqa: ANN001
    out = []
    for name, text in files:
        firsts, fourths = [], []
        for pos, line in _lines_with_offsets(text):
            if pos == 0:
                continue  # header
            parts = line.rstrip("\n").split(",")
            if len(parts) < 4 or any('"' in p for p in parts[:4]):
                continue
            if re.fullmatch(_CSV_FIELD, parts[0]):
                firsts.append(pos + len(parts[0]))
            if re.fullmatch(_CSV_FIELD, parts[3]):
                fourths.append(pos + len(",".join(parts[:4])))
        rng.shuffle(firsts), rng.shuffle(fourths)
        k = 0
        for a, b in zip(firsts, fourths):
            if k >= per_file:
                break
            pa, pb = _prefix(text, a), _prefix(text, b)
            if pa and pb:
                _emit(out, "csv_field", name, k, "first", pa, {})
                _emit(out, "csv_field", name, k + 1, "fourth", pb, {})
                k += 2
    return out


BUILDERS = {"py_string": build_py_string, "py_arg": build_py_arg, "js_string": build_js_string,
            "md_heading": build_md_heading, "md_list": build_md_list, "json_depth": build_json_depth,
            "csv_field": build_csv_field}


# ------------------------------------------------------------------ html_depth
_HTML_VOID = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}
HTML_SHALLOW, HTML_DEEP = 3, 6      # open elements at the read point (html and body count); real pages start at 2
_HTML_TAG = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)[^<>]*?(/?)>")


def _html_text_points(text: str) -> list[tuple[int, int]]:
    """(offset at the end of a word inside element text, open-element depth) for every text run."""
    out, depth, pos, raw = [], 0, 0, False      # raw: inside <script> or <style>, whose text is not page text
    for m in _HTML_TAG.finditer(text):
        seg = text[pos: m.start()]
        if seg.strip() and "<" not in seg and not raw:
            words = _word_end_inside(text, pos, m.start())
            if len(words) >= 3:
                out.append((pos + words[min(len(words) - 1, 2)], depth))
        closing, tag, selfclose = m.group(1), m.group(2).lower(), m.group(3)
        if tag in ("script", "style"):
            raw = not closing
        if not closing and not selfclose and tag not in _HTML_VOID:
            depth += 1
        elif closing:
            depth = max(0, depth - 1)
        pos = m.end()
    return out


def build_html_depth(files, rng: random.Random, per_file: int = 8) -> list[dict]:  # noqa: ANN001
    out = []
    for name, text in files:
        pts = _html_text_points(text)
        sh = [e for e, d in pts if d <= HTML_SHALLOW]
        dp = [e for e, d in pts if d >= HTML_DEEP]
        rng.shuffle(sh), rng.shuffle(dp)
        k = 0
        for a, b in zip(sh, dp):
            if k >= per_file:
                break
            pa, pb = _prefix(text, a), _prefix(text, b)
            if pa and pb:
                _emit(out, "html_depth", name, k, "shallow", pa, {})
                _emit(out, "html_depth", name, k + 1, "deep", pb, {})
                k += 2
    return out


# ------------------------------------------------------------------ latex_math
_TEX_INLINE = re.compile(r"(?<!\\)\$(?!\$)([^$\n]{6,}?)(?<!\\)\$")


def build_latex_math(files, rng: random.Random, per_file: int = 2) -> list[dict]:  # noqa: ANN001
    out = []
    for name, text in files:
        k = 0
        for m in _TEX_INLINE.finditer(text):
            if "$$" in text[max(0, m.start() - 2): m.end() + 2]:
                continue
            body_lo, body_hi = m.start(1), m.end(1)
            ins = [body_lo + w.end() for w in re.finditer(r"[A-Za-z]{1,}(?= )", text[body_lo:body_hi]) if w.end() >= 3]
            after = text[m.end(): m.end() + 200]
            nd = after.find("$")
            lim = nd if nd >= 0 else len(after)
            outs = [m.end() + w.end() for w in re.finditer(r"[A-Za-z]{3,}(?= )", after[:lim])][1:]
            if ins and outs and k < per_file:
                pi, po = _prefix(text, rng.choice(ins)), _prefix(text, rng.choice(outs))
                if pi and po and pi.count("$") % 2 == 1 and po.count("$") % 2 == 0:
                    _emit(out, "latex_math", name, k, "inside", pi, {})
                    _emit(out, "latex_math", name, k + 1, "outside", po, {})
                    k += 2
    return out


BUILDERS = {**BUILDERS, "html_depth": build_html_depth, "latex_math": build_latex_math}


# ------------------------------------------------------------------ code_lang (positive control for the statement rule)
def build_code_lang(files_py, files_js, rng: random.Random, per_file: int = 2) -> list[dict]:  # noqa: ANN001
    """Read points outside string literals in Python vs JavaScript files: a property the NLA is expected to STATE
    (it names the language of code freely), so the statement-shift rule is tested in the positive direction too."""
    out = []
    for it in build_py_string(files_py, rng, per_file):
        if it["value"] == "outside":
            out.append({**it, "id": it["id"].replace("py_string", "code_lang"), "feature": "code_lang", "value": "python"})
    for it in build_js_string(files_js, rng, per_file):
        if it["value"] == "outside":
            out.append({**it, "id": it["id"].replace("js_string", "code_lang"), "feature": "code_lang", "value": "javascript"})
    return out


BUILDERS = {**BUILDERS, "code_lang": None}      # built by build22.py from two file pools
