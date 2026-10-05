"""Item generators for template mining (run inside the Modal build container).

Every item: {id, feature, split ("mine"|"select"), cluster, value, ids (token ids TRUNCATED so the
read position is the last token), text (decoded prefix), meta}. All items are NEW: wikitext
paragraphs already used by 02 (wrap/prose/wrap_report pickles) are excluded, so mining/selection
items are disjoint from 02's gating data; downstream evaluation must exclude the ids recorded in
templates_frozen.json.

Reuse of 02 (read-only): ``s1_data._wikitext_paragraphs`` (cleaning), ``s1_data.build_wrap``
(NJ1 col: same W ~ U{30..90} textwrap generator and col definition), ``s1_data.build_prose`` +
``_clause_tense`` (spaCy tense labels), and 02's POS lexicon recipe (s2c) for NJ3 (on GPU in
tm_modal.py). Indentation, words-since-sentence-start, list index, caps and code generators are ours.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import textwrap
import types

from feature_spec import BINS, N_MINE_PER_VALUE, N_SELECT_PER_VALUE

E02 = "/root/ext02"
OUT02 = "/results/02_nonj_features"
SEED = 7007


def load_02(name: str) -> types.ModuleType:
    """Import a 02 module with a stub ``nonj_common`` (avoids building 02's Modal images)."""
    if "nonj_common" not in sys.modules:
        stub = types.ModuleType("nonj_common")
        stub.OUT, stub.VOLUMES, stub.nonj_image, stub.net_image = OUT02, {}, None, None
        sys.modules["nonj_common"] = stub
    spec = importlib.util.spec_from_file_location(f"e02_{name}", f"{E02}/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up in sys.modules
    spec.loader.exec_module(mod)
    return mod


def _norm(t: str) -> str:
    return " ".join(t.split())[:80]


def fresh_paragraphs(s1) -> list[str]:  # noqa: ANN001
    """Cleaned wikitext paragraphs (02's cleaner) excluding every paragraph 02 used."""
    import pickle

    used = set()
    for name in ("wrap", "prose", "wrap_report"):
        try:
            for it in pickle.load(open(f"{OUT02}/data/{name}.pkl", "rb")):
                txt = it["text"]
                if name == "wrap_report":
                    continue  # derived from wrap passages (already covered)
                used.add(_norm(txt))
        except FileNotFoundError:
            pass
    pool = s1._wikitext_paragraphs(500, 60000, SEED)
    return [p for p in pool if _norm(p) not in used]


def assign_splits(items: list[dict], rng: random.Random, n_mine: int, n_sel: int) -> list[dict]:
    """Per value: shuffle, first n_mine -> mine, next n_sel -> select (by cluster)."""
    out = []
    by_val: dict[str, list[dict]] = {}
    for it in items:
        by_val.setdefault(it["value"], []).append(it)
    for _, lst in sorted(by_val.items()):
        rng.shuffle(lst)
        for k, it in enumerate(lst[: n_mine + n_sel]):
            out.append({**it, "split": "mine" if k < n_mine else "select"})
    return out


def _trunc(tok, text: str, end_char: int) -> list[int]:  # noqa: ANN001
    return tok(text[:end_char], add_special_tokens=False)["input_ids"]


# ------------------------------------------------------------------ NJ1 col (02's build_wrap)
def _in_bin(x: int, lo_hi: tuple[int, int]) -> bool:
    return lo_hi[0] <= x <= lo_hi[1]


def build_col(tok, s1, paras: list[str], rng: random.Random) -> list[dict]:  # noqa: ANN001
    """02's ``build_wrap`` (W ~ U{30..90}, textwrap, 320 tokens) run on FRESH paragraphs; one read
    token per passage, inside a line (02's ``col`` definition: not a newline token, col >= 1)."""
    s1._wikitext_paragraphs = lambda *_a, **_k: paras
    wrap = s1.build_wrap(tok, n=len(paras))
    out = []
    for k, it in enumerate(wrap):
        val = "short" if k % 2 == 0 else "long"
        lo_hi = BINS["nj1_col"][val]
        cand = [j for j in range(8, len(it["ids"]))
                if not it["has_nl"][j] and _in_bin(it["col"][j], lo_hi)]
        if not cand:
            continue
        j = rng.choice(cand)
        out.append({"id": f"col_p{k}_t{j}", "feature": "nj1_col", "cluster": f"p{k}", "value": val,
                    "ids": it["ids"][: j + 1], "text": it["text"][: it["offsets"][j][1]],
                    "meta": {"col": it["col"][j], "width": it["width"], "pos": j}})
    return out


# ------------------------------------------------------------------ NJ2 tense (02's spaCy labels)
def build_tense(tok, nlp, s1, paras: list[str], rng: random.Random) -> list[dict]:  # noqa: ANN001
    s1._wikitext_paragraphs = lambda *_a, **_k: paras  # reuse build_prose on OUR paragraphs
    prose = s1.build_prose(tok, nlp, n=len(paras))
    out = []
    for k, it in enumerate(prose):
        cand = [(j, t) for j, t in enumerate(it["tense"]) if t in ("Past", "Pres") and 8 <= j < 120]
        if not cand:
            continue
        # wikitext is mostly past tense: take a present position whenever the item has one
        pres = [c for c in cand if c[1] == "Pres"]
        j, t = rng.choice(pres if pres else cand)
        out.append({"id": f"nj2_p{k}_t{j}", "feature": "nj2_tense", "cluster": f"p{k}",
                    "value": "past" if t == "Past" else "present", "ids": it["ids"][: j + 1],
                    "text": it["text"][: it["offsets"][j][1]], "meta": {"pos": j}})
    return out


# ------------------------------------------------------------------ NJ3 pool (labels on GPU)
def build_predpos_pool(tok, paras: list[str]) -> list[dict]:  # noqa: ANN001
    out = []
    for k, p in enumerate(paras):
        ids = tok(p, add_special_tokens=False)["input_ids"][:96]
        out.append({"id": f"nj3pool_p{k}", "feature": "nj3_predpos", "cluster": f"p{k}",
                    "value": None, "ids": ids, "text": tok.decode(ids), "meta": {}})
    return out


# ------------------------------------------------------------------ IJ2 language (papluca fr/de)
def build_lang(tok, rng: random.Random) -> list[dict]:  # noqa: ANN001
    from datasets import load_dataset

    out = []
    ds = load_dataset("papluca/language-identification", split="test")
    names = {"fr": "French", "de": "German"}
    seen = set()
    for r, ex in enumerate(ds):
        lg = ex["labels"]
        if lg not in names or len(ex["text"]) < 60 or ex["text"] in seen:
            continue
        seen.add(ex["text"])
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"][:128]
        out.append({"id": f"lang_test{r}", "feature": "ij2_lang", "cluster": f"lang_test{r}",
                    "value": names[lg], "ids": ids, "text": tok.decode(ids),
                    "meta": {"source": "papluca/test", "row": r}})
    return out


# ------------------------------------------------------------------ words since sentence start
def build_wordpos(tok, paras: list[str], rng: random.Random) -> list[dict]:  # noqa: ANN001
    import re

    out = []
    for k, p in enumerate(paras):
        val = "early" if k % 2 == 0 else "late"
        lo_hi = BINS["bk_wordpos"][val]
        # sentence starts: paragraph start or after [.!?] + space + capital
        starts = [0] + [m.end() for m in re.finditer(r"[.!?][\"')]? +(?=[A-Z])", p)]
        cand = []
        for m in re.finditer(r"[A-Za-z]+", p[:1200]):
            s0 = max(x for x in starts if x <= m.start())
            n = len(re.findall(r"\S+", p[s0:m.end()]))
            if _in_bin(n, lo_hi) and m.end() < len(p) and not p[m.end()].isalnum():
                cand.append((m.end(), n))
        if not cand:
            continue
        end, n = rng.choice(cand)
        text = p[:end]
        out.append({"id": f"wpos_p{k}_c{end}", "feature": "bk_wordpos", "cluster": f"p{k}",
                    "value": val, "ids": tok(text, add_special_tokens=False)["input_ids"],
                    "text": text, "meta": {"words": n}})
    return out


# ------------------------------------------------------------------ list item index
_LISTS = {
    "groceries": "milk eggs bread butter cheese apples bananas rice pasta tomatoes onions garlic "
                 "coffee tea flour sugar yogurt carrots spinach lemons chicken beans oats honey",
    "packing": "passport charger toothbrush socks sunglasses umbrella jacket headphones notebook "
               "sandals scarf towel camera adapter wallet keys medicine snacks book pillow",
    "chores": "vacuum the hallway water the plants take out the trash wipe the counters fold the "
              "laundry clean the fridge mop the floor dust the shelves wash the car sweep the porch "
              "change the sheets empty the dishwasher feed the cat",
    "cities": "Lisbon Oslo Prague Vienna Dublin Krakow Seville Porto Bergen Ghent Lyon Turin "
              "Valencia Zagreb Riga Tallinn Bruges Salzburg Bologna Granada",
}
_INTROS = ["Things to buy this week:", "Packing list for the trip:", "Weekend to-do list:",
           "Places we want to visit:", "Here is the list:", "Checklist:"]


def _list_entries(rng: random.Random) -> list[str]:
    key = rng.choice(sorted(_LISTS))
    raw = _LISTS[key]
    if key == "chores":
        verbs = ("vacuum", "water", "take", "wipe", "fold", "clean", "mop", "dust", "wash", "sweep",
                 "change", "empty", "feed")
        toks, cur = raw.split(), []
        entries = []
        for t in toks:
            if t in verbs and cur:
                entries.append(" ".join(cur))
                cur = []
            cur.append(t)
        entries.append(" ".join(cur))
    else:
        entries = raw.split()
    rng.shuffle(entries)
    return entries


def build_listidx(tok, rng: random.Random, n: int) -> list[dict]:  # noqa: ANN001
    out = []
    for k in range(n):
        val = "early" if k % 2 == 0 else "late"
        lo, hi = BINS["bk_listidx"][val]
        idx = rng.randint(lo, hi)
        entries = _list_entries(rng)[:idx]
        if len(entries) < idx:
            continue
        text = rng.choice(_INTROS) + "\n" + "\n".join(f"- {e}" for e in entries)
        out.append({"id": f"list_{k}_i{idx}", "feature": "bk_listidx", "cluster": f"list{k}",
                    "value": val, "ids": tok(text, add_special_tokens=False)["input_ids"],
                    "text": text, "meta": {"index": idx}})
    return out


# ------------------------------------------------------------------ indentation level
def _py_block(rng: random.Random, depth: int, target: int, lines: list[str]) -> None:
    ind = "    " * depth
    names = ["item", "row", "node", "value", "entry", "record", "user", "path"]
    v = rng.choice(names)
    heads = [f"for {v} in {v}s:", f"if {v} is not None:", f"while {v}.pending():",
             f"with open({v}.path) as fh:", f"try:", f"if len({v}s) > {rng.randint(2, 50)}:"]
    stmts = [f"count += {rng.randint(1, 9)}", f"{v}s.append({v})", f"total = total + {v}.size",
             f"log.debug('processing %s', {v})", f"seen.add({v}.key)", f"result[{v}.id] = {v}",
             f"{v} = {v}.next", f"buffer.write(str({v}))"]
    for _ in range(rng.randint(1, 2)):
        lines.append(ind + rng.choice(stmts))
    if depth < target:
        h = rng.choice(heads)
        lines.append(ind + h)
        _py_block(rng, depth + 1, target, lines)
        if h == "try:":
            lines.append(ind + "except ValueError:")
            lines.append(ind + "    continue")


def build_indent(tok, rng: random.Random, n: int) -> list[dict]:  # noqa: ANN001
    out = []
    for k in range(n):
        val = "shallow" if k % 2 == 0 else "deep"
        lo, hi = BINS["bk_indent"][val]
        level = rng.randint(lo, hi)
        fn = rng.choice(["process", "collect", "summarize", "scan", "index", "flush", "merge"])
        lines = [f"def {fn}_{rng.choice(['items', 'rows', 'events', 'files'])}(data):",
                 "    count = 0", "    total = 0"]
        _py_block(rng, 1, 4, lines)  # build up to depth 4, then truncate at a line of `level`
        cand = [i for i, ln in enumerate(lines) if (len(ln) - len(ln.lstrip(" "))) // 4 == level
                and i >= 3 and not ln.rstrip().endswith(":")]
        if not cand:
            continue
        i = rng.choice(cand)
        ln = lines[i]
        body = ln.lstrip(" ")
        cut = len(ln) - len(body) + rng.randint(max(2, len(body) // 3), len(body))
        while cut < len(ln) and ln[cut - 1].isalnum():
            cut += 1
        text = "\n".join(lines[:i] + [ln[:cut]])
        out.append({"id": f"indent_{k}_l{level}", "feature": "bk_indent", "cluster": f"indent{k}",
                    "value": val, "ids": tok(text, add_special_tokens=False)["input_ids"],
                    "text": text, "meta": {"level": level}})
    return out


# ------------------------------------------------------------------ backups
def build_caps(tok, paras: list[str], rng: random.Random) -> list[dict]:  # noqa: ANN001
    out = []
    for k, p in enumerate(paras):
        words = p.split()[: rng.randint(18, 40)]
        text = " ".join(words)
        val = "uppercase" if k % 2 == 0 else "lowercase"
        if val == "uppercase":
            text = text.upper()
        out.append({"id": f"caps_p{k}", "feature": "bk_caps", "cluster": f"p{k}", "value": val,
                    "ids": tok(text, add_special_tokens=False)["input_ids"], "text": text,
                    "meta": {}})
    return out


_NOUNS = ["user", "order", "item", "record", "price", "score", "token", "node", "file", "event",
          "message", "account", "row", "value", "task", "path", "packet", "entry", "city", "image"]
_VERBS = ["load", "parse", "filter", "update", "compute", "build", "fetch", "merge", "count",
          "validate", "normalize", "render", "sort", "group", "encode", "clean"]


def _code_pair(rng: random.Random) -> tuple[str, str]:
    n, v = rng.choice(_NOUNS), rng.choice(_VERBS)
    fn = f"{v}_{n}s"
    fnj = v + n.capitalize() + "s"
    thr = rng.randint(2, 99)
    key = rng.choice(["id", "name", "total", "count", "status", "size"])
    kind = rng.randint(0, 2)
    if kind == 0:
        py = (f"def {fn}({n}s, limit={thr}):\n    result = []\n    for {n} in {n}s:\n"
              f"        if {n}['{key}'] > limit:\n            result.append({n})\n"
              f"    return result\n")
        js = (f"function {fnj}({n}s, limit = {thr}) {{\n  const result = [];\n"
              f"  for (const {n} of {n}s) {{\n    if ({n}.{key} > limit) {{\n"
              f"      result.push({n});\n    }}\n  }}\n  return result;\n}}\n")
    elif kind == 1:
        py = (f"class {n.capitalize()}Store:\n    def __init__(self):\n        self.{n}s = {{}}\n\n"
              f"    def {fn}(self, key, {n}):\n        if key in self.{n}s:\n"
              f"            return None\n        self.{n}s[key] = {n}\n        return {n}\n")
        js = (f"class {n.capitalize()}Store {{\n  constructor() {{\n    this.{n}s = {{}};\n  }}\n\n"
              f"  {fnj}(key, {n}) {{\n    if (key in this.{n}s) {{\n      return null;\n    }}\n"
              f"    this.{n}s[key] = {n};\n    return {n};\n  }}\n}}\n")
    else:
        py = (f"import json\n\ndef {fn}(path):\n    with open(path) as f:\n        data = json.load(f)\n"
              f"    total = sum(x['{key}'] for x in data)\n    print(f\"total: {{total}}\")\n"
              f"    return total / max(len(data), {thr})\n")
        js = (f"const fs = require('fs');\n\nfunction {fnj}(path) {{\n"
              f"  const data = JSON.parse(fs.readFileSync(path, 'utf8'));\n"
              f"  const total = data.reduce((s, x) => s + x.{key}, 0);\n"
              f"  console.log(`total: ${{total}}`);\n  return total / Math.max(data.length, {thr});\n}}\n")
    return py, js


def build_code(tok, rng: random.Random, n_pairs: int) -> list[dict]:  # noqa: ANN001
    out = []
    for k in range(n_pairs):
        py, js = _code_pair(rng)
        for val, src in (("Python", py), ("JavaScript", js)):
            cut = rng.randint(int(len(src) * 0.45), int(len(src) * 0.85))
            while cut < len(src) and src[cut - 1].isalnum():
                cut += 1  # end at a token boundary-ish (after a word)
            text = src[:cut].rstrip()
            out.append({"id": f"code_{k}_{val[:2].lower()}", "feature": "bk_code",
                        "cluster": f"code{k}", "value": val,
                        "ids": tok(text, add_special_tokens=False)["input_ids"], "text": text,
                        "meta": {}})
    return out


def split_code(items: list[dict], rng: random.Random) -> list[dict]:
    """Content pairs stay in one split (both languages of a pair together)."""
    pairs = sorted({it["cluster"] for it in items})
    rng.shuffle(pairs)
    mine = set(pairs[:N_MINE_PER_VALUE])
    sel = set(pairs[N_MINE_PER_VALUE: N_MINE_PER_VALUE + N_SELECT_PER_VALUE])
    return [{**it, "split": "mine" if it["cluster"] in mine else "select"}
            for it in items if it["cluster"] in mine | sel]


# ================================================================== round-4 features (Fable)
def build_col_break(tok, s1, paras: list[str], rng: random.Random) -> list[dict]:  # noqa: ANN001
    """Break DECISION with the count matched: col in BREAK_COL_RANGE for both values; 'soon' =
    W - col <= BREAK_SOON_MAX (the break comes within <= 6 chars since textwrap breaks at <= W),
    'far' = W - col >= BREAK_FAR_MIN. 02's build_wrap on fresh paragraphs, W redrawn per value."""
    from feature_spec import BREAK_COL_RANGE, BREAK_FAR_MIN, BREAK_SOON_MAX

    s1._wikitext_paragraphs = lambda *_a, **_k: paras
    wrap = s1.build_wrap(tok, n=len(paras))
    out = []
    for k, it in enumerate(wrap):
        val = "soon" if k % 2 == 0 else "far"
        W = it["width"]
        ok = (lambda c: W - c <= BREAK_SOON_MAX) if val == "soon" else (lambda c: W - c >= BREAK_FAR_MIN)
        cand = [j for j in range(8, len(it["ids"]))
                if not it["has_nl"][j] and _in_bin(it["col"][j], BREAK_COL_RANGE) and ok(it["col"][j])
                and not it["next_nl"][j]]  # never read AT the break token itself
        if not cand:
            continue
        j = rng.choice(cand)
        out.append({"id": f"colbrk_p{k}_t{j}", "feature": "nj1_col_break", "cluster": f"p{k}",
                    "value": val, "ids": it["ids"][: j + 1], "text": it["text"][: it["offsets"][j][1]],
                    "meta": {"col": it["col"][j], "width": W, "remaining_to_width": W - it["col"][j],
                             "pos": j}})
    return out


def build_listnum(tok, rng: random.Random, n: int) -> list[dict]:  # noqa: ANN001
    out = []
    for k in range(n):
        val = "early" if k % 2 == 0 else "late"
        lo, hi = BINS["ij_listnum"][val]
        idx = rng.randint(lo, hi)
        entries = _list_entries(rng)[:idx]
        if len(entries) < idx:
            continue
        text = rng.choice(_INTROS) + "\n" + "\n".join(f"Item {i + 1}: {e}" for i, e in enumerate(entries))
        out.append({"id": f"listnum_{k}_i{idx}", "feature": "ij_listnum", "cluster": f"listnum{k}",
                    "value": val, "ids": tok(text, add_special_tokens=False)["input_ids"],
                    "text": text, "meta": {"index": idx}})
    return out


def build_quote(tok, paras: list[str], rng: random.Random) -> list[dict]:  # noqa: ANN001
    import re

    out = []
    for k, p in enumerate(paras):
        p = p[:1500]
        spans = [m.span() for m in re.finditer(r'"[^"]{25,400}"', p)]
        if not spans or p.count('"') % 2:
            continue
        val = "inside" if len(out) % 2 == 0 else "outside"
        cand = []
        for m in re.finditer(r"[A-Za-z]+", p):
            e = m.end()
            if e >= len(p) or p[e].isalnum():
                continue
            inside = [sp for sp in spans if sp[0] < m.start() and e < sp[1] - 1]
            if val == "inside" and inside:
                if len(p[inside[0][0] + 1: e].split()) >= 3:
                    cand.append(e)
            elif val == "outside" and not inside and p[:e].count('"') % 2 == 0:
                closes = [sp[1] for sp in spans if sp[1] <= m.start()]
                if closes and len(p[max(closes): e].split()) >= 4:
                    cand.append(e)
        if not cand:
            continue
        e = rng.choice(cand)
        out.append({"id": f"quote_p{k}_c{e}", "feature": "bk_quote", "cluster": f"p{k}", "value": val,
                    "ids": tok(p[:e], add_special_tokens=False)["input_ids"], "text": p[:e], "meta": {}})
    return out


_FR_ART = {"le": ("masculine", "singular"), "un": ("masculine", "singular"),
           "ce": ("masculine", "singular"), "du": ("masculine", "singular"),
           "la": ("feminine", "singular"), "une": ("feminine", "singular"),
           "cette": ("feminine", "singular"), "les": (None, "plural"), "des": (None, "plural")}
_DE_ART = {"einen": "masculine", "eine": "feminine"}


def build_agreement(tok, rng: random.Random) -> tuple[list[dict], list[dict]]:  # noqa: ANN001
    """papluca TRAIN split (lang uses test): cut right after an article; value = gender (FR/DE) or
    number (FR) of the upcoming noun. Each text feeds at most one item, gender and number disjoint."""
    import re

    from datasets import load_dataset

    ds = load_dataset("papluca/language-identification", split="train")
    gen, num = [], []
    for r, ex in enumerate(ds):
        lg, t = ex["labels"], ex["text"]
        if lg not in ("fr", "de") or len(t) < 60:
            continue
        arts = []
        for m in re.finditer(r"(?<![\w'’])([A-Za-zäöüßéèàç]+) (?=[a-zäöüßéèêàç]{3,})", t):
            w = m.group(1)
            if m.start() < 15:
                continue
            if lg == "fr" and w in _FR_ART:
                arts.append((m.start() + len(w), _FR_ART[w]))
            elif lg == "de" and w in _DE_ART:
                arts.append((m.start() + len(w), (_DE_ART[w], None)))
        if not arts:
            continue
        want_num = lg == "fr" and r % 2 == 0
        if want_num:
            want = "plural" if len(num) % 2 == 0 else "singular"
            c = [e for e, (_, n) in arts if n == want]
            if c:
                e = rng.choice(c)
                num.append({"id": f"agrnum_train{r}", "feature": "agr_number", "cluster": f"train{r}",
                            "value": want, "ids": tok(t[:e], add_special_tokens=False)["input_ids"],
                            "text": t[:e], "meta": {"lang": lg, "article": t[:e].split()[-1]}})
        else:
            want = "masculine" if len(gen) % 2 == 0 else "feminine"
            c = [e for e, (g, _) in arts if g == want]
            if c:
                e = rng.choice(c)
                gen.append({"id": f"agrgen_train{r}", "feature": "agr_gender", "cluster": f"train{r}",
                            "value": want, "ids": tok(t[:e], add_special_tokens=False)["input_ids"],
                            "text": t[:e], "meta": {"lang": lg, "article": t[:e].split()[-1]}})
        if len(gen) >= 400 and len(num) >= 400:
            break
    return gen, num
