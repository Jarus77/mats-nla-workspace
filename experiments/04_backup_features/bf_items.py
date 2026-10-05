"""Item builders for 04_backup_features (pure python + a HF tokenizer; runs in the data container).

Every item: {"id", "group", "dataset", "ids", "text", "feats": {fname: {"pos": [...], "y": [...]}},
"eval": {fname: [positions]}}.  Positions index ``ids`` (raw text, no chat template, no BOS). A
position t carries the model's state AFTER reading token t (next-token prediction at t).
Read positions are balanced within item (<= ``per_class`` per class) and items need both classes,
so a probe cannot win by item identity alone. ``group`` = clustering / CV unit.
"""

from __future__ import annotations

import bisect
import random
import re
from typing import Any

SENT_START = re.compile(r"[.!?][\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
MIN_POS = 4


def _enc(tok: Any, text: str, max_tok: int) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"][:max_tok], enc["offset_mapping"][:max_tok]
    return ids, offs, [text[s:e] for s, e in offs]


def _first_char(s: int, st: str) -> int:
    return s + (len(st) - len(st.lstrip()))


def _balanced(cands: list[tuple[int, int]], per_class: int, rng: random.Random) -> dict | None:
    """cands = [(pos, y)] -> {"pos", "y"} with <= per_class per class, both classes required."""
    by = {0: [p for p, y in cands if y == 0], 1: [p for p, y in cands if y == 1]}
    if not by[0] or not by[1]:
        return None
    sel = []
    for y, ps in by.items():
        ps = sorted(set(ps))
        pick = rng.sample(ps, min(per_class, len(ps)))
        sel.extend((p, y) for p in pick)
    sel.sort()
    return {"pos": [p for p, _ in sel], "y": [y for _, y in sel]}


def _word_starts(text: str, offs: list[tuple[int, int]], strs: list[str]) -> list[tuple[int, int]]:
    out = []
    for i, ((s, _e), st) in enumerate(zip(offs, strs)):
        core = st.lstrip()
        if not core or not core[0].isalpha():
            continue
        c = _first_char(s, st)
        if c > 0 and (text[c - 1].isalnum() or text[c - 1] in "'-_"):
            continue
        out.append((i, c))
    return out


# ----------------------------------------------------------------------------- prose


def build_prose(tok: Any, texts: list[str], seed: int = 0, max_tok: int = 160, per_class: int = 4) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for k, text in enumerate(texts):
        ids, offs, strs = _enc(tok, text, max_tok)
        starts = [0] + [m.end() for m in SENT_START.finditer(text)]
        ws = _word_starts(text, offs, strs)
        widx, sidx, last_sent, cnt = [], [], -1, 0
        for _i, c in ws:
            s_ = bisect.bisect_right(starts, c) - 1
            cnt = 0 if s_ != last_sent else cnt + 1
            last_sent = s_
            widx.append(cnt)
            sidx.append(s_)
        cap_mid, cap_all, spos = [], [], []
        for (i, c), wi, si in zip(ws, widx, sidx):
            if i - 1 >= MIN_POS:
                y = int(text[c].isupper())
                cap_all.append((i - 1, y))
                if wi > 0:
                    cap_mid.append((i - 1, y))
            if i >= MIN_POS and si >= 1:
                if 1 <= wi <= 4:
                    spos.append((i, 0))
                elif wi >= 15:
                    spos.append((i, 1))
        feats = {}
        for name, cands in (("cap_next", cap_mid), ("cap_next_all", cap_all), ("sent_pos", spos)):
            b = _balanced(cands, per_class, rng)
            if b is not None:
                feats[name] = b
        if feats:
            items.append({"id": f"prose{k}", "group": k, "dataset": "prose", "ids": ids, "text": text,
                          "feats": feats, "eval": {n: v["pos"] for n, v in feats.items()}})
    return items


# ----------------------------------------------------------------------------- code language


def build_code_pair(tok: Any, recs: list[dict], seed: int = 0, max_tok: int = 400, per_item: int = 8) -> list[dict]:
    """recs: {"task": int, "lang": "python"|"js", "prompt", "solution"}; group = task (pairs never split)."""
    rng = random.Random(seed)
    items = []
    for k, r in enumerate(recs):
        text = r["prompt"] + r["solution"]
        ids, offs, strs = _enc(tok, text, max_tok)
        cut = len(r["prompt"])
        cands = [i for i, ((s, _e), st) in enumerate(zip(offs, strs))
                 if i >= 8 and s >= cut and st.strip()]
        if len(cands) < 4:
            continue
        pos = sorted(rng.sample(cands, min(per_item, len(cands))))
        y = int(r["lang"] == "python")
        feats = {"code_lang": {"pos": pos, "y": [y] * len(pos)}}
        items.append({"id": f"code{k}", "group": int(r["task"]), "dataset": "code_pair", "ids": ids,
                      "text": text, "lang": r["lang"], "feats": feats, "eval": {"code_lang": pos}})
    return items


# ----------------------------------------------------------------------------- indentation


def _lines(text: str) -> list[tuple[int, int, int]]:
    """[(line_start, line_end, indent_spaces or -1 for blank)]."""
    out, s = [], 0
    for ln in text.split("\n"):
        ind = len(ln) - len(ln.lstrip(" ")) if ln.strip() else -1
        out.append((s, s + len(ln), ind))
        s += len(ln) + 1
    return out


def build_code_py(tok: Any, codes: list[str], seed: int = 0, max_tok: int = 320, per_class: int = 4,
                  max_eval: int = 12) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for k, code in enumerate(codes):
        if "\t" in code:
            continue
        lines = _lines(code)
        inds = [ind for _s, _e, ind in lines if ind > 0]
        if not inds or min(inds) != 4 or any(i % 4 for i in inds):
            continue
        starts = [s for s, _e, _i in lines]
        ids, offs, strs = _enc(tok, code, max_tok)
        first_code_tok: dict[int, int] = {}
        cands, evals = [], []
        for i, ((s, e), st) in enumerate(zip(offs, strs)):
            if "\n" in st:
                li = bisect.bisect_right(starts, max(e - 1, 0)) - 1
                nxt = li + 1
                if i >= MIN_POS and i < len(ids) - 1 and nxt < len(lines) and lines[nxt][2] >= 0:
                    evals.append(i)
                continue
            if not st.strip():
                continue
            c0 = _first_char(s, st)
            li = bisect.bisect_right(starts, c0) - 1
            if li not in first_code_tok:
                first_code_tok[li] = i
                continue
            level = lines[li][2] // 4
            if i >= MIN_POS and level in (1, 2):
                cands.append((i, int(level == 2)))
        b = _balanced(cands, per_class, rng)
        if b is None or not evals:
            continue
        ev = sorted(rng.sample(evals, min(max_eval, len(evals))))
        items.append({"id": f"py{k}", "group": k, "dataset": "code_py", "ids": ids, "text": code,
                      "feats": {"indent": b}, "eval": {"indent": ev}})
    return items


# ----------------------------------------------------------------------------- lists


def split_sentences(text: str) -> list[str]:
    cuts = [0] + [m.end() for m in SENT_START.finditer(text)] + [len(text)]
    return [text[a:b].strip() for a, b in zip(cuts[:-1], cuts[1:]) if text[a:b].strip()]


def build_lists(tok: Any, paras: list[str], numbered: bool, seed: int = 0, max_items: int = 9,
                min_items: int = 6, per_class: int = 4, max_tok: int = 512) -> list[dict]:
    """Each paragraph's sentences rendered as a numbered ("Item N: s") or bulleted ("- s") list.
    Feature: index of the current item (early 2-3 = 0, late 5-7 = 1) at word tokens >= 4 tokens into
    the item. Eval (numbered): the token right before each item-index digit of items >= 2 (next
    token = the next index); (bulleted): item-ending newline tokens."""
    name = "list_num" if numbered else "list_bul"
    rng = random.Random(seed + (1 if numbered else 2))
    items = []
    for k, p in enumerate(paras):
        sents = [s for s in split_sentences(p) if len(s) <= 300][:max_items]
        if len(sents) < min_items:
            continue
        lines = [(f"Item {j + 1}: " if numbered else "- ") + s for j, s in enumerate(sents)]
        text = "\n".join(lines) + "\n"
        ids, offs, strs = _enc(tok, text, max_tok)
        if offs[-1][1] < len(text) - 1:
            continue
        starts, c = [], 0
        for ln in lines:
            starts.append(c)
            c += len(ln) + 1
        item_tok: dict[int, int] = {}
        cands, evals = [], []
        for i, ((s, e), st) in enumerate(zip(offs, strs)):
            if "\n" in st:
                li = bisect.bisect_right(starts, max(e - 1, 0)) - 1
                if i >= MIN_POS and li + 1 < len(lines) and li >= 0:
                    evals.append(i)
                continue
            li = bisect.bisect_right(starts, _first_char(s, st)) - 1
            item_tok[li] = item_tok.get(li, 0) + 1
            if item_tok[li] < (6 if numbered else 4) or not st.strip().isalpha() or i < MIN_POS:
                continue
            idx = li + 1
            if idx in (2, 3):
                cands.append((i, 0))
            elif idx in (5, 6, 7):
                cands.append((i, 1))
        if numbered:
            digit_at = {starts[j] + 5 for j in range(1, len(lines))}
            evals = [i - 1 for i, (s_, _e) in enumerate(offs) if s_ in digit_at and i - 1 >= MIN_POS]
        b = _balanced(cands, per_class, rng)
        if b is None or not evals:
            continue
        feats, ev = {name: b}, {name: evals}
        if numbered:  # list_hdr: read at the token before item index N (next token = the digit N)
            hdr = [(t, 0 if j + 2 in (2, 3) else 1) for j, t in enumerate(evals) if j + 2 in (2, 3, 5, 6, 7)]
            bh = _balanced(hdr, per_class, random.Random(seed * 7 + k))
            if bh is not None:
                feats["list_hdr"], ev["list_hdr"] = bh, bh["pos"]
        items.append({"id": f"{name}{k}", "group": k, "dataset": name, "ids": ids, "text": text,
                      "feats": feats, "eval": ev})
    return items


# ----------------------------------------------------------------------------- dialogue (in-quote)


def build_dialogue(tok: Any, stories: list[str], seed: int = 0, max_tok: int = 256, per_class: int = 4) -> list[dict]:
    """in_quote: word token inside an open double-quote span (1) vs narration (0); first/last word
    of each span and quote-bearing tokens excluded. TinyStories (curly quotes normalised)."""
    rng = random.Random(seed)
    items = []
    for k, raw in enumerate(stories):
        text = raw.replace("\u201c", '"').replace("\u201d", '"').strip()
        qs = [m.start() for m in re.finditer('"', text)]
        if len(qs) < 4 or len(qs) % 2:
            continue
        ids, offs, strs = _enc(tok, text, max_tok)
        end_char = offs[-1][1]
        span_words: dict[int, list[int]] = {}
        cands = []
        for i, ((s, _e), st) in enumerate(zip(offs, strs)):
            if i < MIN_POS or not st.strip().isalpha():
                continue
            c = _first_char(s, st)
            nq = bisect.bisect_left(qs, c)
            if nq % 2 == 1:
                if qs[nq] < end_char:  # span closes inside the window
                    span_words.setdefault(nq // 2, []).append(i)
            else:
                cands.append((i, 0))
        for ws in span_words.values():
            cands.extend((i, 1) for i in ws[1:-1])
        b = _balanced(cands, per_class, rng)
        if b is None:
            continue
        items.append({"id": f"dlg{k}", "group": k, "dataset": "dialogue", "ids": ids, "text": text,
                      "feats": {"in_quote": b}, "eval": {"in_quote": b["pos"]}})
    return items


# ----------------------------------------------------------------------------- French gender agreement

NEUTRAL_DET = {"les", "des", "ces", "ses", "mes", "tes", "leurs", "nos", "vos", "aux"}


def build_french(tok: Any, nlp: Any, paras: list[str], seed: int = 0, max_tok: int = 200,
                 per_class: int = 4) -> tuple[list[dict], dict]:
    """gender_next: grammatical gender (Fem = 1, Masc = 0) of the UPCOMING head noun, read at a
    gender-NEUTRAL plural determiner (les/des/ces/ses/...; the noun is not yet in context).
    Also returns a corpus lexicon {token_id: {"Fem": n, "Masc": n}} of word-initial noun/adj tokens."""
    rng = random.Random(seed)
    items, lex = [], {}
    for k, (text, doc) in enumerate(zip(paras, nlp.pipe(paras, batch_size=64))):
        ids, offs, strs = _enc(tok, text, max_tok)
        end_char = offs[-1][1]
        starts = [s for s, _e in offs]
        cands = []
        for t in doc:
            if t.idx >= end_char:
                break
            if t.pos_ in ("NOUN", "ADJ"):
                gnd = t.morph.get("Gender")
                ti = bisect.bisect_right(starts, t.idx) - 1
                if len(gnd) == 1 and ti >= 0 and _first_char(offs[ti][0], strs[ti]) == t.idx:
                    lex.setdefault(ids[ti], {"Fem": 0, "Masc": 0})
                    if gnd[0] in ("Fem", "Masc"):
                        lex[ids[ti]][gnd[0]] += 1
            if t.lower_ in NEUTRAL_DET and t.dep_ in ("det", "case") and t.head.pos_ == "NOUN":
                head = t.head
                gnd = head.morph.get("Gender")
                if len(gnd) != 1 or gnd[0] not in ("Fem", "Masc") or head.i <= t.i:
                    continue
                last = t.idx + len(t.text) - 1
                ti = bisect.bisect_right(starts, last) - 1
                if ti >= MIN_POS and offs[ti][1] == t.idx + len(t.text) and head.idx >= offs[ti][1]:
                    cands.append((ti, int(gnd[0] == "Fem")))
        b = _balanced(cands, per_class, rng)
        if b is None:
            continue
        items.append({"id": f"fr{k}", "group": k, "dataset": "french", "ids": ids, "text": text,
                      "feats": {"gender_next": b}, "eval": {"gender_next": b["pos"]}})
    return items, lex


# ----------------------------------------------------------------------------- position matching


def position_match(items: list[dict], fname: str, bin_size: int = 16, seed: int = 0) -> list[dict]:
    """Subsample read positions so both classes have the same absolute-position distribution
    (equal counts per ``bin_size``-token bin). Returns NEW item dicts (inputs untouched)."""
    rng = random.Random(seed)
    rows = [(k, p, y) for k, it in enumerate(items) if fname in it["feats"]
            for p, y in zip(it["feats"][fname]["pos"], it["feats"][fname]["y"])]
    bins: dict[int, dict[int, list]] = {}
    for r in rows:
        bins.setdefault(r[1] // bin_size, {0: [], 1: []})[r[2]].append(r)
    keep = set()
    for b in bins.values():
        n = min(len(b[0]), len(b[1]))
        for y in (0, 1):
            keep.update(rng.sample(b[y], n))
    out = []
    for k, it in enumerate(items):
        if fname not in it["feats"]:
            out.append(it)
            continue
        sel = [(p, y) for p, y in zip(it["feats"][fname]["pos"], it["feats"][fname]["y"]) if (k, p, y) in keep]
        feats = {n: v for n, v in it["feats"].items() if n != fname}
        ev = {n: v for n, v in it["eval"].items() if n != fname}
        if sel:
            feats[fname] = {"pos": [p for p, _ in sel], "y": [y for _, y in sel]}
            ev[fname] = it["eval"][fname] if it["eval"][fname] != it["feats"][fname]["pos"] else feats[fname]["pos"]
        if feats:
            out.append({**it, "feats": feats, "eval": ev})
    return out
