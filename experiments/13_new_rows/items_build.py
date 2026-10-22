"""Pure text generators for the exp 13 rows. Each returns items WITHOUT token ids:
{id, feature, cluster, value, text, meta}; ``tokenise`` adds ``ids`` (prefix tokenised so the read
position is the last token). Paragraph-based rows take cleaned wikitext paragraphs; synthetic rows
take only an rng. No model, no tokenizer, so every generator is unit-tested locally."""
from __future__ import annotations

import random
import re

from feature_spec import BINS, QA_WORDS_IN, SENT_WORDS_IN, TABLE_ROWS, WORDSLEFT_LIMIT

SEED = 1313
_SENT_SPLIT = re.compile(r"[.!?][\"')]? +(?=[A-Z])")


def _in(x: int, lo_hi: tuple[int, int]) -> bool:
    return lo_hi[0] <= x <= lo_hi[1]


def word_ends(text: str, lo: int = 0, hi: int | None = None) -> list[int]:
    """Char offsets just after a word that is followed by a non-alphanumeric char (or text end)."""
    hi = len(text) if hi is None else hi
    return [m.end() for m in re.finditer(r"[A-Za-z]+", text[:hi])
            if m.end() >= lo and (m.end() == len(text) or not text[m.end()].isalnum())]


def n_words(s: str) -> int:
    return len(re.findall(r"\S+", s))


def tokenise(tok, items: list[dict]) -> list[dict]:  # noqa: ANN001
    return [{**it, "ids": tok(it["text"], add_special_tokens=False)["input_ids"]} for it in items]


def assign_splits(items: list[dict], rng: random.Random, n_mine: int, n_sel: int) -> list[dict]:
    """Per value: shuffle, first n_mine -> mine, next n_sel -> select (exp 07's rule)."""
    out, by_val = [], {}
    for it in items:
        by_val.setdefault(it["value"], []).append(it)
    for _, lst in sorted(by_val.items()):
        rng.shuffle(lst)
        out += [{**it, "split": "mine" if k < n_mine else "select"} for k, it in enumerate(lst[: n_mine + n_sel])]
    return out


# ------------------------------------------------------------------ tb_sentcount (paragraphs)
def build_sentcount(paras: list[str], rng: random.Random) -> list[dict]:
    out = []
    for k, p in enumerate(paras):
        p = p[:1800]
        starts = [0] + [m.end() for m in _SENT_SPLIT.finditer(p)]
        if len(starts) < BINS["tb_sentcount"]["late"][0]:
            continue
        val = "early" if len(out) % 2 == 0 else "late"
        cand = []
        for e in word_ends(p):
            si = max(i for i, s in enumerate(starts) if s <= e)  # 0-based sentence index
            w = n_words(p[starts[si]:e])
            if _in(si + 1, BINS["tb_sentcount"][val]) and _in(w, SENT_WORDS_IN):
                cand.append((e, si + 1, w))
        if not cand:
            continue
        e, si, w = rng.choice(cand)
        out.append({"id": f"sent_p{k}_c{e}", "feature": "tb_sentcount", "cluster": f"p{k}", "value": val,
                    "text": p[:e], "meta": {"sentence": si, "words_in": w}})
    return out


# ------------------------------------------------------------------ tb_paren (paragraphs)
def build_paren(paras: list[str], rng: random.Random) -> list[dict]:
    out = []
    for k, p in enumerate(paras):
        p = p[:1500]
        spans = [m.span() for m in re.finditer(r"\([^()]{25,300}\)", p)]
        if not spans or p.count("(") != p.count(")"):
            continue
        val = "inside" if len(out) % 2 == 0 else "outside"
        cand = []
        for e in word_ends(p):
            inside = [sp for sp in spans if sp[0] < e <= sp[1] - 1]
            if val == "inside" and inside and n_words(p[inside[0][0] + 1:e]) >= 3:
                cand.append(e)
            elif val == "outside" and not inside and p[:e].count("(") == p[:e].count(")"):
                closes = [sp[1] for sp in spans if sp[1] <= e]
                if closes and n_words(p[max(closes):e]) >= 4:
                    cand.append(e)
        if not cand:
            continue
        e = rng.choice(cand)
        out.append({"id": f"paren_p{k}_c{e}", "feature": "tb_paren", "cluster": f"p{k}", "value": val,
                    "text": p[:e], "meta": {}})
    return out


# ------------------------------------------------------------------ tb_wordsleft (paragraphs)
_WL_TOPICS = ["the history of the place", "this subject", "the events described", "the topic below",
              "what happened", "the person described"]


def build_wordsleft(paras: list[str], rng: random.Random) -> list[dict]:
    out = []
    for k, p in enumerate(paras):
        p = p[:1500]
        val = "many" if len(out) % 2 == 0 else "few"
        n_limit = rng.randint(*WORDSLEFT_LIMIT)
        head = f"Write exactly {n_limit} words about {rng.choice(_WL_TOPICS)}.\n\n"
        cand = [(e, n_limit - n_words(p[:e])) for e in word_ends(p)]
        cand = [(e, left) for e, left in cand if _in(left, BINS["tb_wordsleft"][val])]
        if not cand:
            continue
        e, left = rng.choice(cand)
        out.append({"id": f"wl_p{k}_c{e}", "feature": "tb_wordsleft", "cluster": f"p{k}", "value": val,
                    "text": head + p[:e], "meta": {"limit": n_limit, "words_so_far": n_limit - left, "left": left}})
    return out


# ------------------------------------------------------------------ fmt_qa (paragraphs)
_QUESTIONS = ["Can you tell me about this?", "What is known about this topic?", "Could you summarise it?",
              "What happened here?", "Can you explain the background?"]


def build_qa(paras: list[str], rng: random.Random) -> list[dict]:
    out = []
    for k, p in enumerate(paras):
        p = p[:1200]
        val = "qa" if k % 2 == 0 else "prose"
        cand = [e for e in word_ends(p) if _in(n_words(p[:e]), QA_WORDS_IN)]
        if not cand:
            continue
        e = rng.choice(cand)
        body = p[:e]
        text = f"Question: {rng.choice(_QUESTIONS)}\nAnswer: {body}" if val == "qa" else body
        out.append({"id": f"qa_p{k}_c{e}", "feature": "fmt_qa", "cluster": f"p{k}", "value": val,
                    "text": text, "meta": {"words": n_words(body)}})
    return out


# ------------------------------------------------------------------ tb_tablecol (synthetic)
_COLS = {"Name": ["Alice", "Bruno", "Chen", "Dana", "Elif", "Farid", "Greta", "Hugo", "Iris", "Jonas"],
         "City": ["Lisbon", "Oslo", "Prague", "Vienna", "Dublin", "Seville", "Porto", "Bergen", "Lyon", "Turin"],
         "Team": ["Falcons", "Otters", "Herons", "Badgers", "Lynxes", "Condors", "Ravens", "Bison"],
         "Role": ["analyst", "engineer", "designer", "manager", "intern", "auditor", "writer"],
         "Status": ["active", "pending", "retired", "onboarding", "paused"],
         "Office": ["north", "south", "east", "west", "central", "remote"],
         "Level": ["junior", "senior", "lead", "principal", "staff"],
         "Language": ["Portuguese", "Norwegian", "Czech", "German", "Irish", "Spanish", "French", "Italian"]}


def build_tablecol(rng: random.Random, n: int) -> list[dict]:
    out = []
    for k in range(n):
        val = "first" if k % 2 == 0 else "late"
        n_cols = rng.randint(5, 6)
        heads = rng.sample(sorted(_COLS), n_cols)
        n_rows = rng.randint(*TABLE_ROWS)
        rows = [[rng.choice(_COLS[h]) for h in heads] for _ in range(n_rows)]
        col = rng.randint(*BINS["tb_tablecol"][val])
        lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * n_cols]
        lines += ["| " + " | ".join(r) + " |" for r in rows[:-1]]
        last = "| " + " | ".join(rows[-1][:col])  # cut right after the read cell's word
        text = "\n".join(lines + [last])
        out.append({"id": f"tbl_{k}_c{col}", "feature": "tb_tablecol", "cluster": f"tbl{k}", "value": val,
                    "text": text, "meta": {"col": col, "row": n_rows, "n_cols": n_cols}})
    return out


# ------------------------------------------------------------------ tb_codeline (synthetic)
_VARS = ["item", "row", "node", "value", "entry", "record", "user", "path", "batch", "line"]


def _flat_stmts(rng: random.Random, v: str, n: int) -> list[str]:
    pool = [f"count += {rng.randint(1, 9)}", f"{v}s.append({v})", f"total = total + {v}.size",
            f"log.debug('processing %s', {v})", f"seen.add({v}.key)", f"result[{v}.id] = {v}",
            f"{v} = {v}.next", f"buffer.write(str({v}))", f"cache[{v}.key] = {v}.value",
            f"index = index + len({v}s)", f"last = {v}", f"names.append({v}.name)",
            f"errors.extend({v}.errors)", f"score = score * {rng.randint(2, 5)}"]
    return [rng.choice(pool) for _ in range(n)]


def build_codeline(rng: random.Random, n: int) -> list[dict]:
    out = []
    for k in range(n):
        val = "early" if k % 2 == 0 else "late"
        line = rng.randint(*BINS["tb_codeline"][val])
        v = rng.choice(_VARS)
        fn = rng.choice(["process", "collect", "summarize", "scan", "index", "flush", "merge"])
        body = _flat_stmts(rng, v, 13)
        ln = body[line - 1]
        ends = [m.end() for m in re.finditer(r"[A-Za-z0-9]+", ln) if m.end() >= len(ln) // 3]
        cut = rng.choice(ends)
        lines = [f"def {fn}_{v}s(data):"] + ["    " + s for s in body[:line - 1]] + ["    " + ln[:cut]]
        text = "\n".join(lines)
        out.append({"id": f"code_{k}_l{line}", "feature": "tb_codeline", "cluster": f"code{k}", "value": val,
                    "text": text, "meta": {"line": line}})
    return out


# ------------------------------------------------------------------ fmt_listdepth (synthetic)
_TREE = {"Travel": {"Europe": {"Portugal": ["Lisbon old town", "Porto wine cellars", "Algarve beaches"],
                               "Norway": ["Bergen harbour", "fjord cruise", "northern lights"]},
                    "Asia": {"Japan": ["Kyoto temples", "Tokyo markets", "Osaka food"],
                             "Vietnam": ["Hanoi streets", "Ha Long bay", "Hoi An lanterns"]}},
         "Projects": {"Website": {"Frontend": ["fix the navbar", "add dark mode", "compress images"],
                                  "Backend": ["rotate the keys", "add caching", "write the migration"]},
                      "Garden": {"Spring": ["plant tomatoes", "fix the fence", "order compost"],
                                 "Autumn": ["rake the leaves", "store the tools", "prune the roses"]}},
         "Study": {"Maths": {"Algebra": ["group actions", "field extensions", "matrix rank"],
                             "Analysis": ["uniform convergence", "measure zero", "Fourier series"]},
                   "History": {"Ancient": ["Roman roads", "Greek city states", "Persian wars"],
                               "Modern": ["industrial revolution", "cold war", "decolonisation"]}}}


def build_listdepth(rng: random.Random, n: int) -> list[dict]:
    out = []
    for k in range(n):
        val = "shallow" if k % 2 == 0 else "deep"
        depth = BINS["fmt_listdepth"][val][0]
        top = rng.choice(sorted(_TREE))
        lines, target = [f"{top} notes:"], None
        for l1 in sorted(_TREE[top]):
            lines.append(f"- {l1}")
            for l2 in sorted(_TREE[top][l1]):
                lines.append(f"  - {l2}")
                for leaf in _TREE[top][l1][l2]:
                    lines.append(f"    - {leaf}")
        cand = [i for i, ln in enumerate(lines) if i > 1 and (len(ln) - len(ln.lstrip(" "))) // 2 + 1 == depth]
        i = rng.choice(cand[1:] if len(cand) > 1 else cand)  # not the very first item of that depth
        ln = lines[i]
        prefix = ln[: len(ln) - len(ln.lstrip(" ")) + 2]  # indentation + "- "
        words = ln[len(prefix):].split(" ")
        cut = len(prefix) + len(" ".join(words[: rng.randint(1, len(words))]))  # end of a word in the item
        text = "\n".join(lines[:i] + [ln[:cut]])
        target = depth
        out.append({"id": f"ld_{k}_d{target}", "feature": "fmt_listdepth", "cluster": f"ld{k}", "value": val,
                    "text": text, "meta": {"depth": target, "line": i}})
    return out
