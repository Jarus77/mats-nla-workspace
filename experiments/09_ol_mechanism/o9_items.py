"""09 item builders (Modal build container: network + spaCy). DECLARED PILOT SPLIT -- every id is recorded
in pilot_split.json and EXCLUDED from the main analysis.

Sets (all fresh):
  col09   80 hard-wrapped wikitext paragraphs (08 pilot recipe, pilot_items._wrap_item), pool seed 9090.
  sel09   60 prose items (random read token in [8, 120)), pool seed 9292 -- ONLY for choosing the random
          directions (model run), never scored by OL.
  tense09 80 items (40 Past + 40 Pres), 08 calib_items.build_tense recipe, pool seed 9191.
  codeNL  80 Python snippets from codeparrot/codeparrot-clean-valid (a source no earlier experiment used;
          04 used MBPP + HumanEval, 07 synthetic code). Read token = the "\\n" token ending a line whose next
          line is indented 4 or 8 spaces (balanced 40/40 on the next line's indent).
  codeIN  80 snippets (other files): read token = the last content token of a line indented 4 or 8 spaces
          (balanced on the current level 1/2), whose next token in the file is exactly "\\n" and whose next
          line is indented 4 or 8.
  codeTR  400 codeNL-style items (other files) -> the 'nextind' mean-difference direction (8 vs 4).
  One item per code file; files shuffled with seed 9393.

Wikitext exclusion: a paragraph is dropped if any 6-word shingle (lower-cased, whitespace-normalised) equals
the first 6 words of any line of any text used before (02 wrap/prose; 04 prose/list/dialogue pickles; 07
items_base/items_extra/nj3_pool; 08 pilot items, calibration items, round-5 tense + digit lists), or if its
sha1 matches an 08 passage. Sets are built in order col09 -> sel09 -> tense09, each excluding the previous.
"""
from __future__ import annotations

import gzip
import json
import pickle
import random
import re
import sys

sys.path.insert(0, "/root/ext08")
import calib_items as CI  # noqa: E402
import pilot_items as PI  # noqa: E402

COL_SEED, TENSE_SEED, SEL_SEED, CODE_SEED = 9090, 9191, 9292, 9393
N_COL, N_TENSE_PER, N_SEL = 80, 40, 60
N_CODE_EVAL, N_CODE_TRAIN = 80, 400
SH = 6
MAX_TOK = 360


# ----------------------------------------------------------------------------- exclusion
def _texts(obj) -> list[str]:  # noqa: ANN001
    out: list[str] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str):
            out.append(obj["text"])
        for v in obj.values():
            if isinstance(v, (dict, list)):
                out += _texts(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _texts(v)
    return out


def _line_heads(text: str) -> set[str]:
    heads = set()
    for ln in text.split("\n"):
        ln = re.sub(r"^\s*(\d+\.|-|\*)\s+", "", ln)
        w = ln.lower().split()
        if len(w) > SH:
            heads.add(" ".join(w[:SH]))
    return heads


def used_heads(out02: str, out04: str, out07: str, out08: str) -> tuple[set[str], set[str]]:
    texts: list[str] = []
    for f in (f"{out02}/data/wrap.pkl", f"{out02}/data/prose.pkl", f"{out04}/data/prose.pkl",
              f"{out04}/data/list_num.pkl", f"{out04}/data/list_bul.pkl", f"{out04}/data/dialogue.pkl"):
        try:
            with open(f, "rb") as fh:
                texts += _texts(pickle.load(fh))
        except FileNotFoundError:
            pass
    for f in (f"{out07}/items_base.json", f"{out07}/items_extra.json", f"{out07}/nj3_pool.json",
              f"{out08}/items.json", f"{out08}/calib_items.json", f"{out08}/r5_items.json"):
        try:
            with open(f) as fh:
                texts += _texts(json.load(fh))
        except FileNotFoundError:
            pass
    heads: set[str] = set()
    for t in texts:
        heads |= _line_heads(t)
    sha: set[str] = set()
    for f in (f"{out08}/items.json",):
        sha |= {it["passage_sha1_16"] for it in json.load(open(f))}
    for f in (f"{out08}/calib_items.json", f"{out08}/r5_items.json"):
        d = json.load(open(f))
        sha |= {it["source_sha1_16"] for it in d["tense"]}
    return heads, sha


def hit(p: str, heads: set[str]) -> bool:
    w = p.lower().split()
    return any(" ".join(w[i:i + SH]) in heads for i in range(max(0, len(w) - SH + 1)))


# ----------------------------------------------------------------------------- prose sets
def build_col(tok, s1, heads: set[str], sha: set[str]) -> list[dict]:  # noqa: ANN001
    pool = s1._wikitext_paragraphs(900, 60000, COL_SEED)
    rng = random.Random(COL_SEED)
    items = []
    for para in pool:
        if len(items) == N_COL:
            break
        if CI._sha(para) in sha or hit(para, heads):
            continue
        width = PI.WIDTHS[len(items) % len(PI.WIDTHS)]
        it = PI._wrap_item(tok, para, width, rng)
        if it is None:
            continue
        items.append({"id": f"o9_col_{len(items):02d}", "passage_sha1_16": CI._sha(para),
                      "passage_head": PI._norm(para)[:80], **it})
    assert len(items) == N_COL, len(items)
    return items


def build_sel(tok, s1, heads: set[str], sha: set[str]) -> list[dict]:  # noqa: ANN001
    pool = s1._wikitext_paragraphs(300, 60000, SEL_SEED)
    rng = random.Random(SEL_SEED)
    items = []
    for para in pool:
        if len(items) == N_SEL:
            break
        if CI._sha(para) in sha or hit(para, heads):
            continue
        ids = tok(para, add_special_tokens=False)["input_ids"][:160]
        if len(ids) < 24:
            continue
        j = rng.randrange(8, min(len(ids), 120))
        items.append({"id": f"o9_sel_{len(items):02d}", "ids": ids[: j + 1], "text": tok.decode(ids[: j + 1]),
                      "pos": j, "source_sha1_16": CI._sha(para), "source_head": PI._norm(para)[:80]})
    assert len(items) == N_SEL
    return items


def build_tense(tok, s1, out02: str, out07: str, heads: set[str], sha: set[str]) -> list[dict]:  # noqa: ANN001
    pool = s1._wikitext_paragraphs(500, 60000, TENSE_SEED)[:12000]
    bad = {CI._sha(p) for p in pool if hit(p, heads)}
    return CI.build_tense(tok, out02, out07, [], seed=TENSE_SEED, n_per_class=N_TENSE_PER,
                          exclude_sha=sha | bad, id_prefix="o9_tense", pool_slice=12000)


# ----------------------------------------------------------------------------- code
def _code_files() -> list[dict]:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("codeparrot/codeparrot-clean-valid", "file-000000000054.json.gz", repo_type="dataset")
    out = []
    with gzip.open(p, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            out.append({"repo": r.get("repo_name", ""), "path": r.get("path", ""), "content": r["content"]})
    return out


def _ok_file(code: str) -> list[str] | None:
    if "\t" in code or "def " not in code or len(code) > 200000:
        return None
    lines = [ln.rstrip() for ln in code.replace("\r\n", "\n").split("\n")]
    inds = [len(ln) - len(ln.lstrip(" ")) for ln in lines if ln.strip()]
    if not inds or any(i % 4 for i in inds) or 4 not in inds or 8 not in inds:
        return None
    return lines


def _ind(ln: str) -> int:
    return len(ln) - len(ln.lstrip(" ")) if ln.strip() else -1


def _context_start(lines: list[str], i: int) -> int:
    s = i
    while s > 0 and not (lines[s].strip() and _ind(lines[s]) == 0):
        s -= 1
    return max(s, i - 40)


def _nl_item(tok, lines: list[str], rng: random.Random, want_next: int) -> dict | None:  # noqa: ANN001
    cand = [i for i in range(3, len(lines) - 1)
            if lines[i].strip() and _ind(lines[i + 1]) == want_next and lines[i + 1].strip()]
    rng.shuffle(cand)
    for i in cand[:20]:
        s = _context_start(lines, i)
        text = "\n".join(lines[s:i + 1]) + "\n"
        full = text + lines[i + 1]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ids_full = tok(full, add_special_tokens=False)["input_ids"]
        if not (16 <= len(ids) <= MAX_TOK) or ids_full[: len(ids)] != ids or tok.decode([ids[-1]]) != "\n":
            continue
        return {"ids": ids, "text": text, "next_indent": want_next, "cur_indent": _ind(lines[i]),
                "y": int(want_next == 8), "line_index": i}
    return None


def _in_item(tok, lines: list[str], rng: random.Random, want_level: int) -> dict | None:  # noqa: ANN001
    cand = [i for i in range(3, len(lines) - 1)
            if _ind(lines[i]) == 4 * want_level and _ind(lines[i + 1]) in (4, 8) and len(lines[i].split()) >= 2]
    rng.shuffle(cand)
    for i in cand[:20]:
        s = _context_start(lines, i)
        body = "\n".join(lines[s:i + 1])
        ids_b = tok(body, add_special_tokens=False)["input_ids"]
        ids_f = tok(body + "\n" + lines[i + 1], add_special_tokens=False)["input_ids"]
        if not (16 <= len(ids_b) <= MAX_TOK) or ids_f[: len(ids_b)] != ids_b or tok.decode([ids_f[len(ids_b)]]) != "\n":
            continue
        return {"ids": ids_b, "text": body, "cur_level": want_level, "next_indent": _ind(lines[i + 1]),
                "y": int(want_level == 2), "line_index": i}
    return None


def build_code(tok, out04: str) -> dict:  # noqa: ANN001
    used = set()
    try:
        with open(f"{out04}/data/code_py.pkl", "rb") as fh:
            used = {PI._norm(t)[:200] for t in _texts(pickle.load(fh))}
    except FileNotFoundError:
        pass
    files = _code_files()
    rng = random.Random(CODE_SEED)
    rng.shuffle(files)
    want = {"codeNL": N_CODE_EVAL, "codeIN": N_CODE_EVAL, "codeTR": N_CODE_TRAIN}
    out: dict = {k: [] for k in want}
    n_skip_used = 0
    for f in files:
        if all(len(out[k]) == want[k] for k in want):
            break
        lines = _ok_file(f["content"])
        if lines is None:
            continue
        if PI._norm(f["content"])[:200] in used:
            n_skip_used += 1
            continue
        for k in ("codeNL", "codeIN", "codeTR"):  # fill eval sets first, one item per file
            if len(out[k]) < want[k]:
                break
        n = len(out[k])
        if k == "codeIN":
            it = _in_item(tok, lines, rng, 1 if n % 2 == 0 else 2)
        else:
            it = _nl_item(tok, lines, rng, 4 if n % 2 == 0 else 8)
        if it is None:
            continue
        out[k].append({"id": f"o9_{k}_{n:03d}", "repo": f["repo"], "path": f["path"], **it})
    assert all(len(out[k]) == want[k] for k in want), {k: len(v) for k, v in out.items()}
    out["_n_skip_used"] = n_skip_used
    return out
