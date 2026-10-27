"""Build the exp 22 PUBLIC-SOURCE natural bank on the laptop (no GPU): the exp 19 recipe restricted to files that are
published package distributions on this machine (site-packages, dist-packages, node_modules, the bun and uv caches,
conda pkgs) plus arXiv e-prints (LaTeX) and arXiv HTML renderings (HTML), so every item is publishable; tokenised
with the Qwen3.6-27B tokenizer, split 20+20 mine / 80+80 select per row, and checked for lexical cues
(bag-of-tokens probe on the last 48 tokens, grouped by source file). Writes results/items_base.json (untracked, like
exps 13/18) and results/build_info.json (counts, token lengths, lexical AUROC per row, sha256). Then:

    modal volume put rfm-results results/items_base.json /22_public_bank/items_base.json

    .venv/bin/python build22.py --arxiv <dir with src_*/ e-print folders> [--home ~]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import items_build22 as IB  # noqa: E402
from feature_spec22 import FEATURES, LEX_MAX, N_MINE_PER_VALUE, N_SELECT_PER_VALUE, N_PER_VALUE  # noqa: E402
from lib22 import lexical_cue_auroc  # noqa: E402

RES = HERE / "results"
REPO = HERE.parent.parent
EXT = {"py_string": "py", "py_arg": "py", "js_string": "js", "md_heading": "md", "md_list": "md",
       "json_depth": "json", "csv_field": "csv", "html_depth": "html", "latex_math": "tex"}
MIN_CHARS, MAX_CHARS = 2000, 300_000
MAX_FILES = {"py": 6000, "js": 3000, "md": 3000, "json": 2000, "csv": 300, "html": 2000, "tex": 400}
EXCLUDE = ("/Library/", "/.claude/", "/.git/", "/.Trash/", str(REPO) + "/")   # never this repo's own files
PUBLIC_DIRS = ("/site-packages/", "/dist-packages/", "/node_modules/", "/.bun/install/cache/", "/.cache/uv/", "/miniconda3/pkgs/")
MAX_TOKENS = 512


ARXIV_HTML: Path | None = None


def find_files(home: Path, ext: str, arxiv: Path | None) -> list[Path]:
    """Only published package files and arXiv material (exp 22's one change from exp 19)."""
    if ext == "tex":
        return sorted(p for p in (arxiv or Path("/nonexistent")).rglob("*.tex") if p.stat().st_size > 4000)
    cmd = ["find", str(home), "-type", "f", "-name", f"*.{ext}", "-size", "+2k"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.splitlines()
    out = [p for p in out if not any(x in p for x in EXCLUDE) and ".min." not in p and any(d in p for d in PUBLIC_DIRS)]
    paths = [Path(p) for p in out]
    if ext == "html" and ARXIV_HTML is not None:
        paths += [p for p in ARXIV_HTML.glob("*.html") if p.stat().st_size > 20000]
    return sorted(paths)


def load(paths: list[Path], rng: random.Random, cap: int) -> list[tuple[str, str]]:
    paths = list(paths)
    rng.shuffle(paths)
    out, seen = [], set()
    for p in paths[:cap]:
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not (MIN_CHARS <= len(t) <= MAX_CHARS):
            continue
        h = hashlib.sha1(t.encode()).hexdigest()[:12]
        if h in seen:
            continue
        seen.add(h)
        src = "arxiv" if ("/arxiv" in str(p.parent) or "/arxiv_html" in str(p.parent)) else "package"
        out.append((f"{p.name}_{h}", t, src))
    return out


def assign_splits(items: list[dict], rng: random.Random, n_mine: int, n_sel: int) -> list[dict]:
    """Per value: shuffle, first n_mine -> mine, next n_sel -> select (exp 07's rule, as in exp 18)."""
    out, by_val = [], {}
    for it in items:
        by_val.setdefault(it["value"], []).append(it)
    for _, lst in sorted(by_val.items()):
        rng.shuffle(lst)
        out += [{**it, "split": "mine" if k < n_mine else "select"} for k, it in enumerate(lst[: n_mine + n_sel])]
    return out


def build_row(row: str, files: list[tuple[str, str, str]], rng: random.Random) -> list[dict]:
    src_of = {name: src for name, _, src in files}
    items = [{**it, "meta": {**it["meta"], "src": src_of[it["cluster"]]}} for it in IB.BUILDERS[row]([(n, t) for n, t, _ in files], rng)]
    seen, uniq = set(), []
    for it in items:
        if it["text"] not in seen:
            seen.add(it["text"])
            uniq.append(it)
    return assign_splits(uniq, rng, N_MINE_PER_VALUE, N_SELECT_PER_VALUE)


def word_filter(tok):  # noqa: ANN001
    """Alphabetic word tokens only (cached per id): the vocabulary cue, not the structural markers."""
    cache: dict[int, bool] = {}

    def keep(t: int) -> bool:
        if t not in cache:
            s = tok.decode([t]).strip()
            cache[t] = len(s) >= 2 and s.isalpha()
        return cache[t]
    return keep


def tokenise(items: list[dict], tok) -> list[dict]:  # noqa: ANN001
    out = []
    for it in items:
        ids = tok(it["text"], add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_TOKENS:                       # keep the read point: trim the front at a token boundary
            ids = ids[-MAX_TOKENS:]
            it = {**it, "text": tok.decode(ids)}
        out.append({**it, "ids": ids})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(Path.home()))
    ap.add_argument("--arxiv", default=None)
    ap.add_argument("--arxiv-html", default=None)
    ap.add_argument("--rows", default=",".join(FEATURES))
    a = ap.parse_args()
    global ARXIV_HTML
    ARXIV_HTML = Path(a.arxiv_html) if a.arxiv_html else None
    from transformers import AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    rng = random.Random(IB.SEED)
    files_by_ext: dict[str, list[tuple[str, str]]] = {}
    items, info = [], {"rows": {}, "n_source_files": {}}
    for row in a.rows.split(","):
        for ext in (("py", "js") if row == "code_lang" else (EXT[row],)):
            if ext not in files_by_ext:
                paths = find_files(Path(a.home), ext, Path(a.arxiv) if a.arxiv else None)
                files_by_ext[ext] = load(paths, rng, MAX_FILES[ext])
                info["n_source_files"][ext] = {"found": len(paths), "loaded": len(files_by_ext[ext])}
        if row == "code_lang":
            src_of = {n: s for pool in ("py", "js") for n, _, s in files_by_ext[pool]}
            raw = IB.build_code_lang([(n, t) for n, t, _ in files_by_ext["py"]], [(n, t) for n, t, _ in files_by_ext["js"]], rng)
            raw = [{**it, "meta": {**it["meta"], "src": src_of[it["cluster"]]}} for it in raw]
            seen, uniq = set(), []
            for it in raw:
                if it["text"] not in seen:
                    seen.add(it["text"]); uniq.append(it)
            rows_items = tokenise(assign_splits(uniq, rng, N_MINE_PER_VALUE, N_SELECT_PER_VALUE), tok)
        else:
            rows_items = tokenise(build_row(row, files_by_ext[EXT[row]], rng), tok)
        c = Counter((it["split"], it["value"]) for it in rows_items)
        y = [int(it["value"] == FEATURES[row]["values"][0]) for it in rows_items]
        groups = [it["cluster"] for it in rows_items]
        lex = (lexical_cue_auroc([it["ids"] for it in rows_items], y, groups, keep=word_filter(tok))
               if len(set(y)) == 2 and len(rows_items) >= 40 else None)
        L = sorted(len(it["ids"]) for it in rows_items) or [0]
        full = all(c[(s, v)] == n for v in FEATURES[row]["values"] for s, n in (("mine", N_MINE_PER_VALUE), ("select", N_SELECT_PER_VALUE)))
        srcs = Counter(it["meta"]["src"] for it in rows_items)
        info["rows"][row] = {"n": len(rows_items), "counts": {"|".join(k): v for k, v in sorted(c.items())}, "full": full,
                             "sources": dict(srcs),
                             "lexical_auroc": lex, "lexically_cued": (lex is not None and lex >= LEX_MAX),
                             "token_len_q10_q50_q90": [L[len(L) // 10], L[len(L) // 2], L[9 * len(L) // 10]],
                             "n_clusters": len(set(groups))}
        print(f"{row:11s} n={len(rows_items):4d} full={full} lex={lex if lex is None else round(lex, 3)} "
              f"tok q50={L[len(L) // 2]}  clusters={len(set(groups))}", flush=True)
        items += rows_items
    RES.mkdir(exist_ok=True)
    blob = json.dumps(items)
    (RES / "items_base.json").write_text(blob)
    info.update({"n_items": len(items), "sha256": hashlib.sha256(blob.encode()).hexdigest(), "n_per_value": N_PER_VALUE,
                 "lex_max": LEX_MAX, "wall_s": round(time.time() - t0, 1)})
    (RES / "build_info.json").write_text(json.dumps(info, indent=1))
    print("sha256", info["sha256"], "wall", info["wall_s"])


if __name__ == "__main__":
    main()
