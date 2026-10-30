"""Exp 31 analysis (laptop, CPU only): copy overlap of each stored k = 1 readout with its item's last 32 tokens, and
the exps 20/23/27 text-bag carrying measure after deleting the copied part. Reads exp 22/25, exp 19/13/20 and exp 27
results (read-only) -> results/overlap31.json, results/carry31.json, results/table31.md. See RESULTS.md for the design."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP = REPO / "experiments"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXP / "19_natural_bank"))
sys.path.insert(0, str(EXP / "23_random_points"))
from lib31 import (auroc, delete_tokens, derangement, is_content, low_half_mask, ngram_overlap, norm_piece,  # noqa: E402
                   opposite_partners, paired_cluster_bootstrap, strip_tags, token_overlap, words)

SEED = 31
WINDOW = 32
N_BOOT = 2000
N_RAND_HALVES = 20
MIN_EACH = 20
READERS = ("ol", "nla")
MARKED = {"py_string": "inside", "js_string": "inside", "latex_math": "inside", "py_arg": "third", "json_depth": "deep",
          "html_depth": "deep", "md_heading": "deep", "md_list": "nested", "tb_tablecol": "first", "tb_paren": "inside",
          "code_lang": "python"}
ROWS25 = ["py_string", "js_string", "latex_math", "py_arg", "json_depth", "md_heading", "md_list", "code_lang"]
ROWS20 = ["py_string", "js_string", "latex_math", "py_arg", "json_depth", "html_depth", "md_heading", "md_list",
          "tb_tablecol", "tb_paren"]
CARRY4 = ["py_string", "js_string", "latex_math", "json_depth"]
CONDS = ("full", "del_own", "del_bal", "del_rand", "del_row")

_TOK = None
_NORM: dict[int, str] = {}


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    return _TOK


def enc(s: str) -> list[int]:
    return tok()(s or "", add_special_tokens=False)["input_ids"]


def norm_of(t: int) -> str:
    if t not in _NORM:
        _NORM[t] = norm_piece(tok().decode([t]))
    return _NORM[t]


def load(p: Path):
    return json.loads(Path(p).read_text())


def cells():
    """Yield (source, cell, items, y) for the three banks."""
    items22 = load(EXP / "22_public_bank/results/items_base.json")
    for row in ROWS25:
        its = [it for it in items22 if it["feature"] == row and it["split"] == "select"]
        texts = {r: load(EXP / f"25_public_exhibits/results/{r}_sel_{row}.json")[r] for r in READERS}
        yield "e25", row, its, [int(it["value"] == MARKED[row]) for it in its], texts
    items19 = load(EXP / "19_natural_bank/results/items_base.json")
    items13 = load(EXP / "13_new_rows/results/items_base.json")
    for row in ROWS20:
        base = items13 if row.startswith("tb_") else items19
        its = [it for it in base if it["feature"] == row and it["split"] == "select"]
        texts = {r: load(EXP / f"20_ar_structure/results/{r}_sel_{row}.json")[r] for r in READERS}
        yield "e20", row, its, [int(it["value"] == MARKED[row]) for it in its], texts
    from analyze23 import STATES
    items27 = load(EXP / "27_random_2000/results/items_base.json")
    texts = {"ol": load(EXP / "27_random_2000/results/ol27.json")["ol"],
             "nla": load(EXP / "27_random_2000/results/nla27.json")["nla"]}
    kinds = sorted({it["meta"]["kind"] for it in items27})
    for kind, members in [(k, (k,)) for k in kinds] + [("code (py+js)", ("py", "js"))]:
        its = [it for it in items27 if it["meta"]["kind"] in members]
        for st, fn in STATES.items():
            y = [int(fn(it["meta"]["labels"][st])) for it in its]
            c = Counter(y)
            if c.get(0, 0) < MIN_EACH or c.get(1, 0) < MIN_EACH:
                continue
            yield "e27", f"{kind} / {st}", its, y, texts


def oof_scores(X, y, groups, k: int = 5, lam: float = 1.0, seed: int = 0) -> np.ndarray:
    """Out-of-fold scores of mats.probes' grouped logistic probe (same folds and fit as mats.probes.cv_auroc)."""
    import torch

    from mats.probes import fit_logreg, group_folds
    y = np.asarray(y)
    sc = np.empty(len(y))
    for te in group_folds(np.asarray(groups), k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = fit_logreg(X[tr].float(), torch.as_tensor(y[tr]), lam)
        sc[te] = m.decision(X[te]).cpu().numpy()
    return sc


def bag_scores(id_lists, y, groups) -> np.ndarray:
    from lib19 import bag_of_tokens
    return oof_scores(bag_of_tokens(id_lists, window=10**6, min_df=3), y, groups)


def overlap_block(r_texts, windows, wtexts, perm, groups):
    own = {"tok": [], "bi": [], "tri": []}
    shuf = {"tok": [], "bi": [], "tri": []}
    wnorm = [set(norm_of(t) for t in w) for w in windows]
    wwords = [words(t) for t in wtexts]
    for i, text in enumerate(r_texts):
        clean = strip_tags(text)
        rn = [norm_of(t) for t in enc(clean)]
        rw = words(clean)
        for d, j in ((own, i), (shuf, perm[i])):
            d["tok"].append(token_overlap(rn, wnorm[j]))
            d["bi"].append(ngram_overlap(rw, wwords[j], 2))
            d["tri"].append(ngram_overlap(rw, wwords[j], 3))
    out = {}
    for m in ("tok", "bi", "tri"):
        ok = np.array([a is not None and b is not None for a, b in zip(own[m], shuf[m])])
        a = np.array([x if x is not None else np.nan for x in own[m]])
        b = np.array([x if x is not None else np.nan for x in shuf[m]])
        diff = a - b
        ci = paired_cluster_bootstrap({"ex": lambda idx: np.nanmean(diff[idx])}, groups, n_boot=N_BOOT, seed=0)["ex"]
        q = np.nanpercentile(a, [25, 50, 75])
        out[m] = {"n": int(ok.sum()), "own": float(np.nanmean(a)), "shuf": float(np.nanmean(b)),
                  "excess": float(np.nanmean(diff)), "excess_ci": list(ci), "own_q": [float(v) for v in q]}
    return out, own["tok"]


def carry_block(r_texts, windows, y, groups, perm, partner, row_vocab, own_tok):
    ids = [enc(t) for t in r_texts]
    wn = [set(norm_of(t) for t in w) for w in windows]
    lists = {"full": ids,
             "del_own": [delete_tokens(r, norm_of, wn[i]) for i, r in enumerate(ids)],
             "del_bal": [delete_tokens(r, norm_of, wn[i] | wn[partner[i]]) for i, r in enumerate(ids)],
             "del_rand": [delete_tokens(r, norm_of, wn[perm[i]]) for i, r in enumerate(ids)],
             "del_row": [delete_tokens(r, norm_of, row_vocab) for r in ids]}
    y = np.asarray(y)
    sc = {c: bag_scores(lists[c], y, groups) for c in CONDS}
    kept = {c: float(np.mean([len(l) / max(1, len(f)) for l, f in zip(lists[c], ids)])) for c in CONDS}
    stats = {c: (lambda idx, s=sc[c]: auroc(s[idx], y[idx])) for c in CONDS}
    stats["drop_bal"] = lambda idx: auroc(sc["full"][idx], y[idx]) - auroc(sc["del_bal"][idx], y[idx])
    stats["drop_own"] = lambda idx: auroc(sc["full"][idx], y[idx]) - auroc(sc["del_own"][idx], y[idx])
    ci = paired_cluster_bootstrap(stats, groups, n_boot=N_BOOT, seed=0)
    res = {c: {"auroc": auroc(sc[c], y), "ci": list(ci[c]), "kept_frac": kept[c]} for c in CONDS}
    res["drop_bal"] = {"value": res["full"]["auroc"] - res["del_bal"]["auroc"], "ci": list(ci["drop_bal"])}
    res["drop_own"] = {"value": res["full"]["auroc"] - res["del_own"]["auroc"], "ci": list(ci["drop_own"])}
    # low-overlap half vs random halves of the same size
    low = low_half_mask(own_tok)
    def half_auroc(mask):
        idx = np.flatnonzero(mask)
        if min(Counter(y[idx].tolist()).get(0, 0), Counter(y[idx].tolist()).get(1, 0)) < 5:
            return float("nan")
        return auroc(bag_scores([ids[i] for i in idx], y[idx], groups[idx]), y[idx])
    rh = []
    for s in range(SEED, SEED + N_RAND_HALVES):
        m = np.zeros(len(y), bool)
        m[np.random.default_rng(s).choice(len(y), int(low.sum()), replace=False)] = True
        rh.append(half_auroc(m))
    res["low_half"] = {"auroc": half_auroc(low), "n": int(low.sum()), "n_pos": int(y[low].sum()),
                       "rand_half_mean": float(np.nanmean(rh)), "rand_half_sd": float(np.nanstd(rh))}
    return res


def check_alignment(its) -> int:
    """Count items whose text does not end with the decoded tail of their ids (the read point must be the last id)."""
    bad = 0
    for it in its[:20]:
        tail = tok().decode(it["ids"][-4:]).strip()
        if tail and tail not in it["text"][-200:]:
            bad += 1
    return bad


def main() -> None:
    overlap, carry, notes = {}, {}, []
    for src, cell, its, y, texts in cells():
        key = f"{src} | {cell}"
        bad = check_alignment(its)
        if bad:
            notes.append(f"{key}: {bad} of 20 items whose text tail does not match ids[-4:]")
        groups = np.array([it["cluster"] for it in its])
        windows = [it["ids"][-WINDOW:] for it in its]
        wtexts = [tok().decode(w) for w in windows]
        perm = derangement(len(its), SEED)
        partner = opposite_partners(y, SEED)
        row_vocab = set(norm_of(t) for w in windows for t in w)
        yv = np.asarray(y)
        wb = bag_scores(windows, yv, groups)
        wci = paired_cluster_bootstrap({"a": lambda idx: auroc(wb[idx], yv[idx])}, groups, n_boot=N_BOOT)["a"]
        overlap[key], carry[key] = {"n": len(its), "n_pos": int(yv.sum())}, {"n": len(its), "n_pos": int(yv.sum()),
                                                                              "window_bag": {"auroc": auroc(wb, yv), "ci": list(wci)}}
        for r in READERS:
            r_texts = [texts[r][it["id"]] for it in its]
            ov, own_tok = overlap_block(r_texts, windows, wtexts, perm, groups)
            overlap[key][r] = ov
            carry[key][r] = carry_block(r_texts, windows, y, groups, perm, partner, row_vocab, own_tok)
        print(key, json.dumps({r: (round(overlap[key][r]["tok"]["own"], 3), round(overlap[key][r]["tok"]["shuf"], 3),
                                   round(carry[key][r]["full"]["auroc"], 3), round(carry[key][r]["del_bal"]["auroc"], 3))
                               for r in READERS}), flush=True)
    R = HERE / "results"
    (R / "overlap31.json").write_text(json.dumps(overlap, indent=1))
    (R / "carry31.json").write_text(json.dumps({"carry": carry, "notes": notes}, indent=1))
    (R / "table31.md").write_text(render(overlap, carry, notes))
    print(render(overlap, carry, notes))


def _ci(d) -> str:
    return f"{d['auroc']:.2f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]"


def render(overlap, carry, notes) -> str:
    L = ["## (a) Copy overlap with the item's last 32 tokens (own) vs a random other item's window from the same cell (shuffled)", "",
         "| bank | cell | reader | content tokens: own / shuffled | excess [95% CI] | own quartiles | word 2-grams own / shuf | word 3-grams own / shuf |",
         "|---|---|---|---|---|---|---|---|"]
    for key, d in overlap.items():
        src, cell = key.split(" | ")
        for r in READERS:
            t, b, tr = d[r]["tok"], d[r]["bi"], d[r]["tri"]
            L.append(f"| {src} | {cell} | {r} | {t['own']:.2f} / {t['shuf']:.2f} | {t['excess']:+.2f} [{t['excess_ci'][0]:+.2f}, {t['excess_ci'][1]:+.2f}] | "
                     f"{t['own_q'][0]:.2f} / {t['own_q'][1]:.2f} / {t['own_q'][2]:.2f} | {b['own']:.2f} / {b['shuf']:.2f} | {tr['own']:.2f} / {tr['shuf']:.2f} |")
    L += ["", "## (b) Text-bag carrying (AUROC [95% cluster CI]) before and after deleting window tokens from the readout", "",
          "| bank | cell | n (pos) | window bag | reader | full | del_own | del_bal (primary) | drop full-del_bal [CI] | del_rand | del_row | tokens kept: own / bal / row | low half (n) | random halves |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in carry.items():
        src, cell = key.split(" | ")
        for r in READERS:
            c = d[r]
            dr = c["drop_bal"]
            L.append(f"| {src} | {cell} | {d['n']} ({d['n_pos']}) | {d['window_bag']['auroc']:.2f} | {r} | {_ci(c['full'])} | {_ci(c['del_own'])} | {_ci(c['del_bal'])} | "
                     f"{dr['value']:+.2f} [{dr['ci'][0]:+.2f}, {dr['ci'][1]:+.2f}] | {c['del_rand']['auroc']:.2f} | {_ci(c['del_row'])} | "
                     f"{c['del_own']['kept_frac']:.2f} / {c['del_bal']['kept_frac']:.2f} / {c['del_row']['kept_frac']:.2f} | "
                     f"{c['low_half']['auroc']:.2f} ({c['low_half']['n']}, {c['low_half']['n_pos']} pos) | {c['low_half']['rand_half_mean']:.2f} ± {c['low_half']['rand_half_sd']:.2f} |")
    if notes:
        L += ["", "Alignment notes: " + "; ".join(notes)]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
