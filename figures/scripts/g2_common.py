"""Shared loaders for the g2 figure scripts (exps 07, 12-17). Read-only on experiments/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

ROOT = style.ROOT
EXP = ROOT / "experiments"
E07 = EXP / "07_template_mining"
E12 = EXP / "12_concept_recall"
E13 = EXP / "13_new_rows"

READER_CELL = {"nla": "nla@42", "ol": "ol@42", "jlens": "jlens@42"}


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def exp_dir(feature: str) -> Path:
    """Exp 13 owns its new rows; every other concept row comes from exp 07 (used by exp 12)."""
    frozen13 = json.loads((E13 / "templates_frozen.json").read_text())["templates"]["nla"]
    return E13 if feature in frozen13 else E07


def clusters_for(feature: str) -> dict[str, str]:
    d = exp_dir(feature)
    if d == E13:
        items = json.loads((E13 / "results" / "items_base.json").read_text())
    else:
        items = json.loads((E07 / "results" / f"gen_ol_{feature}.json").read_text())["items"]
    return {it["id"]: it["cluster"] for it in items if it["feature"] == feature}


def prompted(feature: str, reader: str, n_boot: int = 2000):
    """Prompted AUROC as the record reports it: max(AUROC of the mean log-odds over the 3 selected
    templates, AUROC of the fixed label). Returns dict with mean3, fixed, reported, which, ci (cluster
    bootstrap of the reported variant), n, sources."""
    d = exp_dir(feature)
    frozen = json.loads((d / "templates_frozen.json").read_text())["templates"][reader][feature]
    sel_ids = [t["id"] for t in frozen["selected"]]
    fixed_id = frozen["fixed_label"]["id"]
    sp = d / "results" / f"score_{'nla' if reader == 'nla' else 'ol'}_{feature}.json"
    sc = json.loads(sp.read_text())["scores"][READER_CELL[reader]]
    y = np.asarray(sc["y"], int)
    cl = clusters_for(feature)
    groups = np.asarray([cl.get(i, i) for i in sc["ids"]])
    out = {"n": int(len(y)), "sources": [rel(sp), rel(d / "templates_frozen.json")]}
    los = [np.asarray(sc["lo"][t], float) for t in sel_ids if t in sc["lo"]]
    mean3 = np.mean(los, 0) if los else None
    fixed = np.asarray(sc["lo"][fixed_id], float) if fixed_id in sc["lo"] else None
    out["mean3"] = style.auroc(mean3, y) if mean3 is not None else float("nan")
    out["fixed"] = style.auroc(fixed, y) if fixed is not None else float("nan")
    cands = [(v, k, s) for v, k, s in ((out["mean3"], "mean3", mean3), (out["fixed"], "fixed", fixed)) if s is not None]
    v, k, s = max(cands, key=lambda c: c[0])
    out["reported"], out["which"] = float(v), k
    out["ci"] = style.bootstrap_ci(style.auroc, s, y, n_boot=n_boot, groups=groups)
    return out


def judged(feature: str, reader: str):
    """Judged free readouts (exp 12 for exp 07 rows, exp 13 for its own rows). reader in nla / ol42 / ol40 / ol44."""
    d = E13 if exp_dir(feature) == E13 else E12
    p = d / "results" / "judged" / f"{feature}_{reader}.json"
    return json.loads(p.read_text()), rel(p)


def states_rate(feature: str, reader: str, n_boot: int = 2000):
    """Share of free readouts judged to state the property. CI: item-clustered bootstrap when k > 0,
    Wilson on readouts when k == 0 (a bootstrap of zeros is degenerate)."""
    d, src = judged(feature, reader)
    rows = d["judged"]
    s = np.asarray([bool(r["stated"]) for r in rows], float)
    g = np.asarray([r["item"] for r in rows])
    k, n = int(s.sum()), len(s)
    vals = d["values"]
    code = {vals[0]: "A", vals[1]: "B"}
    ex = np.asarray([r["exhibited_value"] == code[r["value"]] for r in rows], float)
    if k > 0:
        lo, hi = style.bootstrap_ci(lambda a: float(a.mean()), s, n_boot=n_boot, groups=g)
        method = "item-clustered bootstrap"
    else:
        _, lo, hi = style.wilson(0, n)
        method = "Wilson"
    kx = int(ex.sum())
    if kx > 0:
        xlo, xhi = style.bootstrap_ci(lambda a: float(a.mean()), ex, n_boot=n_boot, groups=g)
    else:
        _, xlo, xhi = style.wilson(0, n)
    return {"k": k, "n": n, "rate": k / n, "ci": (lo, hi), "method": method, "n_items": len(set(g)),
            "exhibit_k": kx, "exhibit_rate": kx / n, "exhibit_ci": (xlo, xhi), "source": src}


def check(quantity: str, computed, record, where: str, tol: float = 0.005) -> dict:
    if isinstance(computed, (int, float)) and isinstance(record, (int, float)):
        match = abs(float(computed) - float(record)) <= tol
    else:
        match = computed == record
    return {"quantity": quantity, "computed": computed, "record": record, "where": where, "match": bool(match)}


# ---- section 4 shared grammar: same row order, same row labels, control set apart in grey
S4_ROWS = [("tb_tablecol", "table column"), ("tb_paren", "inside parentheses"), ("bk_listidx", "list item index"),
           ("ij2_lang", "language\n(control)")]
S4_Y = {"tb_tablecol": 3.9, "tb_paren": 2.9, "bk_listidx": 1.9, "ij2_lang": 0.4}


def s4_rows(ax, rows=None, labels=True):
    """Apply the S4 row axis: rows top to bottom, control row grey and separated by a rule."""
    rows = rows or [r for r, _ in S4_ROWS]
    lab = dict(S4_ROWS)
    ax.set_yticks([S4_Y[r] for r in rows])
    ax.set_yticklabels([lab[r] for r in rows] if labels else [])
    if labels:
        for t, r in zip(ax.get_yticklabels(), rows):
            if r == "ij2_lang":
                t.set_color(style.C["muted"])
    ax.set_ylim(-0.35, 4.45)
    if "ij2_lang" in rows:
        ax.axhline(1.15, color=style.C["line"], lw=1)
