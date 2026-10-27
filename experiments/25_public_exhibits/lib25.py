"""Pure helpers for exp 25 (the reconstructor-based "exhibits" measure of exp 20, run on the exp 22 public bank, and
the rescoring of exp 22's scorecard with it).

The scoring functions (Wilson accuracy, confusion, the markdown table) are exp 22's own, imported from
`score22.py`; the text and probe helpers used on the GPU are exp 20's (`lib20.py`), copied here with their names
unchanged because the Modal image of this experiment only ships this directory's modules."""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping

_HERE = Path(__file__).resolve().parent
_EXP22 = _HERE.parent / "22_public_bank"
if str(_EXP22) not in sys.path:
    sys.path.append(str(_EXP22))                     # appended, not inserted: this directory's env.py must win
from feature_spec22 import CELL_THRESH, FEATURES  # noqa: E402,F401


def score22():  # noqa: ANN201
    """exp 22's scoring module (`accuracy`, `confusion`, `markdown`, `CELLS`), imported on first use: it is not
    shipped to the GPU image, and nothing on the GPU needs it."""
    import score22 as S

    return S


ROWS: tuple[str, ...] = tuple(FEATURES)
MARKED: dict[str, str] = {row: spec["values"][0] for row, spec in FEATURES.items()}   # exactly modal22's `_sel`
EXHIBITS_LINE = 0.70                                 # exp 20's line for the reconstructor measure
CONDITIONS = ("span_32", "span_128", "bag_32", "ol_readout", "nla_readout")
WINDOWS = (32, 128)
BAG_SEED = 25

# exp 22 RESULTS.md "Scorecard": primary = the free-readout cells on every row plus the J-lens cell on the six
# uncued rows (30 of 38); secondary = the prompted forced-choice cells (13 of 16); all 56 cells = 44.
PRIMARY_CELLS = ("nla_states", "nla_names_complement", "ol_exhibits", "ol_states", "jlens_prompted")
SECONDARY_CELLS = ("nla_prompted", "ol_prompted")
EXCLUDE: dict[str, frozenset[str]] = {"jlens_prompted": frozenset({"json_depth", "code_lang"})}

_EXPL = re.compile(r"<explanation>(.*?)</explanation>", re.S)


# ---- rescoring --------------------------------------------------------------------------------------------------

def swap_exhibits(score22_rows: list[dict], recon_by_row: Mapping[str, dict], line: float = EXHIBITS_LINE) -> list[dict]:
    """New rows with the `ol_exhibits` cell re-observed as the reconstructor measure (`auroc.ol_readout` of the
    row's recon25 file) at `line`; the prediction and every other cell are copied unchanged. KeyError names the
    row when its reconstruction is missing."""
    out = []
    for row in score22_rows:
        feat = row["feature"]
        if feat not in recon_by_row:
            raise KeyError(f"no reconstruction result for row {feat!r}")
        v = float(recon_by_row[feat]["auroc"]["ol_readout"])
        old = row["cells"]["ol_exhibits"]
        high = v >= line
        cell = {"predicted_high": bool(old["predicted_high"]), "observed": v, "observed_high": bool(high),
                "correct": bool(high == bool(old["predicted_high"]))}
        out.append({**row, "observed": {**row["observed"], "ol_exhibits": v}, "cells": {**row["cells"], "ol_exhibits": cell}})
    return out


def _subset(rows: Iterable[dict], cells: Iterable[str], exclude: Mapping[str, Iterable[str]]) -> list[dict]:
    return [{c: r["cells"][c] for c in cells if c in r["cells"] and r["feature"] not in set(exclude.get(c, ()))} for r in rows]


def accuracy_split(rows: list[dict], primary_cells: Iterable[str], secondary_cells: Iterable[str],
                   exclude: Mapping[str, Iterable[str]]) -> dict:
    """Wilson accuracy (score22's) over the primary cells, the secondary cells and all cells of `rows`;
    `exclude` maps a cell to the rows on which it is not scored in the primary and secondary counts (exp 22: the
    J-lens cell on the cued rows); "all" is exp 18's convention, every scored cell, nothing excluded."""
    accuracy = score22().accuracy
    primary, secondary = tuple(primary_cells), tuple(secondary_cells)
    return {"primary": accuracy(_subset(rows, primary, exclude)),
            "secondary": accuracy(_subset(rows, secondary, exclude)),
            "all": accuracy(_subset(rows, primary + secondary, {}))}


def per_row_lines(old_rows: list[dict], new_rows: list[dict]) -> list[str]:
    by_feat = {r["feature"]: r for r in new_rows}
    out = []
    for r in old_rows:
        o, n = r["cells"]["ol_exhibits"], by_feat[r["feature"]]["cells"]["ol_exhibits"]
        out.append(f"{r['feature']}: judge {o['observed']:.2f} -> recon {n['observed']:.2f} "
                   f"(predicted {'high' if n['predicted_high'] else 'low'}, {'ok' if n['correct'] else 'MISS'})")
    return out


# ---- the selection split (modal22's `_sel`, as a pure function) -----------------------------------------------------

def select_split(items: list[dict], marked: str):  # noqa: ANN201
    import numpy as np

    idx = [i for i, it in enumerate(items) if it["split"] == "select"]
    y = np.array([items[i]["value"] == marked for i in idx], int)       # 1 = marked value
    return idx, y, np.array([items[i]["cluster"] for i in idx])


# ---- exp 20's text and probe helpers (lib20.py, verbatim) -------------------------------------------------------

def windows(ids: list[int], sizes=WINDOWS) -> dict[int, list[int]]:  # noqa: ANN001
    return {w: ids[-w:] for w in sizes}


def shuffle_tokens(ids: list[int], rng: random.Random) -> list[int]:
    """The same tokens in a random order (structure broken, vocabulary kept)."""
    out = list(ids)
    for _ in range(10):
        rng.shuffle(out)
        if out != list(ids):
            break
    return out


def explanation_text(raw: str) -> str:
    m = _EXPL.search(raw)
    t = (m.group(1) if m else raw.split("</explanation>")[0]).strip()
    return t if t else "(empty)"


def frozen_probe_auroc(X_true, R, y, groups, k: int = 5, lam: float = 1e-3, seed: int = 0) -> float:  # noqa: ANN001
    """Exp 14's measure: a probe fit on the true activations (grouped folds), applied frozen to the reconstructions
    of the held-out items; AUROC against the labels."""
    import numpy as np
    import torch

    from mats.metrics import auroc_labels
    from mats.probes import fit_logreg, group_folds

    y = np.asarray(y)
    sc = np.empty(len(y))
    for te in group_folds(np.asarray(groups), k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = fit_logreg(X_true[tr].float(), torch.as_tensor(y[tr]), lam)
        sc[te] = m.decision(R[te]).numpy()
    return float(auroc_labels(sc, y))


def table(rows: list[dict], key: str = "auroc", conds: tuple[str, ...] = CONDITIONS) -> str:
    """exp 20's table over this experiment's five conditions (lib20.table has the window sweep columns too)."""
    conds = tuple(c for c in conds if any(c in r[key] for r in rows))
    out = ["| row | " + " | ".join(conds) + " |", "|---|" + "---:|" * len(conds)]
    for r in rows:
        out.append(f"| {r['feature']} | " + " | ".join(f"{r[key][c]:.2f}" if c in r[key] else "" for c in conds) + " |")
    return "\n".join(out)
