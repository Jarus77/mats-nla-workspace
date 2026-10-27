"""Pure helpers for exp 26 (the broad Oracle Lens layer sweep on the exp 22 public bank). No model, no torch at
import time; the GPU stages in modal26.py and the laptop analysis in analyze26.py both import this file.

Provenance: `select_items` is modal22's `_sel` (as lib25's `select_split`, returning the items); `prompted_candidates`
and `prompted_logodds` are modal22's `_cands_for` and the `lo = (lp[:, 0::2] - lp[:, 1::2]).T` line of its
`_score_generic` (modal22 is not importable: it binds exp 22's `env`); `prompted_auroc` is select_freeze's
`mean_of_selected` / `fixed_label` and score22's `_prompted` (max of the two); `frozen_probe_auroc` is exp 14's
measure as lib25 ships it. The strict naming regexes of the structural rows are exp 23's (`lib23.STATE_PHRASES`);
`py_arg` and `code_lang` have no exp 23 state, so their regexes are declared here."""
from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _d in (_HERE.parent / "22_public_bank", _HERE.parent / "23_random_points"):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))                     # appended, not inserted: this directory's env.py must win
from feature_spec import fill  # noqa: E402
from feature_spec22 import FEATURES, LAYER  # noqa: E402,F401
from lib23 import STATE_PHRASES  # noqa: E402

ROWS: tuple[str, ...] = tuple(FEATURES)
MARKED: dict[str, str] = {row: spec["values"][0] for row, spec in FEATURES.items()}   # exactly modal22's `_sel`
TRAINED_LAYERS: tuple[int, ...] = tuple(range(20, 61, 4))                            # olens.TRAINED_LAYERS
MEASURES: tuple[str, ...] = ("readable", "ol_prompted", "carries", "names")
LINES: tuple[float, ...] = (0.70, 0.80)

# exp 22 row -> exp 23 state (lib23.STATE_RE); the two rows without one get a regex of their own below.
STATE_OF: dict[str, str] = {"py_string": "in_string", "js_string": "in_string", "json_depth": "brace_depth",
                            "latex_math": "in_math", "md_heading": "heading_level", "md_list": "list_depth"}
_OWN: dict[str, str] = {"py_arg": r"\b(first|second|third|1st|2nd|3rd)\s+(argument|parameter|arg)\b",
                        "code_lang": r"\b(python|javascript|typescript)\b"}
NAME_RE: dict[str, re.Pattern] = {**{row: re.compile(STATE_PHRASES[st], re.IGNORECASE) for row, st in STATE_OF.items()},
                                  **{row: re.compile(p, re.IGNORECASE) for row, p in _OWN.items()}}


def sweep_layers() -> list[int]:
    """olens.SWEEP_LAYERS: the trained grid 20..60 step 4 plus the primary read layer 42, ascending."""
    return sorted({*TRAINED_LAYERS, LAYER})


# ---- naming ---------------------------------------------------------------------------------------------------------

def names_state(text: str | None, row: str) -> bool:
    """Strict: does the readout say the read point is in the row's state (exp 23's phrasing for the structural rows)?"""
    return bool(NAME_RE[row].search(text or ""))


def names_rate(texts: Iterable[str | None], row: str) -> float:
    texts = list(texts)
    if not texts:
        return float("nan")
    return sum(names_state(t, row) for t in texts) / len(texts)


# ---- the selection split --------------------------------------------------------------------------------------------

def select_items(items: list[dict], marked: str):  # noqa: ANN201
    """(selection items in file order, y [1 = marked value], cluster groups)."""
    import numpy as np

    sel = [it for it in items if it["split"] == "select"]
    y = np.array([it["value"] == marked for it in sel], int)
    return sel, y, np.array([it["cluster"] for it in sel])


# ---- Oracle Lens prompted AUROC, exp 22's convention ----------------------------------------------------------------

def ol_templates(frozen: dict, row: str) -> tuple[list[dict], list[str], str]:
    """The four frozen Oracle Lens phrasings of `row` (the three mined, selected ones and the fixed label), each
    reduced to {id, template, fillers}; plus the selected ids and the fixed id."""
    t = frozen["templates"]["ol"][row]
    keep = lambda e: {k: e[k] for k in ("id", "template", "fillers")}  # noqa: E731
    selected = [keep(e) for e in t["selected"]]
    fixed = keep(t["fixed_label"])
    return selected + [fixed], [e["id"] for e in selected], fixed["id"]


def prompted_candidates(templates: Sequence[dict], vals: Sequence[str]) -> list[str]:
    """modal22 `_cands_for`: for each template, the marked then the unmarked candidate."""
    return [fill(t, v) for t in templates for v in vals]


def prompted_logodds(lp):  # noqa: ANN001, ANN201
    """modal22 `_score_generic`: [N, 2T] candidate log-probs -> [T, N] log p(marked) - log p(unmarked)."""
    return (lp[:, 0::2] - lp[:, 1::2]).T


def prompted_auroc(lo_by_id: dict[str, Sequence[float]], selected_ids: Sequence[str], fixed_id: str,
                   y: Sequence[int]) -> dict[str, float]:
    """exp 22's cell value: max of the AUROC of the mean of the three selected phrasings' log-odds and the AUROC of
    the fixed label's (score22 `_prompted`; select_freeze `mean_of_selected`, `fixed_label`)."""
    import numpy as np

    from mats.metrics import auroc_labels

    yy = np.asarray(y, int)
    mean3 = np.mean(np.asarray([lo_by_id[i] for i in selected_ids], float), 0)
    m, f = float(auroc_labels(mean3, yy)), float(auroc_labels(np.asarray(lo_by_id[fixed_id], float), yy))
    return {"mean_of_selected": m, "fixed_label": f, "auroc": max(m, f)}


# ---- carrying (exp 14's frozen probe, as lib25 ships it) -------------------------------------------------------------

def frozen_probe_auroc(X_true, R, y, groups, k: int = 5, lam: float = 1e-3, seed: int = 0) -> float:  # noqa: ANN001
    """A probe fit on the true activations (grouped folds), applied frozen to the reconstructions of the held-out
    items; AUROC against the labels."""
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


# ---- results files ---------------------------------------------------------------------------------------------------

def merge(a: dict, b: dict) -> dict:
    """A new dict with a's keys and b's keys (b wins on a shared key); neither input is touched."""
    return {**a, **b}


def cell(d: dict, L: int):  # noqa: ANN201
    """A per-layer value whether the keys are ints (in memory) or strings (after JSON)."""
    return d.get(L, d.get(str(L)))


def table(rows: list[dict], measure: str) -> str:
    """Markdown: one line per row, one column per layer (ascending, the union of the rows' layers), two decimals."""
    layers = sorted({int(L) for r in rows for L in r["layers"]})
    out = ["| row | " + " | ".join(str(L) for L in layers) + " |", "|---|" + "---:|" * len(layers)]
    for r in rows:
        vals = [cell(r.get(measure, {}), L) for L in layers]
        out.append(f"| {r['feature']} | " + " | ".join("" if v is None else f"{v:.2f}" for v in vals) + " |")
    return "\n".join(out)
