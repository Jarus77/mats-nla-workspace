"""Shared helpers for the g4 (talk section 6 + headline) figures. Read-only on experiments/."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style  # noqa: E402

ROOT = style.ROOT
EXP = ROOT / "experiments"
sys.path.insert(0, str(ROOT / "src"))

ROW_LABEL = {
    "py_string": "inside a Python string",
    "js_string": "inside a JavaScript string",
    "latex_math": "inside LaTeX math",
    "py_arg": "call argument (1st vs 3rd)",
    "json_depth": "JSON nesting depth",
    "html_depth": "HTML nesting depth",
    "md_heading": "Markdown heading level",
    "md_list": "list nesting depth",
    "tb_tablecol": "table column (first vs late)",
    "tb_paren": "inside parentheses",
    "code_lang": "code language (control)",
}
MARKDOWN = ("md_heading", "md_list")
READER = {"nla": "NLA", "ol": "Oracle Lens"}
# slide canvas less the savefig padding (2 x 0.08 in), so the saved PNG is at most 9.4 in wide and text keeps its size
FIG = (style.SLIDE[0] - 0.18, style.SLIDE[1] - 0.08)
# short row labels for slide-size (9.4 in) figures
SHORT = {
    "py_string": "Python string", "js_string": "JavaScript string", "latex_math": "LaTeX math",
    "py_arg": "call argument", "json_depth": "JSON nesting depth", "html_depth": "HTML nesting depth",
    "md_heading": "Markdown heading level", "md_list": "list nesting depth", "tb_tablecol": "table column",
    "tb_paren": "inside parentheses", "code_lang": "code language",
}


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def load(p: Path):
    return json.loads(Path(p).read_text())


def hanley_mcneil(a: float, n1: int, n0: int, z: float = 1.96) -> tuple[float, float]:
    """Analytic 95% CI of an AUROC from its value and class counts (Hanley & McNeil 1982). Used only where no
    per-item scores were saved; it ignores clustering, so it is if anything too narrow."""
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    se = math.sqrt(max(var, 0.0))
    return max(0.0, a - z * se), min(1.0, a + z * se)


def cluster_prop_ci(flags, groups, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Proportion with a cluster (item) bootstrap 95% CI. When the proportion is 0, the upper bound is the Wilson
    bound on the number of clusters (a bootstrap of all-zero data has zero width)."""
    flags = np.asarray(flags, float)
    groups = np.asarray(groups)
    p = float(flags.mean())
    ng = len(np.unique(groups))
    if flags.sum() == 0:
        return p, 0.0, style.wilson(0, ng)[2]
    lo, hi = style.bootstrap_ci(lambda f: float(f.mean()), flags, n_boot=n_boot, seed=seed, groups=groups)
    return p, lo, hi


def oof_scores(X, y, groups, k: int = 5, lam: float = 1e-3, seed: int = 0) -> np.ndarray:
    """Out-of-fold decision scores of mats.probes' grouped k-fold logistic probe (the same folds and fit as
    mats.probes.cv_auroc), so the AUROC can be bootstrapped over items."""
    import torch

    from mats.probes import fit_logreg, group_folds

    y = np.asarray(y)
    sc = np.empty(len(y))
    for te in group_folds(np.asarray(groups), k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = fit_logreg(X[tr].float(), torch.as_tensor(y[tr]), lam)
        sc[te] = m.decision(X[te]).cpu().numpy()
    return sc


def auroc_ci(scores, y, groups, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    a = style.auroc(scores, y)
    lo, hi = style.bootstrap_ci(lambda s, t: style.auroc(s, t), scores, y, n_boot=n_boot, seed=seed, groups=groups)
    return a, lo, hi


_TOK = None


def encode(s: str) -> list[int]:
    global _TOK
    if _TOK is None:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    return _TOK(s or "", add_special_tokens=False)["input_ids"]


def text_bag_scores(texts: list[str], y, groups) -> np.ndarray:
    """exp 23/27's readout classifier: binary bag of Qwen tokens (min_df 3), grouped 5-fold logistic, lam 1.0."""
    sys.path.insert(0, str(EXP / "19_natural_bank"))
    from lib19 import bag_of_tokens

    X = bag_of_tokens([encode(t) for t in texts], window=10**6, min_df=3)
    return oof_scores(X, y, groups, k=5, lam=1.0)


def check(quantity: str, computed: float, record: float, where: str, tol: float = 0.005) -> dict:
    return {"quantity": quantity, "computed": round(float(computed), 4), "record": record, "where": where,
            "match": bool(abs(float(computed) - float(record)) <= tol)}


def label_dashed(ax, x=None, y=None, text="", color=None, **kw):
    color = color or style.C["muted"]
    if x is not None:
        ax.axvline(x, color=style.C["line"], lw=1.5, ls="--", zorder=0)
    if y is not None:
        ax.axhline(y, color=style.C["line"], lw=1.5, ls="--", zorder=0)
