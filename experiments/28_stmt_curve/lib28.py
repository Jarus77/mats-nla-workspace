"""Pure helpers for exp 28 (the statement shift as a graded instrument, on every row that has an NLA stated rate).

No model code and no Modal import here: the laptop steps (collect_rows, paraphrase parsing, the statistics) and the
GPU stage (the loaders, class_direction, statement_shift) share this file. Tested in test_lib28.py.

Sources. A "row" is (experiment, feature). The exp 12 rows are exp 07's 14 concepts (exp 12 judged exp 07's readouts;
its activations and items live under /results/07_template_mining), the exp 13 rows are its seven judged rows, and exps
18, 19 and 22 are the prediction-test banks. Every source stores activations as {"ids": [...], "acts": {L: [N, d]}}
(exp 07 `tm_modal`, exp 13 `modal13`, exps 18/19/22 `instr1`); the items differ: exps 07/13 keep them inside
gen_ol_{feature}.json["items"] (exp 14's `_row`), exps 18/19/22 in items_base.json filtered by feature.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

_PARENTS = Path(__file__).resolve().parents
REPO = _PARENTS[2] if len(_PARENTS) > 2 else None   # None inside the Modal container (/root/lib28.py), where the sibling modules ship flat
LAYER = 42
SPAN_TOKENS = 32
N_PARA = 8
STATED_LINE = 0.30              # the memo's "NLA states iff stmt >= 0.30" line; AUROC target is stated >= this
BAND_EDGES = (0.10, 0.25)       # the memo's three bands of stmt
N_PERM = 20000                  # the memo's permutation count

# Concept type, declared before the run. Self-inducing = a whole-text property a short text can itself exhibit (exp
# 12's judge "exhibited" column is non-zero only for language, code language, caps and tense; the derivation's C3
# calls language and code language self-inducing). Everything else (position in text, nesting, upcoming-word
# syntax, the stance word) is structural.
SELF_INDUCING: frozenset[tuple[str, str]] = frozenset({("12", "ij2_lang"), ("12", "bk_code"), ("12", "bk_caps"),
                                                       ("12", "nj2_tense"), ("18", "ij2_lang"), ("22", "code_lang")})
# The memo's four counter-rows to the 0.30 rule and its language rows (bk_code added: exp 12's code-language row).
COUNTER_ROWS: frozenset[tuple[str, str]] = frozenset({("13", "tb_tablecol"), ("13", "tb_paren"), ("18", "call_arg"),
                                                      ("12", "bk_listidx")})
LANGUAGE_ROWS: frozenset[tuple[str, str]] = frozenset({("12", "ij2_lang"), ("18", "ij2_lang"), ("22", "code_lang"),
                                                       ("12", "bk_code")})

_E = (REPO if REPO is not None else Path("/root")) / "experiments"   # laptop-only paths; never opened in the container
SOURCES: dict[str, dict] = {
    "12": {"spec": _E / "07_template_mining" / "feature_spec.py",
           "score": (_E / "12_concept_recall" / "results" / "recall_table.json", "recall"),
           "vol": "/results/07_template_mining", "items_fmt": "gen_ol", "acts_name": "acts_{f}.pt", "instr": None, "gens": None,
           "prior14": _E / "14_ar_blindspot" / "results" / "analysis14_{f}.json"},
    "13": {"spec": _E / "13_new_rows" / "feature_spec.py",
           "score": (_E / "13_new_rows" / "results" / "recall_table.json", "recall"),
           "vol": "/results/13_new_rows", "items_fmt": "gen_ol", "acts_name": "acts_{f}.pt", "instr": None, "gens": None,
           "prior14": _E / "14_ar_blindspot" / "results" / "analysis14_{f}.json"},
    "18": {"spec": _E / "18_prediction_test" / "feature_spec18.py",
           "score": (_E / "18_prediction_test" / "results" / "score18.json", "score"),
           "vol": "/results/18_prediction_test", "items_fmt": "items_base", "acts_name": "acts18_{f}.pt",
           "instr": _E / "18_prediction_test" / "results" / "instruments18.json",
           "gens": _E / "18_prediction_test" / "results" / "gen_nla_{f}.json"},
    "19": {"spec": _E / "19_natural_bank" / "feature_spec19.py",
           "score": (_E / "19_natural_bank" / "results" / "score19.json", "score"),
           "vol": "/results/19_natural_bank", "items_fmt": "items_base", "acts_name": "acts19_{f}.pt",
           "instr": _E / "19_natural_bank" / "results" / "instruments19.json",
           "gens": _E / "19_natural_bank" / "results" / "gen_nla_{f}.json"},
    "22": {"spec": _E / "22_public_bank" / "feature_spec22.py",
           "score": (_E / "22_public_bank" / "results" / "score22.json", "score"),
           "vol": "/results/22_public_bank", "items_fmt": "items_base", "acts_name": "acts22_{f}.pt",
           "instr": _E / "22_public_bank" / "results" / "instruments22.json",
           "gens": _E / "22_public_bank" / "results" / "gen_nla_{f}.json"},
}
PRIOR_KEYS = ("readable", "span", "span_gap", "stmt", "used_z")


# ================================================================================================ the formula
def fill(template: dict, value: str) -> str:
    f = template["fillers"]
    return template["template"].replace("{v}", f["*"] if "*" in f else f[value])


def class_direction(X: torch.Tensor, y) -> tuple[torch.Tensor, float]:  # noqa: ANN001
    """The true class direction d = mean(marked) - mean(unmarked) (unnormalised) and its norm, the gap."""
    yb = torch.as_tensor(np.asarray(y).astype(bool))
    if int(yb.sum()) == 0 or int((~yb).sum()) == 0:
        raise ValueError("both classes must be present")
    d = X[yb].float().mean(0) - X[~yb].float().mean(0)
    return d, float(d.norm())


def statement_shift(rec_a: torch.Tensor, rec_b: torch.Tensor, d: torch.Tensor) -> float:
    """((rec(A) - rec(B)) . d_hat) / gap: how far the reconstructor moves along the true class direction between the
    two statements, as a fraction of the true class gap. Identical to exp 14's ar_lib.statement_shift['ratio'] and
    to the `stmt` of exps 18/19/22 (modal22 c7lx)."""
    gap = d.float().norm().clamp_min(1e-9)
    u = d.float() / gap
    return float(((rec_a.float() - rec_b.float()) @ u) / gap)


def shift_summary(fixed: float, paras: list[float]) -> dict:
    p = np.asarray(paras, float)
    return {"fixed": float(fixed), "n_para": int(len(p)), "para_mean": float(p.mean()), "para_min": float(p.min()),
            "para_max": float(p.max()), "para_sd": float(p.std(ddof=1)) if len(p) > 1 else float("nan"),
            "para_range": float(p.max() - p.min())}


# ================================================================================================ loaders
def acts_from_blob(blob: dict, layer: int, ids: list[str] | None) -> torch.Tensor:
    """[N, d] fp32 layer activations from a {"ids", "acts": {L: tensor}} blob; layer keys may be int or str."""
    acts = blob["acts"]
    key = layer if layer in acts else str(layer)
    if key not in acts:
        raise KeyError(f"layer {layer} not in the blob (has {sorted(map(str, acts))})")
    if ids is not None and list(blob["ids"]) != list(ids):
        raise ValueError("activation order differs from the item order")
    return acts[key].float()


def items_from(obj, fmt: str, feature: str) -> list[dict]:  # noqa: ANN001
    """Items of one feature: `gen_ol` = a gen_ol_{feature}.json (exps 07/13), `items_base` = an items_base.json list."""
    if fmt == "gen_ol":
        if obj.get("feature") != feature:
            raise ValueError(f"gen_ol file is for {obj.get('feature')!r}, not {feature!r}")
        return list(obj["items"])
    if fmt == "items_base":
        return [it for it in obj if it["feature"] == feature]
    raise ValueError(f"unknown items format {fmt!r}")


def select_split(items: list[dict], marked: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Selection-split indices, labels (1 = the marked value) and cluster groups, as modal22's `_sel`."""
    idx = [i for i, it in enumerate(items) if it["split"] == "select"]
    y = np.array([items[i]["value"] == marked for i in idx], int)
    return idx, y, np.array([items[i]["cluster"] for i in idx])


# ================================================================================================ statistics
def _perm_p(stat, x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> float:  # noqa: ANN001
    obs = stat(x, y)
    rng = np.random.default_rng(seed)
    null = np.array([stat(x, rng.permutation(y)) for _ in range(n_perm)])
    return float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))


def _rank(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(a)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman_perm(x, y, n_perm: int = N_PERM, seed: int = 28) -> dict:  # noqa: ANN001
    """Spearman rho with a two-sided permutation p (y shuffled); NaN rho and p for n < 3."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"rho": float("nan"), "p": float("nan"), "n": int(len(x))}
    return {"rho": _spearman(x, y), "p": _perm_p(_spearman, x, y, n_perm, seed), "n": int(len(x))}


def pearson_perm(x, y, n_perm: int = N_PERM, seed: int = 28) -> dict:  # noqa: ANN001
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": float("nan"), "p": float("nan"), "n": int(len(x))}
    return {"r": _pearson(x, y), "p": _perm_p(_pearson, x, y, n_perm, seed), "n": int(len(x))}


def auroc_threshold(shift, stated, line: float = STATED_LINE) -> float:  # noqa: ANN001
    """AUROC of the shift for the event stated >= line (the memo's 0.71 over 30 rows)."""
    from mats.metrics import auroc_labels
    shift, stated = np.asarray(shift, float), np.asarray(stated, float)
    lab = (stated >= line).astype(int)
    if lab.sum() == 0 or lab.sum() == len(lab):
        return float("nan")
    return float(auroc_labels(shift, lab))


def logistic_fit(X, k, n, iters: int = 100, tol: float = 1e-9) -> dict:  # noqa: ANN001
    """Binomial logistic regression by IRLS (numpy; no statsmodels in the venv): k successes of n trials per row,
    logit p = b0 + X b. Returns coefficients, log-likelihood (with the binomial constant), AIC and k_params."""
    from scipy.special import gammaln
    X = np.asarray(X, float).reshape(len(k), -1)
    k, n = np.asarray(k, float), np.asarray(n, float)
    A = np.column_stack([np.ones(len(k)), X])
    b = np.zeros(A.shape[1])
    converged = False
    for _ in range(iters):
        eta = A @ b
        p = 1 / (1 + np.exp(-eta))
        w = n * p * (1 - p) + 1e-12
        z = eta + (k - n * p) / w
        b_new = np.linalg.solve(A.T @ (w[:, None] * A) + 1e-10 * np.eye(A.shape[1]), A.T @ (w * z))
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            converged = True
            break
        b = b_new
    p = np.clip(1 / (1 + np.exp(-(A @ b))), 1e-12, 1 - 1e-12)
    ll = float(np.sum(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + k * np.log(p) + (n - k) * np.log(1 - p)))
    return {"coef": b.tolist(), "loglik": ll, "aic": 2 * A.shape[1] - 2 * ll, "k_params": int(A.shape[1]),
            "converged": bool(converged)}


def band_table(rows: list[dict], xkey: str, ykey: str, edges: tuple[float, float] = BAND_EDGES) -> list[dict]:
    """The memo's band summary: rows with x < edges[0], edges[0] <= x < edges[1], x >= edges[1]."""
    lo, hi = edges
    bands = [(f"< {lo:.2f}", lambda v: v < lo), (f"{lo:.2f} to {hi:.2f}", lambda v: lo <= v < hi), (f">= {hi:.2f}", lambda v: v >= hi)]
    out = []
    for name, keep in bands:
        ys = [r[ykey] for r in rows if keep(r[xkey])]
        out.append({"band": name, "n": len(ys), "mean_stated": float(np.mean(ys)) if ys else float("nan"),
                    "min_stated": float(min(ys)) if ys else float("nan"), "max_stated": float(max(ys)) if ys else float("nan")})
    return out


# ================================================================================================ paraphrases
REGISTERS: tuple[str, ...] = (
    "declarative, the fixed sentence's own shape with different words",
    "'The text is ...' or 'This text ...' (a description of the text as a whole)",
    "'We are currently ...' or 'We are now ...' (first person plural, present progressive)",
    "short: at most six words, no article at the start if possible",
    "long: two clauses joined by a comma or 'and', about twenty words",
    "a note a code reviewer or editor might write about the position, in the third person",
    "'The model is reading ...' or 'The reader is at ...' (the activation's point of view)",
    "passive voice or a nominal phrase ('Position: ...', 'Current state: ...')",
)

PAIR_SCHEMA = {
    "type": "object",
    "properties": {"pairs": {"type": "array", "items": {
        "type": "object",
        "properties": {"register": {"type": "string"}, "a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["register", "a", "b"], "additionalProperties": False}}},
    "required": ["pairs"], "additionalProperties": False,
}

_PROMPT = """You are writing paraphrases of a short statement for a controlled measurement. The statement describes
a property of the position a language model is reading in a text. The property has two values.

Property: {desc}
Values: A = "{va}", B = "{vb}".
Fixed statement pair (value A / value B):
  A: {fa}
  B: {fb}

Write {n} NEW paraphrase pairs. Each pair states the SAME property as the fixed pair, with value A in sentence "a"
and value B in sentence "b", and the two sentences of a pair must be word-for-word identical except for the words
that name the value (so that the pair differs only in the value, exactly as the fixed pair does). Use each register
below once, in this order (pair i uses register i):
{registers}

Rules: plain English prose, one sentence each, no quotation marks inside the sentences, no numbering, no
explanations; do not copy the fixed pair verbatim; every pair must differ from every other pair; keep the sense of
the value (do not weaken, hedge or negate it).

Return JSON: {{"pairs": [{{"register": "<label>", "a": "<sentence>", "b": "<sentence>"}}, ...]}} with exactly {n} entries."""


def paraphrase_prompt(row: dict) -> str:
    regs = "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(REGISTERS))
    return _PROMPT.format(desc=row["desc"], va=row["values"][0], vb=row["values"][1], fa=row["fixed"][0], fb=row["fixed"][1],
                          n=N_PARA, registers=regs)


def parse_pairs(raw: str | None, fixed: tuple[str, str] | list[str]) -> list[list[str]] | None:
    """Exactly N_PARA pairs, each non-empty, a != b, all pairs distinct, none equal to the fixed pair; else None."""
    if not raw:
        return None
    try:
        pairs = json.loads(raw).get("pairs", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(pairs, list) or len(pairs) != N_PARA:
        return None
    out: list[list[str]] = []
    for p in pairs:
        if not isinstance(p, dict) or not isinstance(p.get("a"), str) or not isinstance(p.get("b"), str):
            return None
        a, b = p["a"].strip(), p["b"].strip()
        if not a or not b or a == b or [a, b] == list(fixed) or [a, b] in out:
            return None
        out.append([a, b])
    return out


# ================================================================================================ collect_rows
def load_spec(path: Path):  # noqa: ANN201
    """Import a feature_spec*.py by path under a unique module name (several sources share the name feature_spec)."""
    name = f"spec28_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stated_rates(path: Path, kind: str) -> dict[str, tuple[float, int | None]]:
    """feature -> (NLA stated rate, n readouts or None). `recall` = exps 12/13 recall_table.json (per-feature
    {"nla": {"stated_rate", "n_samples"}}); `score` = exps 18/19/22 score*.json ({"rows": [{"feature", "observed":
    {"nla_states"}}]}, n filled later from the gen_nla file)."""
    d = json.loads(Path(path).read_text())
    if kind == "recall":
        return {f: (float(v["nla"]["stated_rate"]), int(v["nla"]["n_samples"])) for f, v in d.items()
                if v.get("nla", {}).get("stated_rate") is not None}
    if kind == "score":
        return {r["feature"]: (float(r["observed"]["nla_states"]), None) for r in d["rows"]
                if r.get("observed", {}).get("nla_states") is not None}
    raise ValueError(kind)


def _prior(path: Path | None) -> dict[str, dict]:
    if path is None or not Path(path).exists():
        return {}
    out = {}
    for r in json.loads(Path(path).read_text()):
        ins = r.get("instr") or {}
        out[r["feature"]] = {k: (None if ins.get(k) is None or (isinstance(ins[k], float) and math.isnan(ins[k])) else float(ins[k]))
                             for k in PRIOR_KEYS}
    return out


def _prior14(pattern: Path | None, feature: str) -> dict[str, float]:
    """Exp 14's NLA-ruler fixed-statement shift for its four rows (the values the memo used), else {}."""
    if pattern is None:
        return {}
    p = Path(str(pattern).format(f=feature))
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    v = d.get("nla", {}).get("stmt", {}).get(f"nla:{feature}:fixed")
    return {"stmt": float(v["ratio"]), "stmt_source": "exp14"} if v else {}


def _n_from_gens(gens: Path | None, feature: str) -> int | None:
    if gens is None:
        return None
    p = Path(str(gens).format(f=feature))
    if not p.exists():
        return None
    g = json.loads(p.read_text())["nla_gens"]
    return int(sum(len(v) for v in g.values()))


def collect_rows(sources: dict[str, dict] = SOURCES, types: frozenset | set = SELF_INDUCING) -> tuple[list[dict], list[dict]]:
    """One record per (experiment, feature) with an NLA stated rate and a fixed NLA statement pair; plus the rows
    with a stated rate that could not be collected, with the reason."""
    rows, missing = [], []
    for exp, src in sources.items():
        spec = load_spec(Path(src["spec"]))
        path, kind = src["score"]
        if not Path(path).exists():
            missing.append({"experiment": exp, "feature": "*", "reason": f"scorecard {path} not found"})
            continue
        prior = _prior(src.get("instr"))
        for feature, (stated, n) in stated_rates(Path(path), kind).items():
            feat = spec.FEATURES.get(feature)
            if feat is None:
                missing.append({"experiment": exp, "feature": feature, "reason": "not in the spec's FEATURES"})
                continue
            tmpl = spec.FIXED.get("nla", {}).get(feature)
            if tmpl is None:
                missing.append({"experiment": exp, "feature": feature, "reason": "no FIXED['nla'] statement pair in the spec"})
                continue
            a, b = feat["values"]
            if n is None:
                n = _n_from_gens(src.get("gens"), feature)
            rows.append({
                "experiment": exp, "feature": feature, "key": f"{exp}_{feature}", "values": [a, b], "desc": feat.get("desc", ""),
                "fixed": [fill(tmpl, a), fill(tmpl, b)], "stated": stated, "n_readouts": n,
                "type": "self_inducing" if (exp, feature) in types else "structural",
                "counter_row": (exp, feature) in COUNTER_ROWS, "language_row": (exp, feature) in LANGUAGE_ROWS,
                "items": {"vol": f"{src['vol']}/{'gen_ol_' + feature + '.json' if src['items_fmt'] == 'gen_ol' else 'items_base.json'}",
                          "fmt": src["items_fmt"]},
                "acts": {"vol": f"{src['vol']}/{src['acts_name'].format(f=feature)}", "layer": LAYER},
                "prior": prior.get(feature) or _prior14(src.get("prior14"), feature),
            })
    return rows, missing
