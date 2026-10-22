"""10 operand gate, stage 2a helpers (pure; CPU-testable): few-shot bank, answer parsing, cross-fit
directions and q90 doses, centred steering vectors, the leak and G4 summaries.

Stage 2a is model-side only (no NLA / Oracle Lens). Conventions follow prereg §4.4 and 04 bf_causal:
centred steering h' = h - (v.h) v + s*Δ v at L42, Δ = q90 - median of the natural projection on the
training fold (2-fold cross-fit by item), null bands of random directions at the same ‖Δ‖ reported as z.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

N_SHOTS = 3
ANS_RE = re.compile(r"^\s*(\d+)")
LABEL_WORDS = ("number", "digit", "operand")
K_JC_LEAK, K_JC_G4, K_ISO_G4 = 100, 20, 20


# ----------------------------------------------------------------------------- few-shot bank


def exemplars(a: int, b: int, k: int = N_SHOTS, seed: int = 0, lo: int = 2, hi: int = 99) -> list[tuple]:
    """k deterministic '(x, op, y, z)' exemplars with correct answers, operands disjoint from {a, b},
    positive results (as in the bank, answers >= 1)."""
    rng = np.random.default_rng([seed, a, b])
    out: list[tuple] = []
    while len(out) < k:
        x, y = int(rng.integers(lo, hi + 1)), int(rng.integers(1, hi + 1))
        op = "+" if rng.random() < 0.5 else "-"
        if op == "-" and x <= y:
            continue
        if {x, y} & {a, b}:
            continue
        out.append((x, op, y, x + y if op == "+" else x - y))
    return out


def fewshot_text(item: dict, seed: int = 0) -> str:
    ex = exemplars(item["a"], item["b"], seed=seed)
    return "\n".join(f"{x} {op} {y} = {z}" for x, op, y, z in ex) + "\n" + item["text"]


def parse_answer(s: str) -> int | None:
    m = ANS_RE.match(s)
    return int(m.group(1)) if m else None


def first_digit(n: int) -> int:
    return int(str(abs(int(n)))[0])


# ----------------------------------------------------------------------------- directions and doses


def folds(n: int, seed: int = 0) -> np.ndarray:
    """2-fold assignment by item (cross-fit: steer fold f with directions fitted on 1 - f)."""
    perm = np.random.default_rng(seed).permutation(n)
    f = np.empty(n, dtype=int)
    f[perm] = np.arange(n) % 2
    return f


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def project_out_rows(Q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return v - (v @ Q) @ Q.T


def q90_dose(X: torch.Tensor, v: torch.Tensor) -> float:
    from mats.capture import doses_from_projection

    return float(doses_from_projection((X.float() @ v.float()).cpu(), quantiles=(0.9,), extra_multiple=None)[0])


def centred(h: torch.Tensor, v: torch.Tensor, a: float) -> torch.Tensor:
    """h - (v.h) v + a v for rows of h (v unit)."""
    hf, vf = h.float(), v.float().to(h.device)
    return hf - (hf @ vf)[:, None] * vf + a * vf


def random_dirs(d: int, k_jc: int, k_iso: int, Q: torch.Tensor | None, seed: int) -> torch.Tensor:
    """[k_jc + k_iso, d] unit rows: the first k_jc with span(Q) projected out, the rest isotropic."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(k_jc + k_iso, d, generator=g)
    if Q is not None and k_jc:
        Qc = Q.cpu().float()
        R[:k_jc] = R[:k_jc] - (R[:k_jc] @ Qc) @ Qc.T
    return R / R.norm(dim=1, keepdim=True)


# ----------------------------------------------------------------------------- summaries


def effect_summary(m: dict[str, np.ndarray], names: list[str], null_names: dict[str, list[str]]) -> dict:
    """04 _summ: per-item effect e_i = (m+ - m-)/2 for each named direction, placed against the mean effects
    of each null family (null_band, z)."""
    from mats.metrics import null_band

    def eff(key: str) -> np.ndarray:
        return (m[key + "+"] - m[key + "-"]) / 2

    out: dict = {}
    bands = {fam: np.array([np.nanmean(eff(k)) for k in keys]) for fam, keys in null_names.items()}
    out["null_abs_max"] = {fam: float(np.nanmax(np.abs(v))) for fam, v in bands.items()}
    rng = np.random.default_rng(0)
    for nm in names:
        e = eff(nm)
        ok = np.isfinite(e)
        E = float(np.nanmean(e))
        boot = [np.nanmean(e[rng.integers(0, len(e), len(e))]) for _ in range(2000)]
        out[nm] = {"effect": E, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                   "paired_win": float(np.mean(m[nm + "+"][ok] > m[nm + "-"][ok])), "n_valid": int(ok.sum()),
                   **{f"band_{fam}": null_band(E, v) for fam, v in bands.items()}}
    return out


def drop_summary(m: dict[str, np.ndarray], names: list[str], null_names: dict[str, list[str]]) -> dict:
    """Unsigned damage: mean(m(cond) - m(none)) averaged over the two signs, against the null families."""
    from mats.metrics import null_band

    def drop(key: str) -> float:
        return float(np.nanmean(np.concatenate([m[key + "+"] - m["none"], m[key + "-"] - m["none"]])))

    bands = {fam: np.array([drop(k) for k in keys]) for fam, keys in null_names.items()}
    return {nm: {"drop": drop(nm), **{f"band_{fam}": null_band(drop(nm), v) for fam, v in bands.items()}}
            for nm in names}


def flip_summary(ans: dict[str, list], names: list[str], null_names: dict[str, list[str]]) -> dict:
    """Greedy-answer flip rate vs the unsteered answer (mean over items and both signs), against the nulls."""
    from mats.metrics import null_band

    base = np.array([x if x is not None else -10**9 for x in ans["none"]])

    def rate(key: str) -> float:
        r = []
        for s in ("+", "-"):
            a = np.array([x if x is not None else -10**9 for x in ans[key + s]])
            r.append(float(np.mean(a != base)))
        return float(np.mean(r))

    bands = {fam: np.array([rate(k) for k in keys]) for fam, keys in null_names.items()}
    return {nm: {"flip_rate": rate(nm), **{f"band_{fam}": null_band(rate(nm), v) for fam, v in bands.items()}}
            for nm in names}


def signed_value_shift(ans: dict[str, list], key: str, sign: np.ndarray) -> np.ndarray:
    """Per-item (value(+) - value(-))/2 * expected sign; NaN where either parse failed or sign is 0."""
    p = np.array([x if x is not None else np.nan for x in ans[key + "+"]], dtype=float)
    q = np.array([x if x is not None else np.nan for x in ans[key + "-"]], dtype=float)
    s = sign.astype(float)
    s[s == 0] = np.nan
    return (p - q) / 2 * s


def leak_summary(S: dict[str, np.ndarray], names: list[str], null_keys: list[str]) -> dict:
    """Prereg §4.4.3: paired +/- contrast of the J-Lens set score S against K J-complement contrasts, two-sided."""
    from mats.metrics import null_band, paired_stats

    nulls = [paired_stats(S[k + "+"], S[k + "-"]) for k in null_keys]
    out = {}
    for nm in names:
        r = paired_stats(S[nm + "+"], S[nm + "-"])
        b = {key: null_band(r[key], [x[key] for x in nulls]) for key in ("auroc", "mean_shift")}
        out[nm] = {"auroc": r["auroc"], "mean_shift": r["mean_shift"], "band_auroc": b["auroc"],
                   "band_mean_shift": b["mean_shift"],
                   "fires95": bool(b["auroc"]["outside_two_sided"] or b["mean_shift"]["outside_two_sided"]),
                   "z_auroc": b["auroc"]["z"], "z_mean_shift": b["mean_shift"]["z"]}
    return out


def digit_token_sets(tok: Any) -> dict[str, list[int]]:
    """Content set: single-token digits 0-9 (bare and with a leading space) plus any single-token 10-99;
    label set: number/digit/operand variants."""
    from mats.readers.jlens import token_variants

    content = set(token_variants(tok, [str(d) for d in range(10)]))
    multi = []
    for n in range(10, 100):
        for s in (str(n), " " + str(n)):
            e = tok.encode(s, add_special_tokens=False)
            if len(e) == 1:
                content.add(e[0])
                multi.append(s)
    return {"content": sorted(content), "label": token_variants(tok, list(LABEL_WORDS)), "multi_digit_single_tokens": multi}
