"""Tables and the figure from results/gen29_*, results/recon29_* and results/judged/* (laptop; matplotlib, no seaborn).
  results/table29.md    one table per measure (rows x checkpoints in training order): states (judge), names (strict
                        regex on the k=1 readouts), carries (critic + frozen probe), textbag (bag-of-tokens AUROC on
                        the k=1 readout text), plus fve and the class-direction cosine; the four pre-registered checks
  results/table29.json  the same numbers
  results/fig29.png     four panels, x = checkpoint (textnla, av_base, iter_*), one line per row, code_lang highlighted
    .venv/bin/python analyze29.py [results_dir] [--tokenizer qwen|words]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "12_concept_recall"))
from lib29 import (CONTROL_ROW, FINAL_CKPT, MEASURES, OUT_FILE, READABLE_ROWS, ROWS, STRING_ROWS, STRUCTURAL_ROWS,  # noqa: E402
                   CKPT_BASE, CKPT_TEXT, MARKED, check_predictions, checkpoint_order, mean_delta, merge_nested,
                   names_rate, rate_table, select_split, textbag_auroc, word_tokens)
from recall_lib import aggregate  # noqa: E402

ITEMS22 = ROOT / "experiments" / "22_public_bank" / "results" / "items_base.json"
RECON25 = ROOT / "experiments" / "25_public_exhibits" / "results"
TITLES = {"states": "States (judge): share of the 8 mining readouts per item in which Gemini 3.8 Flash finds an explicit claim about the row's property (exp 22's cell)",
          "names": "Names (strict regex): share of the k=1 selection readouts in which the row's state regex fires (exp 23/26 phrasing)",
          "carries": "Carries (critic): k=1 readouts rebuilt by the final NLA reconstructor, scored by the frozen probe against the true layer-42 activations (exp 14/25 measure)",
          "textbag": "Text bag: grouped 5-fold AUROC of a bag-of-tokens classifier on the k=1 readout text alone (exp 20/23 cross-check)",
          "fve": "FVE of the critic rebuilds of the k=1 readouts against the true activations (per-row predict-the-mean baseline)",
          "cos": "Cosine between the rebuilt and the true class-mean direction"}
PANELS = ("states", "names", "carries", "textbag")
YLIMS = {"states": (-0.02, 1.02), "names": (-0.02, 1.02), "carries": (0.4, 1.0), "textbag": (0.4, 1.0)}
GRID = "#d8d7d2"
CONTROL_COLOR = "#eb6834"
ROW_STYLE = {"py_string": ("#2a78d6", "o", "-"), "js_string": ("#1baf7a", "s", "-"), "latex_math": ("#8e5bd6", "^", "-"),
             "py_arg": ("#b8860b", "v", "-"), "json_depth": ("#2a9fb8", "D", "-"),
             "md_heading": ("#9a9a9a", "x", ":"), "md_list": ("#6f6f6f", "+", ":")}


def _ckpts(res: Path) -> list[str]:
    names = set()
    for p in res.glob("gen29_*.json"):
        stem = p.name[len("gen29_"):-len(".json")]
        for row in ROWS:
            if stem.endswith("_" + row):
                names.add(stem[: -len(row) - 1])
    return checkpoint_order(names)


def _tokeniser(kind: str):  # noqa: ANN202
    if kind == "qwen":
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
            return "qwen", (lambda s: tok(s or "", add_special_tokens=False)["input_ids"])
        except Exception as e:  # noqa: BLE001
            print(f"Qwen tokenizer unavailable ({type(e).__name__}); falling back to word tokens", flush=True)
    return "words", word_tokens


def collect(res: Path, tokenizer: str = "qwen") -> dict:
    items_all = json.loads(ITEMS22.read_text())
    ckpts = _ckpts(res)
    tok_name, enc = _tokeniser(tokenizer)
    out = {m: {} for m in (*MEASURES, "fve", "cos")}
    for ck in ckpts:
        for m in out:
            out[m][ck] = {}
        for row in ROWS:
            gp, rp, jp = res / OUT_FILE("gen", ck, row), res / OUT_FILE("recon", ck, row), res / "judged" / f"{ck}_{row}.json"
            if gp.exists():
                gen = json.loads(gp.read_text())
                items = [it for it in items_all if it["feature"] == row]
                sel, y, groups = select_split(items, MARKED[row])
                texts = [gen["sel"].get(items[i]["id"]) or "(empty)" for i in sel]
                out["names"][ck][row] = names_rate(texts, row)
                out["textbag"][ck][row] = textbag_auroc([enc(t) for t in texts], y, groups)
            if rp.exists():
                rec = json.loads(rp.read_text())
                out["carries"][ck][row] = rec["auroc"]
                out["fve"][ck][row] = rec["fve"]
                out["cos"][ck][row] = rec["class_direction_cos"]
            if jp.exists():
                d = json.loads(jp.read_text())
                out["states"][ck][row] = aggregate(d["judged"], d["values"])["stated_rate"]
    return {"checkpoints": ckpts, "tokenizer": tok_name, **out}


def exp25_check(res_summary: dict) -> list[str]:
    """exp 25 measured the same critic AUROC of iter_000400's k=1 explanations (recon25 `nla_readout`)."""
    lines = []
    for row in ROWS:
        p = RECON25 / f"recon25_{row}.json"
        here = res_summary["carries"].get(FINAL_CKPT, {}).get(row)
        if p.exists() and here is not None:
            v = json.loads(p.read_text())["auroc"]["nla_readout"]
            lines.append(f"- {row}: exp 25 nla_readout {v:.2f}, this run {here:.2f} (new samples, seed 0)")
    return lines


def render_md(s: dict) -> str:
    ck = s["checkpoints"]
    md = [f"Checkpoints in training order: {', '.join(ck)}. Text bag tokeniser: {s['tokenizer']}.", ""]
    for m in (*MEASURES, "fve", "cos"):
        md += [f"### {m}", "", TITLES[m], "", rate_table(s[m], ROWS, ck), ""]
    md += ["### Changes from av_base to " + FINAL_CKPT, ""]
    for m in MEASURES:
        d_struct = mean_delta(s[m], CKPT_BASE, FINAL_CKPT, STRUCTURAL_ROWS)
        d_ctrl = mean_delta(s[m], CKPT_BASE, FINAL_CKPT, [CONTROL_ROW])
        d_text = mean_delta(s[m], CKPT_BASE, CKPT_TEXT, READABLE_ROWS)
        f = lambda v: "n/a" if v is None else f"{v:+.2f}"  # noqa: E731
        md.append(f"- {m}: structural rows mean {f(d_struct)}; code_lang {f(d_ctrl)}; textnla minus av_base on the readable rows {f(d_text)}")
    md += ["", "### Pre-registered checks (RESULTS.md)", ""]
    for k, v in check_predictions(s, final=FINAL_CKPT).items():
        md.append(f"- {k}: {'pending' if v is None else ('holds' if v else 'FAILS')}")
    chk = exp25_check(s)
    if chk:
        md += ["", "### Consistency with exp 25 (iter_000400 carrying)", "", *chk]
    return "\n".join(md) + "\n"


def fig(s: dict, out: Path) -> None:
    ck = s["checkpoints"]
    xs = list(range(len(ck)))
    fig_, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=150)
    for ax, m in zip(axes.ravel(), PANELS):
        for row in ROWS:
            pts = [(x, s[m].get(c, {}).get(row)) for x, c in zip(xs, ck)]
            pts = [(x, v) for x, v in pts if v is not None]
            if not pts:
                continue
            if row == CONTROL_ROW:
                ax.plot(*zip(*pts), color=CONTROL_COLOR, lw=3, marker="o", ms=7, label="code_lang (positive control)", zorder=4)
            else:
                col, mk, ls = ROW_STYLE[row]
                ax.plot(*zip(*pts), color=col, lw=1.3, ls=ls, marker=mk, ms=4, label=row, zorder=3)
        if m == "states":
            ax.axhline(0.25, color=GRID, lw=1, ls="--")
        if m in ("carries", "textbag"):
            for y in (0.7, 0.8):
                ax.axhline(y, color=GRID, lw=1, ls="--")
        ax.set_ylim(*YLIMS[m])
        ax.set_xticks(xs)
        ax.set_xticklabels(ck, rotation=30, ha="right", fontsize=9)
        ax.set_title(m, loc="left", fontsize=11)
        ax.grid(axis="y", color=GRID, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    fig_.suptitle("Exp 29: naming and carrying along the NLA's RL trajectory (exp 22 public bank; textnla = base Qwen on the text)", fontsize=12)
    fig_.tight_layout()
    fig_.savefig(out, bbox_inches="tight")
    plt.close(fig_)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=str(HERE / "results"))
    ap.add_argument("--tokenizer", default="qwen", choices=("qwen", "words"))
    a = ap.parse_args()
    res = Path(a.results).resolve()
    s = collect(res, a.tokenizer)
    if not s["checkpoints"]:
        raise SystemExit(f"no gen29_*.json under {res}: run run29.sh first")
    (res / "table29.json").write_text(json.dumps(s, indent=1))
    md = render_md(s)
    (res / "table29.md").write_text(md)
    fig(s, res / "fig29.png")
    print(md)


if __name__ == "__main__":
    main()
