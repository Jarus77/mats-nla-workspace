"""bk_exp10_operand: operand digits at '=' are decodable by a probe, yet no edit there changes the model's answer."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

E10 = S.ROOT / "experiments" / "10_operand_gate"


def main():
    rows = [json.loads(l) for l in (E10 / "data" / "arithmetic.jsonl").read_text().splitlines() if l.strip()]
    gate = json.loads((E10 / "results" / "gate.json").read_text())
    truth = {"a_tens": [r["a"] // 10 for r in rows], "a_units": [r["a"] % 10 for r in rows],
             "b_tens": [r["b"] // 10 for r in rows], "b_units": [r["b"] % 10 for r in rows]}
    acc = {}
    for s, t in truth.items():
        pred = np.array(gate["slots"][s]["G1"]["pred"])
        k = int((pred == np.array(t)).sum())
        acc[s] = S.wilson(k, len(t)) + (k, len(t))
    pa = np.array(gate["slots"]["a_tens"]["G1"]["pred"]) * 10 + np.array(gate["slots"]["a_units"]["G1"]["pred"])
    pb = np.array(gate["slots"]["b_tens"]["G1"]["pred"]) * 10 + np.array(gate["slots"]["b_units"]["G1"]["pred"])
    ka = int((pa == np.array([r["a"] for r in rows])).sum()); kb = int((pb == np.array([r["b"] for r in rows])).sum())
    ex = {"a": S.wilson(ka, 600) + (ka, 600), "b": S.wilson(kb, 600) + (kb, 600)}

    g4 = torch.load(E10 / "results" / "stage2a_g4" / "g4_raw.pt", weights_only=False, map_location="cpu")
    base = np.array(g4["ans"]["none"])
    flips = {}
    groups = [("own-digit class mean, J-removed", "cm_own_jrem"), ("own-digit class mean, raw", "cm_own_raw"),
              ("magnitude direction, J-removed", "mag_jrem"), ("magnitude direction, raw", "mag_raw")]
    for lab, key in groups:
        names = [f"{key}@{d}{s}" for d in ("r11", "r22") for s in "+-"]
        k = sum(int((np.array(g4["ans"][n]) != base).sum()) for n in names)
        n = len(names) * len(base)
        flips[lab] = S.wilson(k, n) + (k, n)
    for lab, key in (("9-dim class-mean swap, raw", "swap_raw"), ("9-dim class-mean swap, J-removed", "swap_jrem")):
        k = int((np.array(g4["ans"][key]) != base).sum())
        flips[lab] = S.wilson(k, len(base)) + (k, len(base))
    g4b = json.loads((E10 / "results" / "stage2a_g4" / "stage2a_g4b.json").read_text())
    kd = int(round(g4b["conditions"]["patch_full"]["flip_rate"] * g4b["n"]))
    flips["whole vector from a donor item (control)"] = S.wilson(kd, g4b["n"]) + (kd, g4b["n"])

    fig, (ax, bx) = S.plt.subplots(1, 2, figsize=S.FULL, gridspec_kw={"width_ratios": [0.85, 1.25]})
    labs = ["1st\ntens", "1st\nunits", "2nd\ntens", "2nd\nunits"]
    for i, s in enumerate(truth):
        p, lo, hi, k, n = acc[s]
        ax.bar(i, p, 0.62, color=S.C["probe"])
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color=S.C["ink"], capsize=4, lw=1.3)
        ax.text(i, hi + 0.015, f"{p:.2f}", ha="center", va="bottom", fontsize=13)
    ax.axhline(0.1, color=S.C["line"], ls="--", lw=1.5)
    ax.text(3.42, 0.1, " chance", ha="left", va="center", color=S.C["muted"], fontsize=12, clip_on=False)
    ax.set_xticks(range(4), labs, fontsize=13)
    ax.set_xlabel("operand digit")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("probe accuracy, digit at '=' (10 classes)")
    ax.set_title("a probe reads the operands", fontsize=15)

    fl = list(flips)
    y = np.arange(len(fl))[::-1]
    for yi, lab in zip(y, fl):
        p, lo, hi, k, n = flips[lab]
        c = S.C["muted"] if "control" in lab else S.C["probe"]
        bx.errorbar(p * 100, yi, xerr=[[(p - lo) * 100], [(hi - p) * 100]], fmt="o", ms=9, color=c, capsize=4, lw=1.3)
        bx.text(hi * 100 + 0.4, yi, f"{k} of {n:,}", va="center", fontsize=12)
    bx.set_yticks(y, fl, fontsize=12.5)
    bx.set_xlim(-0.3, 6.5)
    bx.set_xlabel("answers changed by the edit at '=' (%)")
    bx.set_title("but no edit at '=' changes the answer", fontsize=15)
    fig.tight_layout(w_pad=2)

    checks = [
        {"quantity": "exact first operand (bare prompts)", "computed": round(ex["a"][0], 3), "record": 0.825, "where": "10_operand_gate RESULTS.md stage 1", "match": abs(ex["a"][0] - 0.825) < 0.0015},
        {"quantity": "exact second operand (bare prompts)", "computed": round(ex["b"][0], 3), "record": 0.948, "where": "10_operand_gate RESULTS.md stage 1", "match": abs(ex["b"][0] - 0.948) < 0.0015},
    ]
    for s, rec in (("a_tens", 0.94), ("a_units", 0.88), ("b_tens", 0.97), ("b_units", 0.98)):
        checks.append({"quantity": f"G1 acc {s}", "computed": round(acc[s][0], 3), "record": rec, "where": "10_operand_gate RESULTS.md stage 1 table", "match": abs(acc[s][0] - rec) < 0.006})
    nflip = sum(v[3] for k, v in flips.items() if "control" not in k)
    nrows = sum(v[4] for k, v in flips.items() if "control" not in k)
    checks.append({"quantity": "flips across steered direction/swap rows (one-digit bank)", "computed": f"{nflip} of {nrows}", "record": "0 (flip rate 0.00 every condition)", "where": "10_operand_gate RESULTS.md stage 2a-G4", "match": nflip == 0})
    checks.append({"quantity": "donor full-patch flips", "computed": f"{kd} of {g4b['n']}", "record": "1 of 117", "where": "10_operand_gate RESULTS.md positive control", "match": kd == 1})
    S.save(fig, "bk_exp10_operand",
           title="Operand digits at '=' are decodable by a probe, yet no edit there changes the model's answer",
           caption=("Left: 5-fold cross-validated linear probes on the layer-42 activation at '=' for 600 bare 'a +/- b =' prompts "
                    "(external prompt set), accuracy per digit slot with Wilson 95% intervals; exact operands 0.83 / 0.95. "
                    "Right: 117 one-digit few-shot items the model answers 100% correctly; centred edits at '=' along "
                    "the digit's class mean or a magnitude direction (+/-, 0.11 and 0.22 of the norm), a 9-dim class-mean swap, "
                    "and a whole-vector donor patch; share of greedy answers that change, Wilson 95%. "
                    "Caveat: even the donor patch flips 1 of 117, so the '=' site at layer 42 cannot show causal use; "
                    "'unused here' is a property of the site, not proof the model ignores operand identity."),
           section="backup:exp10", role="backup",
           sources=["experiments/10_operand_gate/data/arithmetic.jsonl", "experiments/10_operand_gate/results/gate.json",
                    "experiments/10_operand_gate/results/stage2a_g4/g4_raw.pt",
                    "experiments/10_operand_gate/results/stage2a_g4/stage2a_g4b.json"],
           checks=checks,
           notes=("Probe accuracy recomputed from the stored CV predictions in gate.json against the operands in arithmetic.jsonl. "
                  "Flips recomputed from g4_raw.pt greedy answers vs the unsteered 'none' condition (r11 and r22, + and - pooled per "
                  "direction). The donor-patch count comes from the aggregate stage2a_g4b.json (flip_rate x n); its raw file is not "
                  "local. J-removed = top-1536 J-space projected out. Several raw conditions failed the leak test (mag_raw, swaps "
                  "VOID per RESULTS) but flips are zero regardless. Few-shot example: '68 + 85 = 153\\n97 - 23 = 74\\n78 - 34 = 44\\n1 + 1 ='."))
    print(acc, ex)
    for k, v in flips.items():
        print(k, v)
    for c in checks:
        print(c)


if __name__ == "__main__":
    main()
