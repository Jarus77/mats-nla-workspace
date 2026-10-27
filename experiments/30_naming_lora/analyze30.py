"""results/eval30_{base,S,C}.json + results/train30_{S,C}.json + results/p5.json -> results/table30.md (one row per
prediction P1-P5: the measured number, the line, pass/fail; then the per-row numbers per arm) and results/fig30.png
(training loss of both arms)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
sys.path.insert(0, str(HERE))
from lib30 import ALL_ROWS, within  # noqa: E402

ARMS = ("base", "C", "S")
P2_HIGH, P2_LOW = ("py_string", "json_depth", "py_arg"), "md_list"
SERIES = {"S": "#2a78d6", "C": "#eb6834"}          # categorical slots 1 and 2, fixed order


def _load(name: str) -> dict | None:
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(x) -> str:  # noqa: ANN001
    return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _verdict(ok: bool | None) -> str:
    return "n/a" if ok is None else ("PASS" if ok else "FAIL")


def prediction_rows(ev: dict, p5: dict | None) -> list[tuple[str, str, str, str]]:
    """(prediction, measured, line, verdict) for P1-P5 from the eval dicts by arm (missing arms give n/a)."""
    S, C = ev.get("S"), ev.get("C")
    rows = []
    fs, fc = (S["gate"]["fve"] if S else None), (C["gate"]["fve"] if C else None)
    fb = ev["base"]["gate"]["fve"] if ev.get("base") else None
    rows.append(("P1 gate FVE, statement arm vs control", f"S {_fmt(fs)}, C {_fmt(fc)}, base {_fmt(fb)}",
                 "|S - C| <= 0.02", _verdict(within(fs, fc, 0.02) if S and C else None)))
    if S:
        acc = {r: S["rows"][r]["accuracy"] for r in P2_HIGH + (P2_LOW,)}
        ok = all(acc[r] >= 0.85 for r in P2_HIGH) and acc[P2_LOW] <= 0.65
        rows.append(("P2 statement accuracy on held-out clusters (arm S)", ", ".join(f"{r} {acc[r]:.2f}" for r in acc),
                     "py_string, json_depth, py_arg >= 0.85; md_list <= 0.65", _verdict(ok)))
        js, lm = S["rows"]["js_string"]["accuracy"], S["rows"]["latex_math"]["naming_rate"]
        rows.append(("P3 transfer rows (arm S)", f"js_string accuracy {js:.2f}, latex_math naming {lm:.2f}",
                     "js_string >= 0.70; latex_math naming < 0.30", _verdict(js >= 0.70 and lm < 0.30)))
        rp = S["random_points"]["strict_regex"]
        t, f = rp["named_when_true"], rp["named_when_false"]
        rows.append(("P4 exp 23 random points, in-string naming (arm S, lib23 strict regex)", f"when true {_fmt(t)}, when false {_fmt(f)}",
                     ">= 0.50 when the state holds; <= 0.15 when it does not", _verdict(t is not None and f is not None and t >= 0.50 and f <= 0.15)))
    else:
        rows += [(p, "n/a", line, "n/a") for p, line in (("P2 statement accuracy (arm S)", "see RESULTS.md"), ("P3 transfer rows (arm S)", "see RESULTS.md"),
                                                         ("P4 exp 23 random points (arm S)", "see RESULTS.md"))]
    if p5 and not p5.get("skipped"):
        d = p5["pooled"]["mean_diff"]
        rows.append(("P5 within-item FVE, naming minus non-naming (existing k = 8 readouts)", f"{_fmt(d)} over {p5['pooled']['n_items']} items",
                     "|mean| <= 0.02", _verdict(d is not None and abs(d) <= 0.02)))
    else:
        rows.append(("P5 within-item FVE, naming minus non-naming", "skipped: exp 22's k = 8 readouts carry no per-sample FVE", "|mean| <= 0.02", "n/a"))
    return rows


def per_row_table(ev: dict) -> list[str]:
    arms = [a for a in ARMS if a in ev]
    head = "| row | trained | n | " + " | ".join(f"{a} acc | {a} naming | {a} FVE" for a in arms) + " |"
    out = [head, "|---|---|---:|" + "---:|" * (3 * len(arms))]
    for row in ALL_ROWS:
        any_arm = ev[arms[0]]["rows"][row]
        cells = [f"{ev[a]['rows'][row]['accuracy']:.2f} | {ev[a]['rows'][row]['naming_rate']:.2f} | {ev[a]['rows'][row]['fve']:.3f}" for a in arms]
        out.append(f"| {row} | {'yes' if any_arm['trained'] else 'no (transfer)'} | {any_arm['n']} | " + " | ".join(cells) + " |")
    rp = " / ".join(f"{a}: true {_fmt(ev[a]['random_points']['strict_regex']['named_when_true'])}, false {_fmt(ev[a]['random_points']['strict_regex']['named_when_false'])}"
                    + f" (statement check: inside-when-true {_fmt(ev[a]['random_points']['statement_check']['accuracy'])})" for a in arms)
    return out + ["", f"Exp 23 random points, in-string naming by lib23's strict regex: {rp}"]


def figure(train: dict, path: Path) -> bool:
    if not train:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    for arm in ("S", "C"):
        if arm not in train:
            continue
        xs, ys = zip(*train[arm]["losses"])
        label = {"S": "S: explanation + statement", "C": "C: explanation only"}[arm]
        ax.plot(xs, ys, color=SERIES[arm], lw=2, label=label)
        ax.annotate(f"{arm} {ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center", color="#333333")
    ax.set_xlabel("step"); ax.set_ylabel("mean token NLL of the target")
    ax.set_title("Exp 30 training loss per arm", fontsize=10, loc="left")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return True


def main() -> None:
    ev = {a: d for a in ARMS if (d := _load(f"eval30_{a}.json"))}
    train = {a: d for a in ("S", "C") if (d := _load(f"train30_{a}.json"))}
    if not ev:
        raise FileNotFoundError(f"no eval30_*.json under {RES}: fetch first")
    md = ["| prediction | measured | line | verdict |", "|---|---|---|---|"]
    md += [f"| {p} | {m} | {line} | {v} |" for p, m, line, v in prediction_rows(ev, _load("p5.json"))]
    md += ["", f"Per row (held-out clusters; arms present: {', '.join(ev)}):", ""] + per_row_table(ev)
    if train:
        md += ["", "Training: " + "; ".join(f"arm {a} {d['step']} steps, final loss {d['losses'][-1][1]:.3f}, {d['wall_s']:.0f} s" for a, d in train.items())]
    (RES / "table30.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if figure(train, RES / "fig30.png"):
        print("fig30.png written")


if __name__ == "__main__":
    main()
