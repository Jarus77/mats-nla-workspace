"""A4 + A6: side-by-side readouts (NLA / Oracle Lens / J-Lens on the same activation) and the
shared-hallucination catalogue, from stored outputs only.

    ../../.venv/bin/python a11_sidebyside.py   -> results/sidebyside.md, results/catalogue.md, results/catalogue.json

Sources
  exp 01  WorkspaceBench readouts: NLA@42 (1 sample), OL at 12 layers (1 sample each), J-Lens top-10 (12 layers)
  exp 08  col pilot: NLA explanations (k=4) and OL bullets (k=4) at zero dose and +/- J-removed q90; J-Lens
          natural top-20; the passage text ending at the read token
  exp 05  planted bridge: the 12 NLA free explanations, natural vs planted (no OL text stored; B2 adds it)
  exp 10  Kenney's 600 arithmetic prompts: his stored NLA readouts and per-digit probe predictions
  exp 06  claims_gemini.jsonl: readouts split into claims with labels (for the catalogue)

PILOT / EXPLORATORY.
"""
from __future__ import annotations

import glob
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP.parents[0] / "src"))
sys.path.insert(0, str(EXP / "06_resolving_decomposition"))

import a11_lib as L  # noqa: E402
from common import load_items  # noqa: E402  (exp 06: WorkspaceBench items with prompt/target)

D01 = EXP / "01_wsbench_replication" / "readouts"
D05 = EXP / "05_planted_intermediates" / "results"
D06 = EXP / "06_resolving_decomposition" / "data"
D08 = EXP / "08_col_pilot" / "results" / "08_col_pilot"
D10 = EXP / "10_operand_gate" / "data"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

SEED = 11
N_SHOW = 6
TRUNC = 420


def clip(t: str | None, n: int = TRUNC) -> str:
    if not t:
        return "(none)"
    t = t.replace("\n", " ⏎ ").strip()
    return t if len(t) <= n else t[:n] + " …"


def q(t: str) -> str:
    return "> " + t.replace("\n", "\n> ")


# ----------------------------------------------------------------------------- exp 01
def wsbench_rows() -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for fam_file in sorted((D01 / "nla").glob("*.jsonl")):
        fam = fam_file.stem
        for line in fam_file.read_text().splitlines():
            r = json.loads(line)
            rows[(fam, r["id"])] = {"pos": r["pos"], "token": r["token"], "NLA@42": r["samples"][0]}
        for reader in ("olens", "jlens"):
            for line in (D01 / reader / f"{fam}.jsonl").read_text().splitlines():
                r = json.loads(line)
                key = (fam, r["id"])
                if key not in rows:
                    continue
                if reader == "olens":
                    rows[key][f"OL@{r['layer']}"] = r["samples"][0]
                else:
                    rows[key][f"JLens@{r['layer']}"] = r["tokens"]
    return rows


def section_wsbench(rng: random.Random) -> str:
    items = {(it["family"], it["id"]): it for it in load_items()}
    rows = wsbench_rows()
    keys = [k for k in rows if k in items]
    pick = rng.sample(keys, min(N_SHOW, len(keys)))
    out = ["## WorkspaceBench (exp 01), unsteered, layer 42 unless marked\n",
           "One sample per reader. J-Lens = top-10 tokens. Oracle shown at 42 and at the trained neighbours 40 / 44.\n"]
    for key in pick:
        it, r = items[key], rows[key]
        out += [f"### {key[0]} / {key[1]}",
                f"**Prompt (tail):** {clip(it['prompt'][-300:], 300)}  ",
                f"**Gold target:** {it['target']}; intermediates: {it.get('intermediates')}  ",
                f"**NLA@42:** {clip(r.get('NLA@42'))}  ",
                f"**OL@42:** {clip(r.get('OL@42'))}  ",
                f"**OL@40:** {clip(r.get('OL@40'), 200)}  ",
                f"**OL@44:** {clip(r.get('OL@44'), 200)}  ",
                f"**J-Lens@42 top-10:** {r.get('JLens@42')}  ", ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- exp 08
def section_col(rng: random.Random) -> str:
    fres: dict = {}
    for f in sorted(glob.glob(str(D08 / "fve_part*.pt"))):
        fres.update(torch.load(f, weights_only=False)["res"])
    by_cond = {r["cond"]: r["explanations"] for r in fres.values()}
    g = torch.load(D08 / "gen_ol.pt", weights_only=False)
    ol_by = {c: t for c, t in zip(g["conds"], g["texts"])}
    nat = torch.load(D08 / "natural.pt", weights_only=False)
    spec = torch.load(D08 / "spec.pt", weights_only=False)
    items = spec["items"]
    zero = ("zero", None, "add", None, 0, None)
    plus = ("real", "jrem4", "centred", "q90", 1, "jrem4")
    minus = ("real", "jrem4", "centred", "q90", -1, "jrem4")
    ol_plus = ("real", "jrem1", "centred", "q90", 1, "jrem1")
    ol_minus = ("real", "jrem1", "centred", "q90", -1, "jrem1")
    pick = rng.sample(range(len(items)), N_SHOW)
    out = ["## col pilot (exp 08): zero dose vs ± J-removed col direction at q90, layer 42\n",
           "NLA: J-removed at 4r* (the P1 cell); OL: J-removed at r* (the only steered OL text stored; B3 adds 4r* at k=8). "
           "Two of the four samples shown per condition. J-Lens: natural top-10 (steered token lists come from B2).\n"]
    for i in pick:
        it = items[i]
        out += [f"### {it['id']}  (col = {it['col']}, chars remaining on the line = {it['chars_remaining']})",
                f"**Text ending at the read token:** {clip(it['text'][-240:], 240)}  ",
                f"**J-Lens@42 natural top-10:** {[t for t, _ in nat['jl_top20'][i][:10]]}  ", ""]
        for label, cond in (("zero", zero), ("+Δ (longer line)", plus), ("−Δ (shorter line)", minus)):
            ex = by_cond.get(cond, [[]] * len(items))[i]
            out.append(f"**NLA {label}:**  ")
            out += [q(clip(e, 300)) for e in ex[:2]] + [""]
        for label, cond in (("zero", zero), ("+Δ", ol_plus), ("−Δ", ol_minus)):
            ex = ol_by.get(cond, [[]] * len(items))[i]
            out.append(f"**OL {label}:**  ")
            out += [q(clip(e, 300)) for e in ex[:2]] + [""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- exp 05
def section_planted() -> str:
    rj = json.loads((D05 / "readers.json").read_text())
    out = ["## Planted bridge (exp 05): NLA free explanations, natural vs planted (layer-38 swap, read at 42)\n",
           "All 12 stored pairs. The text is unchanged between the two; only the activation differs. "
           "Oracle Lens bullets for the same vectors come from B2 (not stored in exp 05); the J-Lens hit rate was 1.00 planted / 0.01 natural.\n"]
    for e in rj["nla_examples"]:
        planted, original = e["names"]
        out += [f"### planted {planted} (original {original})",
                f"**natural:** {clip(e['natural'][0], 500)}  ",
                f"**planted:** {clip(e['planted'][0], 500)}  ", ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- exp 10
def section_operands(rng: random.Random) -> str:
    rows = [json.loads(l) for l in (D10 / "arithmetic.jsonl").read_text().splitlines()]
    pick = rng.sample(rows, N_SHOW)
    out = ["## Arithmetic operands (exp 10, Kenney's bank): NLA readouts vs the prompt and the probe\n",
           "Bare prompt, read at '=' (layer 42). Probe = his per-digit linear probe predictions; J-Lens never places a digit in its top-10 (exp 10 stage 1); "
           "OL bullets and J-Lens lists for these vectors come from B2.\n",
           "| prompt | probe reads a / b | NLA readout 1 | NLA readout 2 |", "|---|---|---|---|"]
    for r in pick:
        pa = f"{r['probe_a_tens']}{r['probe_a_units']}"
        pb = f"{r['probe_b_tens']}{r['probe_b_units']}"
        n1, n2 = (r["nla_readouts"] + ["", ""])[:2]
        out.append(f"| `{r['text']}` | {pa} / {pb} | {clip(n1, 260)} | {clip(n2, 260)} |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- A6 catalogue
NUM_RE = re.compile(r"\d")


def catalogue() -> tuple[str, dict[str, Any]]:
    rows = [json.loads(l) for l in (D06 / "claims_gemini.jsonl").read_text().splitlines()]
    items = {(it["family"], it["id"]): it for it in load_items()}
    shared = L.shared_contradictions(rows, readers=("NLA@42", "OL@42"))
    n_items = len({(r["family"], r["id"]) for r in rows})
    per_reader = {}
    for rd in ("NLA@42", "OL@42", "TextNLA"):
        rr = [r for r in rows if r["reader"] == rd]
        per_reader[rd] = {"n_readouts": len(rr),
                          "readouts_with_contradiction": sum(any(c["label"] == "contradicted_by_text" for c in r["claims"]) for r in rr)}
    # planted rewrites (exp 05)
    rj = json.loads((D05 / "readers.json").read_text())
    rewrites = [{"planted": e["names"][0], "original": e["names"][1], "text": e["planted"][0]} for e in rj["nla_examples"]]
    # Kenney operand misreads: NLA readouts whose quoted expression differs from the prompt
    ar = [json.loads(l) for l in (D10 / "arithmetic.jsonl").read_text().splitlines()]
    misread = []
    for r in ar:
        for t in r["nla_readouts"]:
            m = re.search(r"(\d+)\s*([+\-−])\s*(\d+)\s*=", t)
            if m and (int(m.group(1)), int(m.group(3))) != (r["a"], r["b"]):
                misread.append({"prompt": r["text"], "quoted": m.group(0), "readout": t[:200]})
                break
    cat = {"n_items_judged": n_items, "per_reader": per_reader, "shared_contradictions": shared,
           "n_shared": len(shared), "planted_rewrites": rewrites,
           "operand_misreads": {"n_prompts": len(ar), "n_with_misquoted_expression": len(misread), "examples": misread[:12]}}
    md = ["# Hallucination catalogue (A6) — PILOT / EXPLORATORY\n",
          f"Claim-level judge (exp 06, Gemini 3.8 Flash): {n_items} WorkspaceBench items, 100 readouts per reader. "
          f"Readouts with at least one claim contradicted by the text: " + ", ".join(f"{k} {v['readouts_with_contradiction']}/{v['n_readouts']}" for k, v in per_reader.items()) + ".\n",
          f"## 1. Same item, both readers contradicted by the text ({len(shared)} items)\n",
          "Whether the two readers' wrong claims name the SAME wrong content is judged in C2; here they are side by side.\n"]
    for s in shared:
        it = items.get((s["family"], s["id"]), {})
        md += [f"### {s['family']} / {s['id']}  (gold: {it.get('target')})",
               "**NLA:** " + " · ".join(clip(c, 200) for c in s["NLA@42"]) + "  ",
               "**OL:** " + " · ".join(clip(c, 200) for c in s["OL@42"]) + "  ", ""]
    md += ["## 2. Re-imagined prompt under a planted fact (exp 05, 12/12)\n",
           "The NLA names the planted country and rewrites the clue to match; the text still says the original clue.\n"]
    md += [f"- **{r['planted']}** (was {r['original']}): {clip(r['text'], 320)}" for r in rewrites] + [""]
    md += [f"## 3. Misquoted arithmetic (exp 10 bank, Kenney's NLA readouts): {len(misread)} of {len(ar)} prompts quote a different expression than the prompt\n"]
    md += [f"- prompt `{m['prompt']}` → readout quotes `{m['quoted'].strip()}`: {clip(m['readout'], 160)}" for m in misread[:12]] + [""]
    md += ["## Categories seen across the three sources\n",
           "- **re-imagined prompt**: the readout restates a prompt that differs from the real one (planted bridges; misquoted operands; exp 06 'the text is …' claims).",
           "- **confabulated specifics**: names, numbers or sources the text does not contain (exp 06 unsupported/contradicted claims).",
           "- **format-driven claims**: statements about quiz/trivia/Q&A format that the text does not have (exp 05 planted explanations).",
           "- **shared wrong answer**: both prose readers assert the same non-gold entity (listed in section 1; quantified in C2)."]
    return "\n".join(md), cat


def main() -> None:
    rng = random.Random(SEED)
    md = ["# Side-by-side readouts (A4) — PILOT / EXPLORATORY\n",
          "Same activation, three readers. Samples are the first ones stored, not selected. Items are a seeded random draw.\n",
          section_wsbench(rng), section_col(rng), section_planted(), section_operands(rng)]
    (OUT / "sidebyside.md").write_text("\n".join(md))
    cmd, cat = catalogue()
    (OUT / "catalogue.md").write_text(cmd)
    (OUT / "catalogue.json").write_text(json.dumps(cat, indent=1))
    print(f"sidebyside.md: {sum(len(x) for x in md)} chars; catalogue: {cat['n_shared']} shared-contradiction items, "
          f"{cat['operand_misreads']['n_with_misquoted_expression']} misquoted operand prompts")


if __name__ == "__main__":
    main()
