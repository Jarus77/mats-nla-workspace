"""Score the prompt-only floor verification arms (data/floor_gen.json) with the regex
instrument and compare with the frozen floor. -> data/floor_tables.md"""
import json
import re

from common import FAMILIES, HERE
from score import JOUT, write_and_judge

FROZEN = {"typo_mt": 0.67, "multilingual_mt": 0.42, "multihop_mt": 0.69, "basic_readout_mt": 0.59}
FROZEN_EMPTY = {"typo_mt": 3, "multilingual_mt": 4, "multihop_mt": 1, "basic_readout_mt": 1}


def answer_only(t: str) -> str:
    return t.split("</think>", 1)[1].strip() if "</think>" in t else ""


def main() -> None:
    g = json.loads((HERE / "data" / "floor_gen.json").read_text(encoding="utf-8"))
    arms, stats = {}, {}
    for v, r in g.items():
        cells = {tuple(k.split("|")[1:]): o["text"] for k, o in r["outs"].items()}
        arms[f"floor_{v}"] = cells
        if v == "think_on":
            arms["floor_think_on_answer_only"] = {k: answer_only(t) for k, t in cells.items()}
    for arm, cells in arms.items():
        for fam in FAMILIES:
            fc = [t for (f, _), t in cells.items() if f == fam]
            stats[(arm, fam)] = dict(
                empty=sum(not t.strip() for t in fc),
                think_tag=sum("<think>" in t or "</think>" in t for t in fc),
                closed=sum("</think>" in t for t in fc))
    write_and_judge(arms)
    trunc = {v: {fam: sum(o["n_new"] >= 256 for k, o in r["outs"].items() if k.split("|")[1] == fam)
                 for fam in FAMILIES} for v, r in g.items()}
    md = ["| arm | " + " | ".join(FAMILIES) + " | mean abs diff vs frozen |", "|---|" + "---|" * 5,
          "| frozen floor (empty cells) | " + " | ".join(f"{FROZEN[f]:.2f} ({FROZEN_EMPTY[f]})" for f in FAMILIES) + " | – |"]
    for arm in arms:
        rates = {}
        for fam in FAMILIES:
            res = json.loads((JOUT / arm / fam / "results.json").read_text())
            rates[fam] = res["numbers"]["value"]
        mad = sum(abs(rates[f] - FROZEN[f]) for f in FAMILIES) / 4
        md.append(f"| {arm} (empty / </think> closed) | " + " | ".join(
            f"{rates[f]:.2f} ({stats[(arm, f)]['empty']} / {stats[(arm, f)]['closed']})" for f in FAMILIES)
            + f" | {mad:.3f} |")
    md += ["", "Cells hitting the 256-token cap: " + json.dumps(trunc)]
    (HERE / "data" / "floor_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
