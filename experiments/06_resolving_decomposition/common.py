"""Shared item loading and prompt templates for 06_resolving_decomposition.

Items: the 4 regex multi-token WorkspaceBench families (n=100 each), plain render, read at the
final prompt token (so the "prefix" handed to TextNLA is exactly the item prompt).
Reader readouts come from experiments/01_wsbench_replication/readouts (k=1 sample per cell).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FORK = REPO / "third_party" / "workspace-bench"
READOUTS_01 = REPO / "experiments" / "01_wsbench_replication" / "readouts"
JUDGED_01 = REPO / "experiments" / "01_wsbench_replication" / "judged"
FAMILIES = ("typo_mt", "multilingual_mt", "multihop_mt", "basic_readout_mt")
LAYER = 42

# wsbench prompt-only floor template (evals/baselines/prompt_only.json _source.prompt_template)
WSB_PROMPT_ONLY = (
    "Below is a prompt that was given to a language model, cut off at a specific token. The "
    "token in question is the final token of the text below: {token!r}.\n\n<prompt>\n{prefix}\n"
    "</prompt>\n\nWhat would a language model be thinking about on this token? Please write me "
    "a summary for this token."
)

# Re-solving ceiling: a text-only LLM sees ONLY a reader's readout. Family-neutral wording.
RESOLVE_TEMPLATE = (
    "Below is a description, produced by an interpretability tool, of what a language model was "
    "internally representing at the final token of a text it was reading. You do not see the "
    "text itself.\n\n<description>\n{expl}\n</description>\n\nUsing only this description, work "
    "out what the model was thinking about at that token. Write a short summary (at most ~150 "
    "words) that (1) reconstructs what the text was most likely about, (2) names the specific "
    "entity, term, word or answer the model was most likely representing or about to produce, "
    "written out in full and spelled correctly, and (3) states the language of the text."
)


def load_items() -> list[dict]:
    out = []
    for fam in FAMILIES:
        d = json.loads((FORK / "evals" / fam / "items.json").read_text(encoding="utf-8"))
        items = d if isinstance(d, list) else d["items"]
        for it in items:
            out.append({"family": fam, "id": it["name"], "prompt": it["prompt"],
                        "target": it["target"], "target_alts": it.get("target_alts", []),
                        "intermediates": it.get("intermediates", []),
                        "lang": it.get("lang"), "units": it["units"]})
    return out


def load_readouts(arm: str, layer: int = LAYER) -> dict[tuple[str, str], str]:
    """(family, id) -> first sample of the arm's readout at `layer`."""
    out = {}
    for fam in FAMILIES:
        for line in (READOUTS_01 / arm / f"{fam}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["layer"] == layer:
                out[(fam, r["id"])] = r["samples"][0] if r.get("samples") else ""
    return out
