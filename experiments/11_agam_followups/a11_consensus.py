"""Phase C: judge calls (Gemini 3.8 Flash via OpenRouter, T=0; kimi-k3 on Fireworks as the second judge).

    ../../.venv/bin/python a11_consensus.py shared      # C2: do NLA and OL assert the SAME wrong content? (exp 06 claims)
    ../../.venv/bin/python a11_consensus.py consensus   # C1: k=8 samples (Phase B) -> claims -> support counts -> + vs -

PILOT / EXPLORATORY. Keys are read from the repo .env by experiments/06's llm.Client and never printed.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP.parents[0] / "src"))
sys.path.insert(0, str(EXP / "06_resolving_decomposition"))

import a11_lib as L  # noqa: E402
from common import load_items  # noqa: E402
from llm import GEMINI, KIMI, Client  # noqa: E402

D06 = EXP / "06_resolving_decomposition" / "data"
OUT = HERE / "results"
SEED = 11

SAME_WRONG_PROMPT = """Two interpretability tools each described a language model's internal state for the text below. Both made at least one claim that contradicts the text or the gold answer. Decide whether the two tools assert the SAME wrong content (the same wrong entity, number, word, language or fact), as opposed to different wrong things.

<text>{text}</text>
Gold answer: {target}

Tool A's contradicted claims:
{a}

Tool B's contradicted claims:
{b}

Return JSON: {{"same_wrong_content": true|false, "shared": "<the shared wrong content, or empty>"}}"""
SAME_SCHEMA = {"type": "object", "properties": {"same_wrong_content": {"type": "boolean"}, "shared": {"type": "string"}},
               "required": ["same_wrong_content", "shared"], "additionalProperties": False}

LINE_CLAIM_PROMPT = """You will see {k} independent descriptions, written by a model, of another model's internal state while reading a passage of hard-wrapped prose. Judge only what each description DESCRIBES in words; ignore the formatting of the descriptions themselves.

For each description, answer whether it makes a claim about the current position within a line, the line's length, text wrapping, or an upcoming line break in the passage being described. Then list the distinct such claims (short paraphrases) and, for each, which descriptions (by number) make it.

{descs}

Return JSON: {{"per_description": [true|false, ...] (length {k}), "claims": [{{"claim": "...", "supporters": [int, ...]}}]}}"""
LINE_SCHEMA = {"type": "object", "properties": {
    "per_description": {"type": "array", "items": {"type": "boolean"}},
    "claims": {"type": "array", "items": {"type": "object", "properties": {
        "claim": {"type": "string"}, "supporters": {"type": "array", "items": {"type": "integer"}}},
        "required": ["claim", "supporters"], "additionalProperties": False}}},
    "required": ["per_description", "claims"], "additionalProperties": False}


def parse(txt: str | None) -> dict | None:
    if not txt:
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


# ----------------------------------------------------------------------------- C2
def shared_wrong() -> dict[str, Any]:
    rows = [json.loads(l) for l in (D06 / "claims_gemini.jsonl").read_text().splitlines()]
    items = {(it["family"], it["id"]): it for it in load_items()}
    shared = L.shared_contradictions(rows, readers=("NLA@42", "OL@42"))
    rng = random.Random(SEED)
    # mismatched baseline: NLA claims from item i against OL claims from a different item j (same family)
    by_fam: dict[str, list[dict]] = {}
    for s in shared:
        by_fam.setdefault(s["family"], []).append(s)
    mism = []
    for s in shared:
        pool = [t for t in by_fam[s["family"]] if t["id"] != s["id"]]
        if pool:
            mism.append((s, rng.choice(pool)))
    out: dict[str, Any] = {"n_shared_items": len(shared), "n_mismatched": len(mism)}
    for judge_name, model in (("gemini", GEMINI), ("kimi", KIMI)):
        c = Client(model=model, temperature=0.0, max_tokens=300, reasoning="low" if model == GEMINI else None)

        def ask(a: dict, b: dict) -> dict | None:
            it = items[(a["family"], a["id"])]
            prompt = SAME_WRONG_PROMPT.format(text=it["prompt"][-1200:], target=it["target"],
                                              a="\n".join("- " + x for x in a["NLA@42"]),
                                              b="\n".join("- " + x for x in b["OL@42"]))
            # Fireworks' strict json_schema mode returned mostly unparseable text for kimi-k3; plain JSON-in-text parses.
            return parse(c.chat(prompt, SAME_SCHEMA if model == GEMINI else None))

        with ThreadPoolExecutor(6) as ex:
            real = list(ex.map(lambda s: ask(s, s), shared))
            base = list(ex.map(lambda p: ask(p[0], p[1]), mism))
        rr = [r for r in real if r]
        bb = [r for r in base if r]
        out[judge_name] = {"matched_same_rate": sum(r["same_wrong_content"] for r in rr) / max(1, len(rr)), "n_matched": len(rr),
                           "mismatched_same_rate": sum(r["same_wrong_content"] for r in bb) / max(1, len(bb)), "n_mismatched": len(bb),
                           "examples_shared": [{"id": s["id"], "shared": r["shared"]} for s, r in zip(shared, real) if r and r["same_wrong_content"]][:15],
                           "spend_usd": round(c.spend_usd, 4), "n_failed": c.n_failed}
    (OUT / "c2_shared_wrong.json").write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- C1
def consensus() -> dict[str, Any]:
    import torch
    pb = HERE / "results" / "phaseB"
    sets: dict[str, dict] = {}
    if (pb / "nla_gens.pt").exists():
        sets["NLA@42"] = torch.load(pb / "nla_gens.pt", weights_only=False)["col_k8@42"]
    olg = torch.load(pb / "ol_gens.pt", weights_only=False)
    sets.update({"OL@42": olg["col_k8@42"], "OL@44": olg["col_k8@44"]})
    prev = json.loads((OUT / "c1_consensus.json").read_text()) if (OUT / "c1_consensus.json").exists() else {}
    c = Client(model=GEMINI, temperature=0.0, max_tokens=1200, reasoning="low")
    out: dict[str, Any] = {"note": "k=8 free readouts per item per condition; claims about line position judged by Gemini (T=0); support = number of the 8 samples making the claim"}
    out.update({k: v for k, v in prev.items() if k in ("NLA@42", "OL@42", "OL@44") and k not in sets})

    def judge_item(samples: list[str | None]) -> dict | None:
        ss = [s or "(no explanation)" for s in samples]
        descs = "\n\n".join(f"[{i + 1}] {s[:900]}" for i, s in enumerate(ss))
        for attempt in range(3):
            try:  # the shared client does not catch dropped connections (http.client.RemoteDisconnected)
                return parse(c.chat(LINE_CLAIM_PROMPT.format(k=len(ss), descs=descs), LINE_SCHEMA))
            except Exception as e:  # noqa: BLE001
                print("judge_item retry", attempt, repr(e)[:120], flush=True)
                time.sleep(3 * (attempt + 1))
        return None

    for reader, conds in sets.items():
        out[reader] = {}
        for cond, rows in conds.items():
            with ThreadPoolExecutor(6) as ex:
                res = list(ex.map(judge_item, rows))
            per = [r["per_description"] if r else [] for r in res]
            support = [sum(bool(x) for x in p) for p in per]
            k = max((len(p) for p in per), default=8)
            max_claim_support = [max((len(cl["supporters"]) for cl in r["claims"]), default=0) if r else 0 for r in res]
            out[reader][cond] = {
                "n_items": len(rows), "n_judged": sum(1 for r in res if r),
                "single": sum(support) / max(1, sum(len(p) for p in per)),
                "items_any": sum(s > 0 for s in support) / max(1, len(support)),
                "consensus_ge2": sum(s >= 2 for s in support) / max(1, len(support)),
                "consensus_ge3": sum(s >= 3 for s in support) / max(1, len(support)),
                "same_claim_ge2": sum(m >= 2 for m in max_claim_support) / max(1, len(max_claim_support)),
                "k": k, "support": support,
                "example_claims": [cl["claim"] for r in res if r for cl in r["claims"] if len(cl["supporters"]) >= 2][:10]}
        # + vs - : item-level McNemar on "any consensus(>=2) claim"
        if "jrem4+" in out[reader] and "jrem4-" in out[reader]:
            sp, sm = out[reader]["jrem4+"]["support"], out[reader]["jrem4-"]["support"]
            b = sum(1 for p, m in zip(sp, sm) if p >= 2 and m < 2)
            cc = sum(1 for p, m in zip(sp, sm) if p < 2 and m >= 2)
            out[reader]["plus_vs_minus"] = {"plus_only": b, "minus_only": cc, "both": sum(1 for p, m in zip(sp, sm) if p >= 2 and m >= 2),
                                            "mcnemar_chi2": ((abs(b - cc) - 1) ** 2 / (b + cc)) if (b + cc) else 0.0}
    out["spend_usd"] = round(c.spend_usd, 4)
    out["n_failed"] = c.n_failed
    (OUT / "c1_consensus.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "shared"
    r = shared_wrong() if what == "shared" else consensus()
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, dict) or k in ("gemini", "kimi")}, indent=1, default=str)[:3000])
