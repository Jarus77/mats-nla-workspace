"""Data for the side-by-side readout viewer Agam asked for (PILOT / EXPLORATORY).

    ../../.venv/bin/python a11_viewer_data.py   -> results/viewer_data.json

Sets: WorkspaceBench (exp 01: Oracle and J-lens at 12 layers, NLA at 42, last token), col and tense pilot items
(exp 11 grid: J-lens at 10 layers x 4 positions, Oracle at its 6 layers x 4 positions, NLA at 42 x 4 positions;
col also has the +/- J-removed edit at 42/44), planted bridge (exp 05 vectors, Oracle at 42/44, J-lens, NLA where
stored) and arithmetic (exp 10 '=' vectors). Samples are the first ones generated; items are seeded draws.
results/pos_tokens.json holds the last four tokens of each col / tense item (decoded with the Qwen tokenizer).
"""
from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
sys.path.insert(0, str(EXP / "05_planted_intermediates"))
R, PB = HERE / "results", HERE / "results" / "phaseB"
SEED = 11
FAMILIES = ("typo_mt", "multilingual_mt", "multihop_mt", "basic_readout_mt")
N_WSB_PER_FAMILY, N_PLANTED, N_ARITH = 12, 24, 24


def _b2u() -> dict[int, str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


DEC = {v: k for k, v in _b2u().items()}


def fix_tok(t: str) -> str:
    """Byte-level BPE token string -> readable text (exp 01 stored raw token strings)."""
    if not any(ord(c) > 127 for c in t):
        return t
    b = bytearray()
    for c in t:
        b.extend([DEC[c]] if c in DEC else c.encode("utf-8"))
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        s = b.decode("utf-8", errors="replace")
        return s if any(ord(c) > 127 and c != "�" for c in s) else t


def toks(lst) -> list[str]:  # noqa: ANN001
    return [t if isinstance(t, str) else t[0] for t in lst]


def wsb() -> list[dict]:
    rd = REPO / "experiments" / "01_wsbench_replication" / "readouts"
    rows = {r: [json.loads(l) for f in sorted((rd / r).glob("*.jsonl")) for l in open(f)] for r in ("olens", "jlens", "nla")}
    fork = REPO / "third_party" / "workspace-bench" / "evals"
    rng = random.Random(SEED)
    out = []
    for fam in FAMILIES:
        d = json.loads((fork / fam / "items.json").read_text(encoding="utf-8"))
        its = d if isinstance(d, list) else d["items"]
        for it in sorted(rng.sample(its, N_WSB_PER_FAMILY), key=lambda x: x["name"]):
            iid = it["name"]
            ol = {str(x["layer"]): x["samples"] for x in rows["olens"] if x["id"] == iid}
            jl = {str(x["layer"]): [fix_tok(t) for t in x["tokens"]] for x in rows["jlens"] if x["id"] == iid}
            nla = [x for x in rows["nla"] if x["id"] == iid]
            tokx = next((x["token"] for x in rows["olens"] if x["id"] == iid), None)
            out.append({"id": iid, "family": fam, "prompt": it["prompt"], "target": it["target"],
                        "intermediates": it.get("intermediates", []), "read_token": tokx,
                        "ol": ol, "jl": jl, "nla": {"42": nla[0]["samples"]} if nla else {}})
    return out


def col_tense(items: dict, acts: dict, pos_tokens: dict) -> dict:
    gp = PB / "grid.pt"
    grid = torch.load(gp, map_location="cpu", weights_only=False) if gp.exists() else None
    top = torch.load(PB / "jl_top10.pt", weights_only=False)
    olg = torch.load(PB / "ol_gens.pt", weights_only=False)
    nlg = torch.load(PB / "nla_gens.pt", weights_only=False)["col_k8@42"]
    out = {}
    for feat in ("col", "tense"):
        idx = grid["sel"][feat] if grid else []
        rows = []
        for j, i in enumerate(idx):
            it = items[feat][i]
            rec = {"id": it["id"], "text": it["text"], "pos_tokens": pos_tokens[it["id"]],
                   "model_top5": acts["mass"][feat]["top5"][i], "p_newline": round(acts["mass"][feat]["p_newline"][i], 4),
                   "jl": {f"{L}|{p}": toks(grid["jl"][(feat, L, p)][j]) for (f, L, p) in grid["jl"] if f == feat},
                   "ol": {f"{L}|{p}": grid["ol"][(feat, L, p)][j] for (f, L, p) in grid["ol"] if f == feat},
                   "nla": {f"{L}|{p}": grid["nla"][(feat, L, p)][j] for (f, L, p) in grid["nla"] if f == feat}}
            if feat == "col":
                rec.update({"col": int(it["col"]), "chars_remaining": int(it["chars_remaining"]), "line_len": int(it["line_len"]),
                            "next_is_nl": it["next_is_nl"] == "True",
                            "steer": {c: {"jl": toks(top[f"col_{c}"][i]), "ol42": olg["col_k8@42"][c][i], "ol44": olg["col_k8@44"][c][i],
                                          "nla42": nlg[c][i]} for c in ("zero", "jrem4+", "jrem4-")}})
            else:
                rec["label"] = it["label"]
            rows.append(rec)
        out[feat] = rows
    out["grid_ready"] = grid is not None
    return out


def _as_list(v):  # noqa: ANN001, ANN202
    return ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else (v if isinstance(v, list) else [v])


def planted() -> list[dict]:
    from plant_items import build_items
    b05 = torch.load(EXP / "05_planted_intermediates" / "results" / "readers_base.pt", map_location="cpu", weights_only=False)
    rj = json.loads((EXP / "05_planted_intermediates" / "results" / "readers.json").read_text())
    # exp 05 stored NLA explanations for its first 12 items, in order (plant_modal.py: sel = range(12))
    ex = rj["nla_examples"]
    prompts = {it["id"]: it["prompt"] for it in build_items()}
    top = torch.load(PB / "jl_top10.pt", weights_only=False)
    olg = torch.load(PB / "ol_gens.pt", weights_only=False)
    names, keep = b05["names"], b05["keep"]
    n_ex = len(ex)
    chosen = list(range(n_ex)) + sorted(random.Random(SEED).sample(range(n_ex, len(names)), N_PLANTED - n_ex))
    out = []
    for i in chosen:
        Y, X = names[i]
        e = ex[i] if i < n_ex else None
        out.append({"id": keep[i]["id"], "prompt": prompts.get(keep[i]["id"], ""), "planted": Y, "original": X,
                    "jl": {"natural": toks(top["planted_natural"][i]), "planted": toks(top["planted_planted"][i])},
                    "ol42": {c: olg["planted@42"][c][i] for c in ("natural", "planted")},
                    "ol44": {c: olg["planted@44"][c][i] for c in ("natural", "planted")},
                    "nla": {"natural": _as_list(e["natural"]), "planted": _as_list(e["planted"])} if e else None})
    return out


def arith() -> list[dict]:
    ar = [json.loads(l) for l in (EXP / "10_operand_gate" / "data" / "arithmetic.jsonl").read_text().splitlines()]
    top = torch.load(PB / "jl_top10.pt", weights_only=False)
    olg = torch.load(PB / "ol_gens.pt", weights_only=False)
    ids = olg.get("operand_ids") or [r["id"] for r in ar[: len(olg["operand@42"])]]
    by_id = {r["id"]: r for r in ar}
    pick = sorted(random.Random(SEED).sample(range(len(olg["operand@42"])), N_ARITH))
    out = []
    for i in pick:
        r = by_id[ids[i]]
        nla = r["nla_readouts"]
        nla = ast.literal_eval(nla) if isinstance(nla, str) else nla
        out.append({"id": r["id"], "text": r["text"], "answer": r["answer"], "model_next": r["model_next"],
                    "probe": f"{r['probe_a_tens']}{r['probe_a_units']} / {r['probe_b_tens']}{r['probe_b_units']}",
                    "ol42": olg["operand@42"][i], "jl": toks(top["operand"][i]), "nla": nla})
    return out


def main() -> None:
    items = json.loads((HERE / "data" / "items.json").read_text())
    acts = torch.load(PB / "acts.pt", map_location="cpu", weights_only=False)
    pos_tokens = json.loads((R / "pos_tokens.json").read_text())
    ct = col_tense(items, acts, pos_tokens)
    data = {"meta": {"model": "Qwen3.6-27B", "positions": [-4, -3, -2, -1], "grid_layers": [24, 28, 32, 36, 40, 42, 44, 48, 52, 56],
                     "ol_layers": [36, 40, 42, 44, 48, 52], "wsb_layers": [20, 24, 28, 32, 36, 40, 42, 44, 48, 52, 56, 60],
                     "grid_ready": ct.pop("grid_ready")},
            "wsb": wsb(), "col": ct["col"], "tense": ct["tense"], "planted": planted(), "arith": arith()}
    s = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (R / "viewer_data.json").write_text(s)
    print({k: (len(v) if isinstance(v, list) else v) for k, v in data.items() if k != "meta"}, "grid_ready", data["meta"]["grid_ready"], f"{len(s.encode()) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
