"""Exp 28, laptop step 1: collect the rows and freeze eight paraphrase pairs per row (Gemini 3.8 Flash via OpenRouter,
exp 06's Client; one call per row, about 47 calls, well under $1).

  1. lib28.collect_rows() reads each source experiment's spec (the FIXED NLA statement pair) and scorecard (the NLA
     stated rate) and writes results/rows28.json (one record per row: experiment, feature, values, fixed pair, stated
     rate, n readouts, concept type, the volume paths of its items and activations, and the source's own instrument
     values where recorded) plus results/rows28_missing.json (rows with a stated rate that could not be collected).
  2. For each row the model is asked for N_PARA paraphrase pairs that state the same property in eight registers
     (lib28.REGISTERS), the two values aligned, each pair identical except for the value words. Replies are validated
     by lib28.parse_pairs (exactly eight, non-empty, a != b, distinct, not the fixed pair) and retried up to three
     times; every accepted reply is appended to results/paraphrases28.partial.jsonl, so a rerun resumes.
  3. When every row has its pairs the frozen file results/paraphrases28.json is written and its sha256 recorded in
     results/paraphrases28.sha256. The file is never rewritten once it exists (delete it to regenerate).

    .venv/bin/python paraphrase28.py --dry-run     # writes rows28.json, prints every prompt, makes no API call
    .venv/bin/python paraphrase28.py [--cap 2.0] [--rows 22_py_string,13_tb_paren]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "06_resolving_decomposition"))
import lib28 as L  # noqa: E402

RES = HERE / "results"
ROWS = RES / "rows28.json"
MISSING = RES / "rows28_missing.json"
PARTIAL = RES / "paraphrases28.partial.jsonl"
FROZEN = RES / "paraphrases28.json"
SHA = RES / "paraphrases28.sha256"
ATTEMPTS = 3
TEMPERATURE = 0.7
MAX_TOKENS = 1600


def write_rows() -> list[dict]:
    rows, missing = L.collect_rows()
    RES.mkdir(exist_ok=True)
    ROWS.write_text(json.dumps(rows, indent=1))
    MISSING.write_text(json.dumps(missing, indent=1))
    print(f"rows28.json: {len(rows)} rows ({sum(r['type'] == 'self_inducing' for r in rows)} self-inducing); "
          f"{len(missing)} rows with a stated rate could not be collected", flush=True)
    for m in missing:
        print("  missing:", m, flush=True)
    return rows


def load_partial() -> dict[str, dict]:
    if not PARTIAL.exists():
        return {}
    out = {}
    for line in PARTIAL.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["key"]] = d
    return out


def ask(client, row: dict) -> dict | None:  # noqa: ANN001
    prompt = L.paraphrase_prompt(row)
    for attempt in range(1, ATTEMPTS + 1):
        raw = client.chat(prompt, json_schema=L.PAIR_SCHEMA)
        pairs = L.parse_pairs(raw, row["fixed"])
        if pairs is not None:
            regs = [p.get("register", "") for p in json.loads(raw)["pairs"]]
            return {"key": row["key"], "experiment": row["experiment"], "feature": row["feature"], "fixed": row["fixed"],
                    "pairs": pairs, "registers": regs, "attempts": attempt}
        print(f"  {row['key']}: attempt {attempt} rejected ({'no reply' if raw is None else 'validation failed'})", flush=True)
    return None


def freeze(rows: list[dict], done: dict[str, dict], spend: float, model: str) -> None:
    body = {"model": model, "temperature": TEMPERATURE, "n_para": L.N_PARA, "registers": list(L.REGISTERS),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "spend_usd_this_run": round(spend, 4),
            "rows": {r["key"]: {k: done[r["key"]][k] for k in ("experiment", "feature", "fixed", "pairs", "registers", "attempts")} for r in rows}}
    FROZEN.write_text(json.dumps(body, indent=1, ensure_ascii=False))
    digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    SHA.write_text(digest + "  paraphrases28.json\n")
    print(f"frozen {FROZEN.name}: {len(body['rows'])} rows, sha256 {digest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="write rows28.json and print the prompts; no API call")
    ap.add_argument("--cap", type=float, default=2.0, help="stop issuing new calls once the spend passes this (USD)")
    ap.add_argument("--rows", default="", help="comma-separated row keys (default: all)")
    args = ap.parse_args()
    rows = write_rows()
    if args.rows:
        want = set(args.rows.split(","))
        rows = [r for r in rows if r["key"] in want]
    if args.dry_run:
        for r in rows:
            print(f"\n===== {r['key']} (stated {r['stated']:.3f}, {r['type']}) =====\n{L.paraphrase_prompt(r)}")
        print(f"\n[dry run] {len(rows)} prompts, 0 calls", flush=True)
        return
    if FROZEN.exists():
        print(f"{FROZEN} exists; not regenerating (delete it to redo). sha256: {SHA.read_text().strip() if SHA.exists() else 'n/a'}")
        return
    from llm import GEMINI, Client

    client = Client(model=GEMINI, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, reasoning="minimal")
    done = load_partial()
    with PARTIAL.open("a") as fh:
        for r in rows:
            if r["key"] in done:
                continue
            if client.spend_usd > args.cap:
                print(f"spend cap {args.cap} reached; stopping", flush=True)
                break
            got = ask(client, r)
            if got is None:
                print(f"  {r['key']}: FAILED after {ATTEMPTS} attempts", flush=True)
                continue
            fh.write(json.dumps(got, ensure_ascii=False) + "\n")
            fh.flush()
            done[r["key"]] = got
            print(f"  {r['key']}: ok (attempt {got['attempts']}); e.g. {got['pairs'][0][0]!r} / {got['pairs'][0][1]!r}", flush=True)
    print(f"calls {client.n_calls}, failed {client.n_failed}, spend ${client.spend_usd:.4f}", flush=True)
    left = [r["key"] for r in rows if r["key"] not in done]
    if left:
        print(f"{len(left)} rows still without pairs: {left}; rerun to resume", flush=True)
        return
    freeze(rows, done, client.spend_usd, GEMINI)


if __name__ == "__main__":
    main()
