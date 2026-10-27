"""Exp 22's "states" judge (exp 12 run_judge / recall_lib: Gemini 3.8 Flash via OpenRouter, temperature 0, 20 readouts
per call, the same prompt, schema and feature descriptions) run on the mining readouts of every checkpoint of exp 29.

    .venv/bin/python judge29.py --dry-run                 # counts, cost estimate, two prompts; no call
    .venv/bin/python judge29.py --cap 15                  # judge everything not yet judged, stop at $15 total
Options: --ckpts a,b  --features f,g  --samples 8|4  --per-ckpt-cap 15 (8 samples per item unless a checkpoint
would cost more than this, then 4)  --limit N (readouts per cell, smoke)  --results DIR.
Resumable: results/judged/{ckpt}_{row}.json is skipped when complete. Keys from .env only (never printed).
Output format = exp 22's judged files (feature, reader, values, n_rows, n_judged, judged[{item, value, k, stated,
stated_value, exhibited_value}]) plus "ckpt" and "samples", so recall_lib.aggregate reads them unchanged."""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "06_resolving_decomposition"))
sys.path.insert(0, str(ROOT / "experiments" / "12_concept_recall"))
from lib29 import FEATURES, ROWS, OUT_FILE, checkpoint_order, judge_cost_estimate, judge_rows, samples_for_budget  # noqa: E402
from recall_lib import CHUNK, SCHEMA, Readout, build_prompt, chunks, parse_answers  # noqa: E402

ITEMS22 = ROOT / "experiments" / "22_public_bank" / "results" / "items_base.json"
THREADS = 6                                      # exp 12 run_judge


def values_by_id() -> dict[str, str]:
    return {it["id"]: it["value"] for it in json.loads(ITEMS22.read_text()) if it["split"] == "mine"}


def available_ckpts(res: Path) -> list[str]:
    names = {p.name[len("gen29_"):-len(".json")] for p in res.glob("gen29_*.json")}
    ck = set()
    for n in names:
        for row in ROWS:
            if n.endswith("_" + row):
                ck.add(n[: -len(row) - 1])
    return checkpoint_order(ck)


def cell_rows(res: Path, ckpt: str, row: str, value_of: dict, samples: int, limit: int | None) -> list[Readout] | None:
    p = res / OUT_FILE("gen", ckpt, row)
    if not p.exists():
        return None
    gen = json.loads(p.read_text())
    rows = judge_rows(gen, value_of, samples)[:limit]
    return [Readout(row, f"nla:{ckpt}", r["item"], r["value"], r["k"], r["text"]) for r in rows]


def judge_cell(client, out_dir: Path, ckpt: str, row: str, rows: list[Readout], samples: int, log) -> dict:  # noqa: ANN001
    spec = FEATURES[row]
    path = out_dir / f"{ckpt}_{row}.json"
    if path.exists():
        prev = json.loads(path.read_text())
        if prev.get("n_judged") == len(rows) and prev.get("samples") == samples:
            log(f"skip {ckpt}/{row} (complete)")
            return prev

    def run(chunk: list[Readout]) -> list[dict]:
        raw = client.chat(build_prompt(spec["desc"], spec["values"], chunk), json_schema=SCHEMA)
        ans = parse_answers(raw, len(chunk))
        return [{"item": r.item, "value": r.value, "k": r.k, **ans[i]} for i, r in enumerate(chunk) if i in ans]

    with ThreadPoolExecutor(THREADS) as ex:
        judged = [r for part in ex.map(run, chunks(rows)) for r in part]
    out = {"feature": row, "reader": f"nla:{ckpt}", "ckpt": ckpt, "values": spec["values"], "samples": samples,
           "n_rows": len(rows), "n_judged": len(judged), "judged": judged}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=0))
    log(f"{ckpt}/{row}: judged {len(judged)}/{len(rows)}  spend ${client.spend_usd:.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--ckpts", default="")
    ap.add_argument("--features", default=",".join(ROWS))
    ap.add_argument("--samples", type=int, default=0, help="readouts per mining item (0 = 8, or 4 if over --per-ckpt-cap)")
    ap.add_argument("--per-ckpt-cap", type=float, default=15.0)
    ap.add_argument("--cap", type=float, default=15.0, help="total dollars; stop before the next cell once exceeded")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    res = Path(a.results)
    out_dir = res / "judged"
    value_of = values_by_id()
    n_items = len(value_of) // len(ROWS) * len(ROWS)        # 40 mining items x 8 rows = 320 (judge rows per sample)
    samples = a.samples or samples_for_budget(n_items, a.per_ckpt_cap)
    ckpts = a.ckpts.split(",") if a.ckpts else available_ckpts(res)
    rows = a.features.split(",")
    plan = []
    for ck in ckpts:
        for row in rows:
            cell = cell_rows(res, ck, row, value_of, samples, a.limit)
            if cell is None:
                print(f"missing {OUT_FILE('gen', ck, row)}; skipped", flush=True)
                continue
            plan.append((ck, row, cell))
    n_readouts = sum(len(c) for _, _, c in plan)
    n_calls = sum(math.ceil(len(c) / CHUNK) for _, _, c in plan)
    print(f"checkpoints {ckpts}; rows {rows}; samples per item {samples}; cells {len(plan)}; readouts {n_readouts}; "
          f"calls {n_calls}; estimated ${judge_cost_estimate(n_readouts):.2f} (exp 22 rate; "
          f"${judge_cost_estimate(samples * n_items):.2f} per checkpoint)", flush=True)
    if a.dry_run:
        shown = 0
        for ck, row, cell in plan:
            if shown >= 2:
                break
            if shown == 1 and row != "code_lang" and any(r == "code_lang" for _, r, _ in plan):
                continue
            print(f"\n----- prompt {shown + 1}: {ck}/{row}, first chunk of {len(cell)} readouts -----")
            print(build_prompt(FEATURES[row]["desc"], FEATURES[row]["values"], chunks(cell)[0]))
            shown += 1
        print("\n(dry run: no API call made)")
        return
    from llm import Client

    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (res / "judge.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    client = Client(temperature=0.0, max_tokens=4000)
    for ck, row, cell in plan:
        if client.spend_usd > a.cap:
            log(f"cap ${a.cap} reached; stopping")
            break
        judge_cell(client, out_dir, ck, row, cell, samples, log)
    log(f"done: {client.n_calls} calls, {client.n_failed} failed, ${client.spend_usd:.2f}")


if __name__ == "__main__":
    main()
