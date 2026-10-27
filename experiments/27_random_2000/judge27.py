"""Exp 27 judge pass (laptop, no GPU): a blind yes/no judge on the in-string readouts.

For every read point where `in_string` holds (about 500 of the 2,000) and an equal number of randomly chosen points
where it does not (seed 27, the same negatives for both readers), the judge (Gemini 3.8 Flash via OpenRouter, T=0,
exp 06's Client) sees ONLY the readout text, never the document, and answers whether the readout states that the
current position is inside a string literal, with a one-line quote of its evidence. The same pass covers exp 23's 134
in-string readouts plus 134 matched negatives, so the open item from exp 23 (does the strict regex miss namings?)
closes here.

Every judgement is appended to results/judged27.jsonl (resumable; keyed by source/reader/id, since exps 23 and 27
share the rnd_<kind>_<n> id format). results/judge27.json holds per-source, per-reader judged-yes rates when
in_string holds and when it does not, counts, Wilson intervals, and the strict-regex rate on the same items.

Cost: about 2,300 short calls (2 readers x (about 500 + 500) for exp 27, 2 x (134 + 134) for exp 23), one readout
each, about 400 input and 30 output tokens per call: roughly $2 to $3 at Gemini 3.8 Flash prices (exp 22's judge
spent $1.17 on 256 twenty-readout calls). The --cap flag (default $10) stops new calls once the spend passes it.

    .venv/bin/python judge27.py --dry-run          # counts and the first two prompts, no API call
    .venv/bin/python judge27.py [--sources e27,e23] [--workers 6] [--cap 10] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "23_random_points"))
sys.path.insert(0, str(REPO / "experiments" / "06_resolving_decomposition"))
from lib23 import names_state  # noqa: E402
from lib27 import (READERS, TRUNC, append_cache, cache_key, judge_prompt, judge_rates, load_cache,  # noqa: E402
                   parse_verdict, select_judge_items)

STATE = "in_string"
SEED = 27
SOURCES = {"e27": (HERE / "results", "27"), "e23": (REPO / "experiments" / "23_random_points" / "results", "23")}
CACHE = HERE / "results" / "judged27.jsonl"
OUT = HERE / "results" / "judge27.json"


def load_source(name: str) -> tuple[list[dict], dict[str, dict[str, str]]] | None:
    res, suf = SOURCES[name]
    paths = [res / "items_base.json", res / f"ol{suf}.json", res / f"nla{suf}.json"]
    if not all(p.exists() for p in paths):
        return None
    items = json.load(open(paths[0]))
    texts = {"ol": json.load(open(paths[1]))["ol"], "nla": json.load(open(paths[2]))["nla"]}
    assert all(it["id"] in texts[r] for it in items for r in READERS), f"{name}: readouts missing for some items"
    return items, texts


def build_rows(sources: list[str], dry_run: bool) -> list[dict]:
    rows = []
    for name in sources:
        loaded = load_source(name)
        if loaded is None:
            msg = f"{name}: readout files not found under {SOURCES[name][0]}"
            if dry_run:
                print("WARNING", msg, "(skipped in dry run)", flush=True)
                continue
            raise SystemExit(msg)
        items, texts = loaded
        texts = {r: texts[r] for r in READERS}
        for reader, iid, text, label in select_judge_items(items, texts, STATE, SEED):
            rows.append({"key": cache_key(name, reader, iid), "source": name, "reader": reader, "id": iid, "label": label,
                         "regex": names_state(text, STATE), "text": (text or "")[:TRUNC]})
    return rows


def summarise(rows: list[dict], cache: dict[str, dict], meta: dict) -> dict:
    judged = [{**r, "verdict": cache[r["key"]]["verdict"]} for r in rows if r["key"] in cache]
    rates = {name: judge_rates([r for r in judged if r["source"] == name]) for name in dict.fromkeys(r["source"] for r in rows)}
    if len(rates) > 1:
        rates["pooled"] = judge_rates(judged)
    by = {}
    for r in rows:
        by.setdefault(f"{r['source']}/{r['reader']}/{'pos' if r['label'] else 'neg'}", 0)
        by[f"{r['source']}/{r['reader']}/{'pos' if r['label'] else 'neg'}"] += 1
    return {**meta, "state": STATE, "seed": SEED, "n_rows": len(rows), "n_judged": len(judged),
            "n_unparsed": sum(r["verdict"] is None for r in judged), "counts": by, "rates": rates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the number of items and the first two prompts; no API call")
    ap.add_argument("--sources", default="e27,e23")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cap", type=float, default=10.0, help="USD; no new calls once the spend passes it")
    ap.add_argument("--limit", type=int, default=None, help="judge at most N not-yet-cached rows")
    a = ap.parse_args()
    rows = build_rows(a.sources.split(","), a.dry_run)
    cache = load_cache(CACHE)
    todo = [r for r in rows if r["key"] not in cache][:a.limit]
    counts = {f"{s}/{r}": sum(1 for x in rows if x["source"] == s and x["reader"] == r) for s in dict.fromkeys(x["source"] for x in rows) for r in READERS}
    print(f"rows {len(rows)} ({counts}); cached {len(rows) - len(todo) if a.limit is None else len(cache)}; to judge {len(todo)}", flush=True)
    if a.dry_run:
        for r in rows[:2]:
            print(f"\n--- prompt for {r['key']} (label in_string={r['label']}, regex={r['regex']}) ---\n{judge_prompt(r['text'])}")
        print("\ndry run: no API call made", flush=True)
        return
    from llm import Client

    client = Client(temperature=0.0, max_tokens=120, reasoning="minimal")
    lock = threading.Lock()

    def judge_one(r: dict) -> None:
        if client.spend_usd > a.cap:
            return
        raw = client.chat(judge_prompt(r["text"]))
        if raw is None:
            return
        row = {**r, "verdict": parse_verdict(raw), "reply": raw.strip()[:400], "model": client.model}
        with lock:
            append_cache(CACHE, row)
            cache[r["key"]] = row

    CACHE.parent.mkdir(exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(judge_one, todo))
    meta = {"model": client.model, "temperature": client.temperature, "calls_this_run": client.n_calls,
            "failed_this_run": client.n_failed, "spend_usd_this_run": round(client.spend_usd, 4)}
    out = summarise(rows, cache, meta)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rates"}), flush=True)
    for src, by_reader in out["rates"].items():
        for reader, r in by_reader.items():
            print(f"{src}/{reader}: yes when in string {r['rate_pos']} ({r['yes_pos']}/{r['n_pos']}), when not {r['rate_neg']} "
                  f"({r['yes_neg']}/{r['n_neg']}); regex {r['regex_rate_pos']} / {r['regex_rate_neg']}; unparsed {r['n_unparsed']}", flush=True)
    print(f"done: {client.n_calls} calls, {client.n_failed} failed, ${client.spend_usd:.2f}", flush=True)


if __name__ == "__main__":
    main()
