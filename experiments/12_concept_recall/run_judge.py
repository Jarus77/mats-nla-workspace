"""Judge spontaneous recall of each concept type in the exp 07 free readouts (Gemini 3.8 Flash).

Usage: python run_judge.py [--readers nla,ol42] [--features f1,f2] [--limit N] [--cap 10]
Resumable: results/judged/<feature>_<reader>.json is skipped when complete. Keys from .env only.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "06_resolving_decomposition"))
sys.path.insert(0, str(ROOT / "experiments" / "07_template_mining"))
from feature_spec import FEATURES  # noqa: E402
from llm import Client  # noqa: E402
from recall_lib import SCHEMA, Readout, build_prompt, chunks, load_readouts, parse_answers  # noqa: E402

OUT = HERE / "results" / "judged"


def judge_cell(client: Client, feature: str, reader: str, limit: int | None) -> dict:
    spec = FEATURES[feature]
    rows = load_readouts(feature, reader)[:limit]
    path = OUT / f"{feature}_{reader}.json"
    if path.exists():
        prev = json.loads(path.read_text())
        if prev.get("n_judged") == len(rows):
            print(f"skip {feature}/{reader} (complete)", flush=True)
            return prev

    def run(chunk: list[Readout]) -> list[dict]:
        raw = client.chat(build_prompt(spec["desc"], spec["values"], chunk), json_schema=SCHEMA)
        ans = parse_answers(raw, len(chunk))
        return [{"item": r.item, "value": r.value, "k": r.k, **ans[i]}
                for i, r in enumerate(chunk) if i in ans]

    with ThreadPoolExecutor(6) as ex:
        judged = [row for part in ex.map(run, chunks(rows)) for row in part]
    out = {"feature": feature, "reader": reader, "values": spec["values"], "n_rows": len(rows),
           "n_judged": len(judged), "judged": judged}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print(f"{feature}/{reader}: judged {len(judged)}/{len(rows)}  spend ${client.spend_usd:.2f}",
          flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", default="nla,ol42")
    ap.add_argument("--features", default=",".join(FEATURES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cap", type=float, default=10.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    client = Client(temperature=0.0, max_tokens=4000)
    for feature in a.features.split(","):
        for reader in a.readers.split(","):
            if client.spend_usd > a.cap:
                print(f"cap ${a.cap} reached; stopping", flush=True)
                return
            judge_cell(client, feature, reader, a.limit)
    print(f"done: {client.n_calls} calls, {client.n_failed} failed, ${client.spend_usd:.2f}", flush=True)


if __name__ == "__main__":
    main()
