"""TextNLA with Gemini 3.8 Flash (OpenRouter): same user turns as the Qwen arms (data/jobs.json),
k=1, T=1, reasoning effort minimal, max_tokens 256 (= the readers' 256 new tokens). Resumable cache: data/gemini_gen.jsonl.

    python gemini_gen.py [--limit N]
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from common import HERE
from llm import Client

CAP_USD = 8.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = json.loads((HERE / "data" / "jobs.json").read_text(encoding="utf-8"))
    cache = HERE / "data" / "gemini_gen.jsonl"
    done = set()
    if cache.exists():
        done = {json.loads(l)["key"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [j for j in jobs if j["key"] not in done][: a.limit or None]
    c = Client(max_tokens=256)

    def run(j: dict) -> dict | None:
        if c.spend_usd > CAP_USD:
            return None
        out = c.chat(j["user"])
        return None if out is None else {"key": j["key"], "text": out}

    with ThreadPoolExecutor(16) as ex, cache.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(ex.map(run, todo)):
            if r:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(i, len(todo), f"${c.spend_usd:.3f}", flush=True)
    print(dict(n_calls=c.n_calls, n_failed=c.n_failed, spend_usd=round(c.spend_usd, 4)))
    log = HERE / "data" / "spend_log.jsonl"
    with log.open("a") as fh:
        fh.write(json.dumps({"step": "gemini_gen", "n_calls": c.n_calls, "usd": c.spend_usd}) + "\n")


if __name__ == "__main__":
    main()
