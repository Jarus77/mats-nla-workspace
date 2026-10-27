"""Exp 23 bank (laptop): 100 random read points per kind (py, js, json, tex, md) in random PUBLIC documents (the exp 22
source rule), no selection by state; state labels from the prefix (lib23.labels); tokenised; written to
results/items_base.json (tracked) with build_info.json. Then modal23.py reads both readers once per point."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "22_public_bank"))
import build22 as B22  # noqa: E402
from lib23 import candidate_points, labels  # noqa: E402

KINDS = ("py", "js", "json", "tex", "md")
N_PER_KIND = 100
SEED = 2323
MAX_TOKENS = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(Path.home()))
    ap.add_argument("--arxiv", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    t0 = time.time()
    rng = random.Random(SEED)
    items, info = [], {"kinds": {}}
    for kind in KINDS:
        paths = B22.find_files(Path(a.home), kind, Path(a.arxiv) if a.arxiv else None)
        files = B22.load(paths, rng, 3000)
        rng.shuffle(files)
        n = 0
        for name, text, src in files:
            if n >= N_PER_KIND:
                break
            pts = candidate_points(text)
            if not pts:
                continue
            end = rng.choice(pts)
            prefix = text[:end]
            if len(prefix) > 1200:
                cut = prefix.rfind("\n", 0, len(prefix) - 1200)
                prefix = prefix[cut + 1:] if cut >= 0 else prefix[-1200:]
            ids = tok(prefix, add_special_tokens=False)["input_ids"]
            if len(ids) > MAX_TOKENS:
                ids = ids[-MAX_TOKENS:]
                prefix = tok.decode(ids)
            items.append({"id": f"rnd_{kind}_{n}", "feature": f"rnd_{kind}", "cluster": name, "value": "random",
                          "text": prefix, "ids": ids, "split": "select", "meta": {"kind": kind, "src": src, "labels": labels(prefix, kind)}})
            n += 1
        c = Counter()
        for it in items:
            if it["meta"]["kind"] == kind:
                for k, v in it["meta"]["labels"].items():
                    c[f"{k}={'yes' if (v is True or (isinstance(v, int) and v > 0)) else 'no'}"] += 1
        info["kinds"][kind] = {"n": n, "n_files": len(files), "label_counts": dict(sorted(c.items()))}
        print(kind, n, dict(sorted(c.items())), flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    blob = json.dumps(items)
    (HERE / "results" / "items_base.json").write_text(blob)
    info.update({"n_items": len(items), "sha256": hashlib.sha256(blob.encode()).hexdigest(), "seed": SEED, "wall_s": round(time.time() - t0, 1)})
    (HERE / "results" / "build_info.json").write_text(json.dumps(info, indent=1))
    print("sha256", info["sha256"])


if __name__ == "__main__":
    main()
