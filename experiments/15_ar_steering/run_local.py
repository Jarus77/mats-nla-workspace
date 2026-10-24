"""Run the exp 15 stages on a bare GPU machine (RunPod etc.) without Modal: same code, called locally.

    cd experiments/15_ar_steering
    python run_local.py --stage prep
    python run_local.py --stage behave
    python run_local.py --stage score_nla
    python run_local.py --stage score_ol
    python run_local.py --stage analyze

Needs /results/<inputs> (experiments/handoff_inputs/README.md), the weights in /weights/hf (HF cache
layout; infra/download_weights.py lists the repos) and the pinned packages (env.py _PINS). A stage skips
a row whose output already exists under /results/15_ar_steering/. prep needs ~92 GB of GPU memory
(NLA + its reconstructor): one H200 / B200, or an 80 GB card will OOM. behave / score_* fit an 80 GB card.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
os.environ.setdefault("RFM_LOCAL", "1")
os.environ.setdefault("HF_HOME", "/weights/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for p in (HERE, REPO / "infra", REPO / "src"):
    sys.path.insert(0, str(p))

import modal15 as M  # noqa: E402

STAGES = {"prep": M.k7q, "behave": M.m3zd, "score_nla": M.v8t, "score_ol": M.w2hx,
          "analyze": M.p5nq}


def _gpu_check(stage: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    need = 120 if stage == "prep" else 75
    if gb < need:
        raise SystemExit(f"stage {stage} needs ~{need} GB of GPU memory, found {gb:.0f} GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--features", default="", help="comma-separated subset of " + ",".join(M.ROWS))
    a = ap.parse_args()
    feats = a.features.split(",") if a.features else list(M.ROWS)
    if a.stage != "analyze":
        _gpu_check(a.stage)
    args = [(f, c) for f in feats for c in range(M.N_BEHAVE_CHUNKS)] if a.stage == "behave" else [(f, 0, 0) for f in feats] if a.stage.startswith("score") else feats
    M.RES.mkdir(exist_ok=True)
    Path(M.OUT).mkdir(parents=True, exist_ok=True)
    for x in args:
        r = STAGES[a.stage].local(x)
        print(a.stage, json.dumps(r, default=str)[:600], flush=True)
        if isinstance(r, dict) and not r.get("skipped") and a.stage in ("prep", "analyze"):
            (M.RES / f"{a.stage}15_{r['feature']}.json").write_text(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
