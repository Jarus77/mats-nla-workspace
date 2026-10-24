"""Run the exp 16 stages on a bare GPU machine (RunPod etc.) without Modal: same code, called locally.

    cd experiments/16_difference_term
    python run_local.py --stage s1        # needs ~92 GB GPU memory (NLA + reconstructor): H200 / B200
    python run_local.py --stage s2        # fits an 80 GB card
    python run_local.py --stage analyze   # CPU

Needs /results/<inputs> (experiments/handoff_inputs/README.md) and the weights in /weights/hf. A stage
skips a row whose output already exists under /results/16_difference_term/.
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

import modal16 as M  # noqa: E402

STAGES = {"s1": M.u4yd, "s2": M.h9le, "analyze": M.t6mw}


def _gpu_check(stage: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    need = 120 if stage == "s1" else 75
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
    M.RES.mkdir(exist_ok=True)
    Path(M.OUT).mkdir(parents=True, exist_ok=True)
    for f in feats:
        r = STAGES[a.stage].local(f)
        print(a.stage, json.dumps(r, default=str)[:600], flush=True)
        if isinstance(r, dict) and not r.get("skipped"):
            (M.RES / f"{a.stage}16_{r['feature']}.json").write_text(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
