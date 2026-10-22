"""Server-side driver: spawn operand_gate on the DEPLOYED app "rfm" (no local client tie); status; collect."""
import json
import sys
from pathlib import Path

import modal

APP = "rfm"
ids_f = Path("results/call_ids.json")
ids = json.loads(ids_f.read_text()) if ids_f.exists() else {}
stage = sys.argv[1]
if stage == "spawn":
    ids["operand_gate"] = modal.Function.from_name(APP, "operand_gate").spawn().object_id
    ids_f.parent.mkdir(exist_ok=True)
    ids_f.write_text(json.dumps(ids, indent=1))
    print("spawned", ids["operand_gate"])
elif stage == "status":
    try:
        modal.FunctionCall.from_id(ids["operand_gate"]).get(timeout=0)
        print("done")
    except TimeoutError:
        print("running")
    except Exception as e:  # noqa: BLE001
        print("ERR", repr(e)[:300])
elif stage == "collect":
    try:
        out = modal.FunctionCall.from_id(ids["operand_gate"]).get(timeout=5)
    except Exception as e:  # noqa: BLE001
        out = {"error": repr(e)[:4000]}
    Path("results/run_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print("collected")
