"""Server-side driver: spawn/status/collect a named function on the DEPLOYED app "rfm"."""
import json
import sys
from pathlib import Path

import modal

APP = "rfm"
ids_f = Path("results/phaseB/call_ids.json")
ids = json.loads(ids_f.read_text()) if ids_f.exists() else {}
stage, fn = sys.argv[1], sys.argv[2]
if stage == "spawn":
    ids[fn] = modal.Function.from_name(APP, fn).spawn().object_id
    ids_f.parent.mkdir(parents=True, exist_ok=True)
    ids_f.write_text(json.dumps(ids, indent=1))
    print("spawned", fn, ids[fn])
elif stage == "status":
    try:
        modal.FunctionCall.from_id(ids[fn]).get(timeout=0)
        print("done")
    except TimeoutError:
        print("running")
    except Exception as e:  # noqa: BLE001
        print("ERR", repr(e)[:300])
elif stage == "collect":
    try:
        out = modal.FunctionCall.from_id(ids[fn]).get(timeout=5)
    except Exception as e:  # noqa: BLE001
        out = {"error": repr(e)[:4000]}
    Path(f"results/phaseB/{fn}_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print("collected", fn)
