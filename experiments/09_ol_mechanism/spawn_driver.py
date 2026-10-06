"""Server-side driver: spawn functions on the DEPLOYED app (no local client tie), record call ids."""
import json, sys, modal
from pathlib import Path
APP = "rfm"
OL_CHUNKS = {"codeNL": 2, "codeIN": 2, "col": 6, "tense": 7}
ids_f = Path("results/call_ids.json")
ids = json.loads(ids_f.read_text()) if ids_f.exists() else {}
f = lambda n: modal.Function.from_name(APP, n)
stage = sys.argv[1]
if stage == "prep":
    ids["prep"] = f("prep").spawn().object_id
elif stage == "run":
    calls = {f"behave:{S}": f("behave").spawn(S).object_id for S in OL_CHUNKS}
    calls["tense_r5"] = f("tense_r5").spawn().object_id
    for ad in ("on", "off"):
        for S, n in OL_CHUNKS.items():
            for i in range(n):
                calls[f"score_ol:{S}:{i}:{ad}"] = f("score_ol").spawn(S, i, n, ad).object_id
    ids["run"] = calls
elif stage == "status":
    def st(cid):
        try:
            modal.FunctionCall.from_id(cid).get(timeout=0); return "done"
        except TimeoutError: return "running"
        except Exception as e: return "ERR " + repr(e)[:150]
    if "prep" in ids: print("prep", st(ids["prep"]))
    for k, cid in ids.get("run", {}).items(): print(k, st(cid))
    sys.exit()
elif stage == "collect":
    out = {}
    for k, cid in ids.get("run", {}).items():
        try: out[k] = modal.FunctionCall.from_id(cid).get(timeout=5)
        except Exception as e: out[k] = {"error": repr(e)[:2000]}
    Path("results/run_summary.json").write_text(json.dumps(out, indent=1, default=str)); print("collected", len(out)); sys.exit()
ids_f.write_text(json.dumps(ids, indent=1)); print(stage, "spawned")
