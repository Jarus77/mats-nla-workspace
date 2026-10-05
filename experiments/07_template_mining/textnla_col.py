"""Fable round 5: natural col vs TextNLA on the SAME selection items with the SAME frozen templates.

Local Qwen (H100): TextNLALocal (base Qwen3.6-27B, no activation, NLA warm-start instruction,
mode="prompt") scores the frozen NLA and OL candidate pairs (selected + secondary + fixed) given
the passage text up to the read token. Reads templates_frozen.json; never writes it.

    modal run textnla_col.py          -> results/textnla_local_<feature>.json
"""

import json
from pathlib import Path

import modal

from env import OUT, VOLUMES, gpu_image

app = modal.App("mats-07-textnla-col")
HERE = Path(__file__).resolve().parent
FEATS = ("nj1_col", "nj1_col_break")


def frozen_templates(fz: dict, feature: str) -> list[dict]:
    out = []
    for reader in ("nla", "ol"):
        e = fz["templates"][reader][feature]
        for key in ("selected", "secondary_numeric"):
            out += [{**t, "reader": reader, "role": key} for t in e.get(key, [])]
        for key in ("fixed_label", "fixed_numeric"):
            if key in e:
                out.append({**e[key], "reader": reader, "role": key})
    return out


@app.function(image=gpu_image.add_local_file(str(HERE / "templates_frozen.json"), "/root/templates_frozen.json"),
              volumes=VOLUMES, gpu="H100", timeout=2 * 3600, memory=131072, max_inputs=1)
def score(feature: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats.textnla import TextNLALocal, nla_instruction

    VOLUMES["/results"].reload()
    fz = json.load(open("/root/templates_frozen.json"))
    temps = frozen_templates(fz, feature)
    items = {it["id"]: it for it in json.load(open(f"{OUT}/items_base.json")) +
             json.load(open(f"{OUT}/items_extra.json"))}
    ids = fz["items"][feature]["select"]
    vals = fz["meta"]["values"][feature]
    texts = [items[i]["text"] for i in ids]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    tn = TextNLALocal(model, tok, instruction=nla_instruction())
    cands = [c for t in temps for c in (t["candidates"][vals[0]], t["candidates"][vals[1]])]
    lp = tn.logprob_of_strings(texts, cands, mode="prompt", batch=max(16, len(cands)))
    lo = lp[:, 0::2] - lp[:, 1::2]
    return {"feature": feature, "ids": ids, "y": [int(items[i]["value"] == vals[0]) for i in ids],
            "lo": {t["id"]: lo[:, k].tolist() for k, t in enumerate(temps)},
            "templates": [{k: t[k] for k in ("id", "reader", "role", "candidates")} for t in temps]}


@app.local_entrypoint()
def main() -> None:
    for r in score.map(FEATS):
        (HERE / "results" / f"textnla_local_{r['feature']}.json").write_text(json.dumps(r))
        print(r["feature"], "done")
