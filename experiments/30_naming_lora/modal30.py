"""Exp 30 (app `j9rw`): can the NLA be taught to name what it carries, without losing reconstruction? A supervised
naming term on the NLA's own LoRA (notes/draft_critique_intervention.md, design (b)). Stages, one GPU at a time
(run30.sh):
  k3vd (CPU):  the frozen data file                                   -> data30.json, data30.sha256
  b4mt (H200): train one arm, --arm S (explanation + statement) or C  -> adapter_{arm}/, train30_{arm}.json, train30_{arm}.done
  l7kz (H200): eval one arm, --arm base | S | C, AR loaded            -> eval30_{arm}.json
    cd experiments/30_naming_lora && modal run --detach modal30.py --stage train --arm S
"""
import json
import time
from pathlib import Path

import modal

from env import OUT, OUT22, OUT23, OUT25, VOLUMES, gpu_image
from lib30 import ALL_ROWS, TRAIN_ROWS, trained_adapter_path

app = modal.App("j9rw")
LAYER = 42
SEED = 30
GPU = dict(volumes=VOLUMES, timeout=4 * 3600, memory=131072, single_use_containers=True, max_containers=1)
CPU = dict(volumes=VOLUMES, timeout=1800, memory=8192, max_containers=1)


def _acts_by_id(rows) -> dict:  # noqa: ANN001
    """item id -> true layer-42 activation (fp32) from exp 22's acts22_{row}.pt."""
    import torch

    out = {}
    for row in rows:
        items = [it for it in json.load(open(f"{OUT22}/items_base.json")) if it["feature"] == row]
        blob = torch.load(f"{OUT22}/acts22_{row}.pt", weights_only=False)
        assert blob["ids"] == [it["id"] for it in items], f"{row}: acts22 ids differ from items_base"
        acts = blob["acts"][LAYER].float()
        out.update({iid: acts[i] for i, iid in enumerate(blob["ids"])})
    return out


def _data() -> dict:
    return json.load(open(f"{OUT}/data30.json"))


@app.function(image=gpu_image, **CPU)
def k3vd(force: bool = False) -> dict:
    import data30 as D

    VOLUMES["/results"].reload()
    if Path(f"{OUT}/data30.json").exists() and not force:
        return {"skipped": True, "sha256": Path(f"{OUT}/data30.sha256").read_text().split()[0]}
    info = D.main(OUT, f"{OUT22}/items_base.json", OUT25)
    VOLUMES["/results"].commit()
    return info


@app.function(image=gpu_image, gpu="H200", **GPU)
def b4mt(arm: str, steps: int = 400, batch: int = 8, micro: int = 8, lr: float = 1e-5) -> dict:
    import torch

    import train30 as T
    from mats.readers import nla as NL

    if arm not in ("S", "C"):
        raise ValueError(f"arm must be S or C, got {arm!r}")
    t0 = time.time()
    VOLUMES["/results"].reload()
    done_marker = Path(f"{OUT}/train30_{arm}.done")
    if done_marker.exists():
        return {"skipped": True, "arm": arm}
    data = _data()
    state = T.load_state(OUT, arm)
    resume = state is not None and state["step"] > 0 and Path(trained_adapter_path(OUT, arm)).is_dir()
    adapter = trained_adapter_path(OUT, arm) if resume else NL.DEFAULT_ADAPTER
    torch.manual_seed(SEED)
    nla = NL.load_nla(adapter=adapter, load_ar=False, trainable=True)
    acts_by_id = _acts_by_id(TRAIN_ROWS)
    acts = torch.stack([acts_by_id[t["id"]] for t in data["train"]])
    targets = [t[f"target_{arm}"] for t in data["train"]]
    print(f"arm {arm}: {len(targets)} items, {nla._vec['n_trainable']:,} trainable params, resume={resume}, "
          f"adapter={adapter}, load {time.time() - t0:.0f}s")
    st = T.train(nla, acts, targets, OUT, arm, steps=steps, batch=batch, micro=micro, lr=lr, seed=SEED,
                 on_save=VOLUMES["/results"].commit)
    if st["done"]:
        done_marker.write_text(json.dumps({"step": st["step"], "wall_s": st["wall_s"]}))
        VOLUMES["/results"].commit()
    return {"arm": arm, "step": st["step"], "done": st["done"], "losses": st["losses"][-5:], "train_wall_s": st["wall_s"],
            "n_trainable": nla._vec["n_trainable"], "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu="H200", **GPU)
def l7kz(arm: str, k: int = 1) -> dict:
    import torch

    from feature_spec import FIXED, fill
    from lib23 import naming_by_state
    from lib30 import statement_accuracy
    from mats.readers import nla as NL

    t0 = time.time()
    VOLUMES["/results"].reload()
    path = Path(f"{OUT}/eval30_{arm}.json")
    if path.exists():
        return {"skipped": True, "arm": arm}
    adapter = NL.DEFAULT_ADAPTER if arm == "base" else trained_adapter_path(OUT, arm)
    if arm != "base" and not Path(adapter).is_dir():
        raise FileNotFoundError(f"{adapter} missing: train arm {arm} first")
    torch.manual_seed(SEED)
    nla = NL.load_nla(adapter=adapter, load_ar=True)
    data = _data()
    acts_by_id = _acts_by_id(ALL_ROWS)
    res = {"arm": arm, "adapter": adapter, "k": k, "data_sha256": Path(f"{OUT}/data30.sha256").read_text().split()[0], "rows": {}}
    gate_acts, gate_expl = [], []
    for row in ALL_ROWS:
        ev = [e for e in data["eval"] if e["row"] == row]
        X = torch.stack([acts_by_id[e["id"]] for e in ev])
        raw = nla.verbalize_raw(X, k=k, batch=64)
        expl = [NL.extract_explanation(r[0]) for r in raw]
        texts = {e["id"]: (x if x is not None else "") for e, x in zip(ev, expl)}
        pair = {v: fill(FIXED["nla"][row], v) for v in FIXED["nla"][row]["fillers"]}
        acc = statement_accuracy(texts, ev, pair)
        f = nla.fve_on(X, k=1, explanations=[[x] for x in expl])
        res["rows"][row] = {"trained": row in TRAIN_ROWS, "n": len(ev), "statement_pair": pair, **{c: acc[c] for c in acc if c != "per_item"},
                            "fve": float(torch.nanmean(f["per_example"])), "fve_baseline": f["baseline"], "n_failed": f["n_failed"],
                            "per_item": {e["id"]: {"value": e["value"], "verdict": acc["per_item"][e["id"]], "fve": float(f["per_example"][i]),
                                                   "raw": raw[i][0]} for i, e in enumerate(ev)}}
        if row in TRAIN_ROWS:
            gate_acts.append(X)
            gate_expl += [[x] for x in expl]
        print(f"{row}: acc {acc['accuracy']:.2f} naming {acc['naming_rate']:.2f} fve {res['rows'][row]['fve']:.3f}")
    g = nla.fve_on(torch.cat(gate_acts), k=1, explanations=gate_expl)
    res["gate"] = {"n": len(gate_expl), "fve": float(torch.nanmean(g["per_example"])), "baseline": g["baseline"], "n_failed": g["n_failed"]}
    # exp 23's random read points (P4)
    items23 = json.load(open(f"{OUT23}/items_base.json"))
    blob = torch.load(f"{OUT23}/acts23.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items23]
    raw23 = nla.verbalize_raw(blob["acts"].float(), k=1, batch=64)
    t23 = {it["id"]: (NL.extract_explanation(raw23[i][0]) or "") for i, it in enumerate(items23)}
    pair = {v: fill(FIXED["nla"]["py_string"], v) for v in FIXED["nla"]["py_string"]["fillers"]}
    code = [it for it in items23 if it["meta"]["kind"] in ("py", "js", "json")]
    stmt_items = [{"id": it["id"], "value": "inside" if it["meta"]["labels"]["in_string"] else "outside"} for it in items23]
    res["random_points"] = {"n": len(items23), "strict_regex": naming_by_state(t23, items23, "in_string"),
                            "strict_regex_code_kinds": naming_by_state(t23, code, "in_string"),
                            "statement_check": {c: statement_accuracy(t23, stmt_items, pair)[c] for c in ("n", "correct", "wrong", "none", "accuracy", "naming_rate")},
                            "readouts": {it["id"]: raw23[i][0] for i, it in enumerate(items23)}}
    res["wall_s"] = round(time.time() - t0, 1)
    path.write_text(json.dumps(res, indent=1))
    VOLUMES["/results"].commit()
    return {"arm": arm, "gate_fve": res["gate"]["fve"], "rows": {r: (v["accuracy"], v["naming_rate"], round(v["fve"], 3)) for r, v in res["rows"].items()},
            "random_points": res["random_points"]["strict_regex"], "wall_s": res["wall_s"]}


@app.local_entrypoint()
def main(stage: str = "data", arm: str = "S", steps: int = 400, batch: int = 8, micro: int = 8, force: bool = False) -> None:
    if stage == "data":
        r = k3vd.remote(force)
    elif stage == "train":
        r = b4mt.remote(arm, steps, batch, micro)
    elif stage == "eval":
        r = l7kz.remote(arm)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    print(stage, arm, json.dumps(r, indent=1, default=str)[:3000])
