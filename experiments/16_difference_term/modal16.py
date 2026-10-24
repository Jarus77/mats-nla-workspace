"""Exp 16 Modal stages (app `pm9`):
  s1 (H200, NLA + AR): reconstruct OLens bullets through the NLA-AR, build the swap vectors,
                       score the NLA's prompted log-odds on every variant        -> s1_{f}.pt
  s2 (H100, OLens@42): OLens prompted log-odds on the same vectors                -> s2_{f}.pt
  analyze (CPU):       follow-AUROCs, component probes, FVEs                      -> analysis16_{f}.json

    modal run modal16.py --stage s1        then   --stage s2        then   --stage analyze
"""
import json
import os
import time
from pathlib import Path

import modal

from env import OUT, SRC, SRC14, VOLUMES, gpu_image, templates_path
from spec16 import LAYER, OL_GEN_LAYER, ROWS, SEED, VARIANTS

app = modal.App("pm9")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GPU = dict(volumes=VOLUMES, timeout=3 * 3600, memory=131072, single_use_containers=True, max_containers=int(os.environ.get("RFM_MAXC", "2")))  # GPU cap: 2 on the main workspace (RFM_MAXC overrides)


def _row(feature: str):  # noqa: ANN202
    import numpy as np
    import torch

    src = SRC[ROWS[feature]["src"]]
    items = json.load(open(f"{src}/gen_ol_{feature}.json"))["items"]
    blob = torch.load(f"{src}/acts_{feature}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items]
    y = np.array([it["value"] == ROWS[feature]["values"][1] for it in items], int)
    return items, blob["acts"][LAYER].float(), y


def _pairs(feature: str, reader: str) -> list[list[str]]:
    exp = {"13": "13_new_rows", "07": "07_template_mining"}[ROWS[feature]["src"]]
    t = json.loads(templates_path(exp).read_text())["templates"][reader][feature]
    a, b = ROWS[feature]["values"]
    return [[r["id"], r["candidates"][a], r["candidates"][b]] for r in [t["fixed_label"]] + t["selected"]]


def _done(path: str) -> bool:
    VOLUMES["/results"].reload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    return Path(path).exists()


def _save(path: str, obj) -> None:  # noqa: ANN001
    import torch

    torch.save(obj, path)
    VOLUMES["/results"].commit()


@app.function(image=gpu_image, gpu="H200", **GPU)
def u4yd(feature: str) -> dict:
    import torch

    from lib16 import construct, fve_rows, partners
    from mats.readers import nla as NL

    t0 = time.time()
    path = f"{OUT}/s1_{feature}.pt"
    if _done(path):
        return {"feature": feature, "skipped": True}
    items, h, y = _row(feature)
    ids = [it["id"] for it in items]
    gol = json.load(open(f"{SRC14}/gen14_ol_{feature}.json"))
    assert gol["ids"] == ids
    r14 = torch.load(f"{SRC14}/recon14_nla_{feature}.pt", weights_only=False)
    assert r14["ids"] == ids
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=True)
    bullets = [gol["gens"][OL_GEN_LAYER][i] for i in ids]            # [N][k] strings
    k = len(bullets[0])
    flat = [b for row in bullets for b in row]
    r_ol = nla.reconstruct(flat, batch=16).float().reshape(len(ids), k, -1).mean(1)
    r_nla = r14["recon"].float().mean(1)
    j = partners(y, [it["cluster"] for it in items], SEED)
    vecs = construct(h, r_ol, r_nla, j)
    pairs = _pairs(feature, "nla")
    cands = [x for _, a, b in pairs for x in (a, b)]
    lo = {}
    for v in VARIANTS:
        lp = NL.logprob_of_strings_batch(nla, vecs[v], cands, batch=112)
        lo[v] = torch.stack([lp[:, 2 * t + 1] - lp[:, 2 * t] for t in range(len(pairs))])  # B - A
    fve = {"r_ol": fve_rows(r_ol, h), "r_nla": fve_rows(r_nla, h), "ol_only_vs_h": fve_rows(vecs["ol_only"], h)}
    _save(path, {"ids": ids, "y": y, "j": j, "h": h, "r_ol": r_ol, "r_nla": r_nla, "vecs": vecs,
                 "nla_lo": lo, "pairs": pairs, "bullets": bullets, "fve": fve,
                 "clusters": [it["cluster"] for it in items]})
    return {"feature": feature, "fve": fve, "n": len(ids), "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu="H100", **GPU)
def h9le(feature: str) -> dict:
    import torch

    from mats.readers.olens import load_olens

    t0 = time.time()
    path = f"{OUT}/s2_{feature}.pt"
    if _done(path):
        return {"feature": feature, "skipped": True}
    s = torch.load(f"{OUT}/s1_{feature}.pt", weights_only=False)
    ol = load_olens()
    pairs = _pairs(feature, "ol")
    cands = [x for _, a, b in pairs for x in (a, b)]
    lo = {}
    for v in VARIANTS:
        lp = ol.logprob_of_strings(s["vecs"][v], LAYER, cands, batch=112)
        lo[v] = torch.stack([lp[:, 2 * t + 1] - lp[:, 2 * t] for t in range(len(pairs))])
    _save(path, {"ids": s["ids"], "ol_lo": lo, "pairs": pairs})
    return {"feature": feature, "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, volumes=VOLUMES, timeout=1800, memory=32768, cpu=4)
def t6mw(feature: str) -> dict:
    import numpy as np
    import torch

    from lib16 import decompose, follow
    from mats import probes as P

    VOLUMES["/results"].reload()
    s = torch.load(f"{OUT}/s1_{feature}.pt", weights_only=False)
    s2p = f"{OUT}/s2_{feature}.pt"
    ol_lo = torch.load(s2p, weights_only=False)["ol_lo"] if Path(s2p).exists() else None
    y, j, groups = s["y"], s["j"], np.array(s["clusters"])
    comp = decompose(s["h"], s["r_ol"], s["r_nla"])
    probes = {k: float(P.cv_auroc(X, y, groups)["auroc"]) for k, X in
              {"h": s["h"], "r_ol": s["r_ol"], "r_nla": s["r_nla"], "res": comp["res"], "resn": comp["resn"]}.items()}
    out = {"feature": feature, "role": ROWS[feature]["role"], "fve": s["fve"], "probe_auroc": probes, "readers": {}}
    for name, lo in (("nla", s["nla_lo"]), ("ol", ol_lo)):
        if lo is None:
            continue
        out["readers"][name] = {v: {"mean": follow(lo[v].mean(0).numpy(), y, j),
                                    "fixed": follow(lo[v][0].numpy(), y, j)} for v in VARIANTS}
    Path(f"{OUT}/analysis16_{feature}.json").write_text(json.dumps(out, indent=1, default=float))
    VOLUMES["/results"].commit()
    return out


@app.function(image=gpu_image, volumes=VOLUMES, timeout=600, memory=8192, cpu=2)
def c1rp(feature: str) -> dict:
    """Warms the image build and checks the inputs exist (no GPU)."""
    VOLUMES["/results"].reload()
    src = SRC[ROWS[feature]["src"]]
    need = [f"{src}/gen_ol_{feature}.json", f"{src}/acts_{feature}.pt", f"{SRC14}/recon14_nla_{feature}.pt", f"{SRC14}/gen14_ol_{feature}.json"]
    return {"feature": feature, "missing": [p for p in need if not Path(p).exists()], "skipped": True}


@app.local_entrypoint()
def main(stage: str = "s1", features: str = "") -> None:
    feats = features.split(",") if features else list(ROWS)
    RES.mkdir(exist_ok=True)
    fn = {"s1": u4yd, "s2": h9le, "analyze": t6mw, "noop": c1rp}[stage]
    for r in fn.map(feats, return_exceptions=True):
        if isinstance(r, Exception):
            print("ERROR", stage, r)
            continue
        print(stage, json.dumps(r, default=str)[:600])
        if not r.get("skipped"):
            (RES / f"{stage}16_{r['feature']}.json").write_text(json.dumps(r, indent=1, default=float))
