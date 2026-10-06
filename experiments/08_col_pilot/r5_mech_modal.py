"""08 round-5 follow-up (PILOT, EXCLUDED FROM MAIN ANALYSIS): mechanism check for the OL nl_cont cell.

Question: does OL's nl_cont shift ("- \\nthe" vs "- the" after the injected activation) need the OL LoRA,
or does the injected vector act through the plain Qwen LM in OL's carrier prompt? Same activations as
r5 (r5_spec.pt, col set, centred construction), scored with the adapter DISABLED (ol.base()), plus an
adapter-on re-score of the real cells (reproduction check), plus a J-Lens content-token leak check.

    modal run r5_mech_modal.py          -> rfm-results:/08_col_pilot/r5_mech.pt, copied to results/08_col_pilot/
"""
import time
from pathlib import Path

import modal

from env import OUT, VOLUMES, app_image

app = modal.App("mats-08-r5-mech")
L = 42
CONTENT3 = ["the", "and", "of"]  # the nl_cont continuation words
CONTENT10 = ["the", "a", "of", "and", "to", "in", "is", "was", "that", "for"]


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def mech() -> dict:
    import torch

    from mats.readers.jlens import load_jlens, token_variants
    from mats.readers.olens import load_olens
    from pilot_lib import eol_token_ids
    from r5_lib import build, conditions
    from r5_templates import OL_PAIRS, flat

    t0 = time.time()
    VOLUMES["/results"].reload()
    sp = torch.load(f"{OUT}/r5_spec.pt", weights_only=False)["col"]
    conds = conditions("col")
    ol = load_olens()
    tok = ol.tok

    # ---- J-Lens leak checks over every col condition (eol vs content token sets)
    jl = load_jlens(ol.model, tok, layers=[L])
    A = torch.cat([build(sp, c) for c in conds])
    eol = eol_token_ids(tok)
    sets = {"eol_broad": eol["broad"], "eol_narrow": eol["narrow"],
            "content3": token_variants(tok, CONTENT3), "content10": token_variants(tok, CONTENT10)}
    jout = {}
    for k, ids in sets.items():
        sc = torch.cat([jl.scores(A[i:i + 4096], L, ids=ids).max(1).values for i in range(0, A.shape[0], 4096)])
        jout[k] = sc.reshape(len(conds), -1)
    del jl
    torch.cuda.empty_cache()
    t_j = time.time() - t0

    # ---- OL nl_cont, adapter OFF (all conditions) and ON (zero + real cells)
    fp = flat({"nl_cont": OL_PAIRS["col"]["nl_cont"]})
    cands = [x for _, a, b in fp for x in (a, b)]

    def score(acts):  # noqa: ANN001, ANN202
        lp = ol.logprob_of_strings(acts, L, cands, batch=112)
        return torch.stack([lp[:, 2 * t] - lp[:, 2 * t + 1] for t in range(len(fp))])  # [T, N]

    with ol.base():
        lo_off = score(A).reshape(len(fp), len(conds), -1)
    on_idx = [i for i, c in enumerate(conds) if c[0] in ("zero", "real")]
    A_on = torch.cat([build(sp, conds[i]) for i in on_idx])
    lo_on = score(A_on).reshape(len(fp), len(on_idx), -1)
    out = {"conds": conds, "templates": fp, "jlens": jout, "jlens_sets_n": {k: len(v) for k, v in sets.items()},
           "ol_off": lo_off, "ol_on": lo_on, "on_idx": on_idx,
           "t": {"jlens": round(t_j, 1), "total": round(time.time() - t0, 1)}}
    torch.save(out, f"{OUT}/r5_mech.pt")
    VOLUMES["/results"].commit()
    return {"t": out["t"], "jlens_sets_n": out["jlens_sets_n"]}


@app.local_entrypoint()
def main() -> None:
    import subprocess

    print(mech.remote())
    dst = Path(__file__).resolve().parent / "results" / "08_col_pilot"
    subprocess.run(["modal", "volume", "get", "--force", "rfm-results", "08_col_pilot/r5_mech.pt", str(dst)], check=False)
