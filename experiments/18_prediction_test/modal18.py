"""Exp 18 Modal stages (app `h5tq`, opaque function names). Stage 1 = the item bank:
  build  w3ke (CPU): synthetic rows + stance row (fresh wikitext paragraphs) + the two exp 07 control rows
                     (language FR/DE, quote state), tokenised, split 20+20 mine / 80+80 select, frozen with a
                     sha256 -> /results/18_prediction_test/items_base.json, results/build_info.json

    cd experiments/18_prediction_test && modal run modal18.py --stage build
"""
import hashlib
import json
import time
from pathlib import Path

import modal

from env import OUT, OUT07, OUT08, OUT13, VOLUMES, build_image
from feature_spec18 import FEATURES, N_GEN, N_MINE_PER_VALUE, N_SELECT_PER_VALUE

app = modal.App("h5tq")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def _load_07_items_build():  # noqa: ANN202
    import importlib.util
    import sys

    def load(name: str, path: str):  # noqa: ANN202
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    saved = sys.modules.get("feature_spec")
    sys.modules["feature_spec"] = load("feature_spec07", "/root/ext07/feature_spec.py")
    try:
        return load("items_build07", "/root/ext07/items_build.py")
    finally:
        if saved is not None:
            sys.modules["feature_spec"] = saved


def assign_splits(items: list[dict], rng, n_mine: int, n_sel: int) -> list[dict]:  # noqa: ANN001
    """Per value: shuffle, first n_mine -> mine, next n_sel -> select (exp 07's rule)."""
    out, by_val = [], {}
    for it in items:
        by_val.setdefault(it["value"], []).append(it)
    for _, lst in sorted(by_val.items()):
        rng.shuffle(lst)
        out += [{**it, "split": "mine" if k < n_mine else "select"} for k, it in enumerate(lst[: n_mine + n_sel])]
    return out


@app.function(image=build_image, volumes=VOLUMES, timeout=3 * 3600, memory=32768, cpu=8)
def w3ke() -> dict:
    import collections
    import random

    from transformers import AutoTokenizer

    import items_build18 as IB

    t0 = time.time()
    VOLUMES["/results"].reload()
    IB07 = _load_07_items_build()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    s1 = IB07.load_02("s1_data")
    paras = IB07.fresh_paragraphs(s1)
    used = set()
    for path in (f"{OUT07}/items_base.json", f"{OUT07}/items_extra.json", f"{OUT08}/items.json", f"{OUT13}/items_base.json"):
        try:
            used |= {IB07._norm(it["text"]) for it in json.load(open(path)) if it.get("text")}
        except FileNotFoundError:
            pass
    paras = [p for p in paras if IB07._norm(p) not in used]
    rng = random.Random(IB.SEED)
    NM, NS = N_MINE_PER_VALUE, N_SELECT_PER_VALUE
    items: list[dict] = []
    for f, builder in IB.BUILDERS.items():
        items += assign_splits(builder(rng, N_GEN), rng, NM, NS)
    items += assign_splits(IB.build_stance_word(paras[0:400], rng), rng, NM, NS)
    items = [{**it, "ids": tok(it["text"], add_special_tokens=False)["input_ids"]} for it in items]
    lang = [{**it, "feature": "ij2_lang"} for it in IB07.build_lang(tok, rng)]          # exp 07 generator, fresh draw
    quote = IB.length_band([{**it, "feature": "bk_quote"} for it in IB07.build_quote(tok, paras[400:2400], rng)], 40, 250)
    items += assign_splits(lang, rng, NM, NS) + assign_splits(quote, rng, NM, NS)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    blob = json.dumps(items)
    Path(f"{OUT}/items_base.json").write_text(blob)
    VOLUMES["/results"].commit()
    summ = collections.Counter((it["feature"], it["split"], it["value"]) for it in items)
    by_val = {}
    for it in items:
        by_val.setdefault(it["feature"] + "|" + it["value"], []).append(len(it["ids"]))
    toklen = {f: sorted(len(it["ids"]) for it in items if it["feature"] == f) for f in {it["feature"] for it in items}}
    return {"n_items": len(items), "n_paragraphs": len(paras), "sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "counts": {"|".join(map(str, k)): v for k, v in sorted(summ.items())},
            "token_len_q10_q50_q90": {f: [L[len(L) // 10], L[len(L) // 2], L[9 * len(L) // 10]] for f, L in toklen.items()},
            "median_token_len_by_value": {k: sorted(v)[len(v) // 2] for k, v in sorted(by_val.items())},
            "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(stage: str = "build") -> None:
    RES.mkdir(exist_ok=True)
    fn = {"build": w3ke}[stage]
    r = fn.remote()
    print(stage, json.dumps(r, indent=1, default=str)[:3000])
    (RES / "build_info.json").write_text(json.dumps(r, indent=1, default=str))
