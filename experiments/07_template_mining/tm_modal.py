"""07 template mining: Modal stages.

  build   (CPU, network): new items for all 8 features (items_build.py) + 02's POS lexicon (NJ3).
  gen_ol  (H100, .map over features): capture block 40/42/44 at the read token for every item;
          NJ3 labels from the model's next-token POS mass; k=8 OL readouts at L42/40/44 for the
          MINING items; J-Lens top-50 at L42 for the mining items.
  gen_nla (H200, .map): k=8 NLA@42 (iter_000400) readouts for the mining items.
  score_ol / score_nla (H100 / H200, .map): paired log-odds of every candidate template on the
          SELECTION items (all candidates of one activation in one forward batch).

    modal run tm_modal.py --stage build|gen|score
"""

import json
import time
from pathlib import Path

import modal

from env import OUT, VOLUMES, build_image, gpu_image
from feature_spec import FEATURES, K_READOUTS, NLA_LAYER, OL_LAYERS

app = modal.App("mats-07-template-mining")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
POS_ADJ_SHARE, POS_MIN_MASS = 0.6, 0.3


# ============================================================================ build (CPU)
@app.function(image=build_image, volumes=VOLUMES, timeout=3 * 3600, memory=32768, cpu=8)
def build() -> dict:
    import collections
    import pickle
    import random
    import spacy
    from transformers import AutoTokenizer

    import items_build as IB
    from feature_spec import N_MINE_PER_VALUE as NM
    from feature_spec import N_SELECT_PER_VALUE as NS

    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    s1 = IB.load_02("s1_data")
    f02 = IB.load_02("features")
    rng = random.Random(IB.SEED)
    paras = IB.fresh_paragraphs(s1)
    print("fresh paragraphs", len(paras), flush=True)
    blk = {"col": paras[0:600], "tense": paras[600:2600], "nj3": paras[2600:3400],
           "wordpos": paras[3400:3800], "caps": paras[3800:4100]}
    items: list[dict] = []
    items += IB.assign_splits(IB.build_col(tok, s1, blk["col"], rng), rng, NM, NS)
    items += IB.assign_splits(IB.build_lang(tok, rng), rng, NM, NS)
    nlp = spacy.load("en_core_web_lg")
    items += IB.assign_splits(IB.build_tense(tok, nlp, s1, blk["tense"], rng), rng, NM, NS)
    items += IB.assign_splits(IB.build_indent(tok, rng, 2 * (NM + NS) + 60), rng, NM, NS)
    items += IB.assign_splits(IB.build_wordpos(tok, blk["wordpos"], rng), rng, NM, NS)
    items += IB.assign_splits(IB.build_listidx(tok, rng, 2 * (NM + NS) + 20), rng, NM, NS)
    items += IB.assign_splits(IB.build_caps(tok, blk["caps"], rng), rng, NM, NS)
    items += IB.split_code(IB.build_code(tok, rng, NM + NS + 10), rng)
    pool = IB.build_predpos_pool(tok, blk["nj3"])

    # 02's NJ3 lexicon recipe (s2c): word-initial tokens whose next-word UPOS is >=80% one class
    prose02 = pickle.load(open(f"{IB.OUT02}/data/prose.pkl", "rb"))
    cnt = collections.defaultdict(collections.Counter)
    for it in prose02:
        for j, lab in enumerate(it["next_pos"]):
            if lab and j + 1 < len(it["ids"]):
                cnt[it["ids"][j + 1]][lab] += 1
    lex = {}
    for t, c in cnt.items():
        tot = sum(c.values())
        lab, k = c.most_common(1)[0]
        if tot >= 3 and k / tot >= 0.8 and lab in f02.POS_CLASSES:
            lex[int(t)] = lab
    Path(OUT).mkdir(parents=True, exist_ok=True)
    json.dump(items, open(f"{OUT}/items_base.json", "w"))
    json.dump(pool, open(f"{OUT}/nj3_pool.json", "w"))
    json.dump(lex, open(f"{OUT}/pos_lexicon_02.json", "w"))
    VOLUMES["/results"].commit()
    summ = collections.Counter((it["feature"], it["split"], it["value"]) for it in items)
    info = {"n_items": len(items), "nj3_pool": len(pool), "lexicon": len(lex),
            "lexicon_classes": collections.Counter(lex.values()),
            "counts": {"|".join(map(str, k)): v for k, v in sorted(summ.items())}}
    print(json.dumps(info, indent=1, default=str))
    return info


@app.function(image=build_image, volumes=VOLUMES, timeout=3 * 3600, memory=32768, cpu=8)
def build_extra() -> dict:
    """Round-4 (Fable) features: col break decision, Item-N list index, in-quote, agreement.
    Uses paragraph blocks disjoint from build()'s blocks (same fresh pool, same seed)."""
    import collections
    import random

    from transformers import AutoTokenizer

    import items_build as IB
    from feature_spec import N_MINE_PER_VALUE as NM
    from feature_spec import N_SELECT_PER_VALUE as NS

    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    s1 = IB.load_02("s1_data")
    rng = random.Random(IB.SEED + 4)
    paras = IB.fresh_paragraphs(s1)
    base_ids = {it["id"] for it in json.load(open(f"{OUT}/items_base.json"))}
    items: list[dict] = []
    items += IB.assign_splits(IB.build_col_break(tok, s1, paras[4100:5300], rng), rng, NM, NS)
    items += IB.assign_splits(IB.build_listnum(tok, rng, 2 * (NM + NS) + 20), rng, NM, NS)
    items += IB.assign_splits(IB.build_quote(tok, paras[5300:12000], rng), rng, NM, NS)
    gen, num = IB.build_agreement(tok, rng)
    items += IB.assign_splits(gen, rng, NM, NS) + IB.assign_splits(num, rng, NM, NS)
    assert not ({it["id"] for it in items} & base_ids)
    json.dump(items, open(f"{OUT}/items_extra.json", "w"))
    VOLUMES["/results"].commit()
    summ = collections.Counter((it["feature"], it["split"], it["value"]) for it in items)
    info = {"n_items": len(items), "counts": {"|".join(map(str, k)): v for k, v in sorted(summ.items())}}
    print(json.dumps(info, indent=1))
    return info


# ============================================================================ helpers (GPU)
def _load_items(feature: str) -> list[dict]:
    p = Path(f"{OUT}/items_{feature}.json")
    if p.exists():
        return json.load(open(p))
    allit = json.load(open(f"{OUT}/items_base.json"))
    if Path(f"{OUT}/items_extra.json").exists():
        allit += json.load(open(f"{OUT}/items_extra.json"))
    return [it for it in allit if it["feature"] == feature]


def _label_nj3(model, tok, pool: list[dict], lex: dict) -> list[dict]:  # noqa: ANN001
    """Model-PREDICTED next-word POS (02 s2c definition; ADJ vs NOUN, >=0.3 mass, >=0.6 share)."""
    import random

    import torch

    from feature_spec import N_MINE_PER_VALUE as NM
    from feature_spec import N_SELECT_PER_VALUE as NS
    from items_build import assign_splits

    V = model.get_output_embeddings().weight.shape[0]
    cls = sorted(set(lex.values()))
    M = torch.zeros(V, len(cls), device="cuda")
    for t, c in lex.items():
        M[int(t), cls.index(c)] = 1.0
    ia, ino = cls.index("ADJ"), cls.index("NOUN")
    rng = random.Random(11)
    out = []
    for it in pool:
        with torch.no_grad():
            pr = torch.softmax(model(input_ids=torch.tensor([it["ids"]], device="cuda")).logits[0].float(), -1)
        cm = pr @ M
        tot = cm.sum(1)
        share = cm.max(1).values / tot.clamp_min(1e-9)
        arg = cm.argmax(1)
        ok = (tot >= POS_MIN_MASS) & (share >= POS_ADJ_SHARE)
        ok[:6] = False
        adj = [j for j in range(len(it["ids"])) if ok[j] and arg[j] == ia]
        noun = [j for j in range(len(it["ids"])) if ok[j] and arg[j] == ino]
        # alternate the preferred class so both classes fill (most items contain both)
        first, second = ((adj, "adjective"), (noun, "noun")) if len(out) % 2 == 0 else ((noun, "noun"), (adj, "adjective"))
        if first[0]:
            j, val = rng.choice(first[0]), first[1]
        elif second[0]:
            j, val = rng.choice(second[0]), second[1]
        else:
            continue
        out.append({**it, "id": f"nj3_{it['cluster']}_t{j}", "value": val, "ids": it["ids"][: j + 1],
                    "text": tok.decode(it["ids"][: j + 1]),
                    "meta": {"pos": j, "adj_mass": float(cm[j, ia]), "noun_mass": float(cm[j, ino]),
                             "lex_mass": float(tot[j])}})
    return assign_splits(out, rng, NM, NS)


def _strip_expl(t: str) -> str:
    t = t.strip()
    if t.startswith("<explanation>"):
        t = t[len("<explanation>"):]
    return t.split("</explanation>")[0].strip()


# ============================================================================ gen_ol (H100)
@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100", timeout=4 * 3600, memory=131072, max_inputs=1)
def gen_ol(feature: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats import capture as C
    from mats.readers.jlens import load_jlens
    from mats.readers.olens import WSBENCH_SAMPLING, load_olens

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    ol = load_olens(model, tok)
    if feature == "nj3_predpos":
        pool = json.load(open(f"{OUT}/nj3_pool.json"))
        lex = json.load(open(f"{OUT}/pos_lexicon_02.json"))
        with ol.base():
            items = _label_nj3(ol.model, tok, pool, lex)
        json.dump(items, open(f"{OUT}/items_{feature}.json", "w"))
    else:
        items = _load_items(feature)
    seqs = [it["ids"] for it in items]
    with ol.base():
        acts = C.capture(ol.model, seqs, list(OL_LAYERS), [[-1]] * len(seqs), batch_size=8)
    acts = {L: a[:, 0].contiguous() for L, a in acts.items()}
    torch.save({"ids": [it["id"] for it in items], "acts": acts}, f"{OUT}/acts_{feature}.pt")
    VOLUMES["/results"].commit()
    mine = [i for i, it in enumerate(items) if it["split"] == "mine"]
    gens = {}
    for L in OL_LAYERS:
        g = ol.verbalize(acts[L][mine], L, k=K_READOUTS, sampling=WSBENCH_SAMPLING, batch=32)
        gens[L] = {items[i]["id"]: g[r] for r, i in enumerate(mine)}
    jl = load_jlens(ol.model, tok, layers=[42])
    jtop = jl.readout(acts[42][mine], 42, k=50)
    out = {"feature": feature, "items": items, "ol_gens": gens,
           "jlens_top50_L42": {items[i]["id"]: jtop[r] for r, i in enumerate(mine)},
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/gen_ol_{feature}.json", "w"))
    VOLUMES["/results"].commit()
    return out


# ============================================================================ gen_nla (H200)
@app.function(image=gpu_image, volumes=VOLUMES, gpu="H200", timeout=4 * 3600, memory=131072, max_inputs=1)
def gen_nla(feature: str) -> dict:
    import torch

    from mats.readers import nla as NL

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    items = _load_items(feature)
    blob = torch.load(f"{OUT}/acts_{feature}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items]
    mine = [i for i, it in enumerate(items) if it["split"] == "mine"]
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=False)
    raw = nla.verbalize_raw(blob["acts"][NLA_LAYER][mine], k=K_READOUTS, batch=64)
    out = {"feature": feature,
           "nla_gens": {items[i]["id"]: [_strip_expl(t) for t in raw[r]] for r, i in enumerate(mine)},
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/gen_nla_{feature}.json", "w"))
    VOLUMES["/results"].commit()
    return out


# ============================================================================ scoring
def _cands_for(templates: list[dict], vals: list[str] | None, words: tuple[str, str] | None) -> list[str]:
    """[2T] candidates: per template (fill(a), fill(b)); for typo a/b = the pair's two words."""
    from feature_spec import fill

    out = []
    for t in templates:
        if vals is None:
            out += [fill(t, "*", words[0]), fill(t, "*", words[1])]
        else:
            out += [fill(t, vals[0]), fill(t, vals[1])]
    return out


def _selection(feature: str):  # noqa: ANN202
    import torch

    items = _load_items(feature)
    blob = torch.load(f"{OUT}/acts_{feature}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items]
    sel = [i for i, it in enumerate(items) if it["split"] == "select"]
    return items, blob["acts"], sel


def _typo_pairs(items: list[dict], sel: list[int]) -> list[tuple[int, int]]:
    s = sorted(sel, key=lambda i: items[i]["id"])
    return [(s[2 * m], s[2 * m + 1]) for m in range(len(s) // 2)]


def _score_generic(lp_fn, feature: str, templates: list[dict], items, acts_L, sel) -> dict:  # noqa: ANN001
    """lp_fn(acts [n, d], candidates) -> [n, C]. Returns {template_id: lo list}, y list, ids."""
    import torch

    vals = FEATURES[feature]["values"]
    T = len(templates)
    if vals is not None:
        lp = lp_fn(acts_L[sel], _cands_for(templates, vals, None))
        lo = (lp[:, 0::2] - lp[:, 1::2]).T  # [T, n]
        y = [int(items[i]["value"] == vals[0]) for i in sel]
        ids = [items[i]["id"] for i in sel]
    else:
        rows, y, ids = [], [], []
        for i, j in _typo_pairs(items, sel):
            lp = lp_fn(acts_L[[i, j]], _cands_for(templates, None, (items[i]["value"], items[j]["value"])))
            d = lp[:, 0::2] - lp[:, 1::2]  # [2, T]: LO(w_i vs w_j) on act i, act j
            rows += [d[0], d[1]]
            y += [1, 0]
            ids += [f"{items[i]['id']}|{items[j]['id']}", f"{items[j]['id']}|{items[i]['id']}"]
        lo = torch.stack(rows).T if rows else torch.zeros(T, 0)
    return {"lo": {t["id"]: lo[k].tolist() for k, t in enumerate(templates)}, "y": y, "ids": ids}


@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100", timeout=4 * 3600, memory=131072, max_inputs=1)
def score_ol(feature: str, templates: list[dict], jsets: list[dict]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats.readers.jlens import load_jlens, token_variants
    from mats.readers.olens import load_olens

    t0 = time.time()
    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    ol = load_olens(model, tok)
    items, acts, sel = _selection(feature)
    res = {}
    for L in OL_LAYERS:
        fn = lambda A, cands, L=L: ol.logprob_of_strings(A, L, cands, batch=max(64, 2 * len(cands)))  # noqa: E731
        res[f"ol@{L}"] = _score_generic(fn, feature, templates, items, acts[L], sel)
    # J-Lens: word-set "templates" (max-variant score difference)
    jl = load_jlens(ol.model, tok, layers=list(OL_LAYERS))
    vals = FEATURES[feature]["values"]

    def ids_for(words: list[str], w: str | None = None) -> list[int]:
        ws = [x.replace("{w}", w) for x in words] if w else words
        v = token_variants(tok, ws)
        if not v and w:
            v = [tok.encode(" " + w, add_special_tokens=False)[0]]
        return v

    for L in OL_LAYERS:
        lo, y, idl = {}, None, None
        for js in jsets:
            if vals is not None:
                a, b = ids_for(js["words"][vals[0]]), ids_for(js["words"][vals[1]])
                if not a or not b:
                    continue
                lo[js["id"]] = jl.paired_logodds(acts[L][sel], L, a, b).tolist()
                y = [int(items[i]["value"] == vals[0]) for i in sel]
                idl = [items[i]["id"] for i in sel]
            else:
                vals_lo, y, idl = [], [], []
                for i, j in _typo_pairs(items, sel):
                    a = ids_for(js["words"]["*"], items[i]["value"])
                    b = ids_for(js["words"]["*"], items[j]["value"])
                    d = jl.paired_logodds(acts[L][[i, j]], L, a, b)
                    vals_lo += [float(d[0]), float(d[1])]
                    y += [1, 0]
                    idl += [items[i]["id"], items[j]["id"]]
                lo[js["id"]] = vals_lo
        res[f"jlens@{L}"] = {"lo": lo, "y": y, "ids": idl}
    out = {"feature": feature, "scores": res, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/score_ol_{feature}.json", "w"))
    VOLUMES["/results"].commit()
    return out


@app.function(image=gpu_image, volumes=VOLUMES, gpu="H200", timeout=4 * 3600, memory=131072, max_inputs=1)
def score_nla(feature: str, templates: list[dict]) -> dict:
    from mats.readers import nla as NL

    t0 = time.time()
    VOLUMES["/results"].reload()
    items, acts, sel = _selection(feature)
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=False)
    fn = lambda A, cands: NL.logprob_of_strings_batch(nla, A, cands, batch=max(64, 2 * len(cands)))  # noqa: E731
    out = {"feature": feature,
           "scores": {f"nla@{NLA_LAYER}": _score_generic(fn, feature, templates, items, acts[NLA_LAYER], sel)},
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/score_nla_{feature}.json", "w"))
    VOLUMES["/results"].commit()
    return out


# ============================================================================ entrypoint
@app.local_entrypoint()
def main(stage: str = "gen", features: str = "", readers: str = "ol,nla") -> None:
    feats = [f for f in (features.split(",") if features else FEATURES)]
    RES.mkdir(exist_ok=True)
    if stage == "build":
        info = build.remote()
        (RES / "build_info.json").write_text(json.dumps(info, indent=1, default=str))
    elif stage == "build_extra":
        info = build_extra.remote()
        (RES / "build_extra_info.json").write_text(json.dumps(info, indent=1, default=str))
    elif stage == "gen_ol":
        for r in gen_ol.map(feats, return_exceptions=True):
            _save(r, "gen_ol")
    elif stage == "gen_nla":
        for r in gen_nla.map(feats, return_exceptions=True):
            _save(r, "gen_nla")
    elif stage == "score":
        cand = json.loads((RES / "candidates.json").read_text())
        calls_ol = [(f, cand["ol"][f], cand["jlens"][f]) for f in feats]
        calls_nla = [(f, cand["nla"][f]) for f in feats]
        rd = readers.split(",")
        hs = [("score_ol", score_ol.spawn(*c)) for c in calls_ol] if "ol" in rd else []
        hs += [("score_nla", score_nla.spawn(*c)) for c in calls_nla] if "nla" in rd else []
        for kind, h in hs:
            try:
                _save(h.get(), kind)
            except Exception as e:  # noqa: BLE001
                _save(e, kind)


def _save(r, kind: str) -> None:  # noqa: ANN001
    if isinstance(r, Exception):
        print(f"[{kind}] FAILED: {r!r}")
        return
    (RES / f"{kind}_{r['feature']}.json").write_text(json.dumps(r, default=str))
    print(f"[{kind}] {r['feature']} done in {r.get('wall_s')} s")
