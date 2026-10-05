"""Step 5 (v, reported as a property, not a gate): can the model report feature F about a context
truncated at the probed position? Three formats per feature, chat template, thinking disabled:
  free     greedy generation (<=8 tokens), parsed
  forced   K-way forced choice read from first-token logits over option letters
  numeric  0-9 rating (or digit) read from first-token logits -> expected value / AUROC / Spearman

    modal run s5_selfreport.py [--features col,tense] [--n 300]
"""

import modal

from nonj_common import OUT, VOLUMES, nonj_image

app = modal.App("nonj-s5-selfreport")

POS4 = ["NOUN", "VERB", "ADJ", "ADP"]
POS4_NAMES = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADP": "preposition"}
TOPICS = ["World", "Sports", "Business", "SciTech"]
LETTERS = "ABCDEFGHIJ"


def _contexts(fname: str, items: list[dict], rng, n: int) -> list[dict]:  # noqa: ANN001
    """Sample (text prefix up to and including token j, label) pairs."""
    out = []
    order = rng.permutation(len(items))
    for i in order:
        it = items[i]
        offs, ids = it["offsets"], it["ids"]
        cands = []
        if fname == "col":
            cands = [(j, it["col"][j]) for j in range(4, len(ids)) if not it["has_nl"][j]
                     and 3 <= it["col"][j] and offs[j][1] < len(it["text"]) and it["text"].count("\n", 0, offs[j][1]) >= 1]
        elif fname == "tense":
            cands = [(j, it["tense"][j]) for j in range(4, len(ids)) if it["tense"][j] in ("Past", "Pres")]
        elif fname == "next_pos":
            cands = [(j, it["next_pos"][j]) for j in range(4, len(ids)) if it["next_pos"][j] in POS4]
        elif fname == "depth":
            cands = [(j, it["depth"][j]) for j in range(4, len(ids)) if 1 <= it["depth"][j] <= 7
                     and "\n" not in it["text"][offs[j][0]:offs[j][1]]]
        elif fname == "topic":
            cands = [(len(ids) - 1, it["topic"])]
        if not cands:
            continue
        j, lab = cands[rng.integers(0, len(cands))]
        out.append({"item": int(i), "pos": int(j), "label": lab, "prefix": it["text"][: offs[j][1]]})
        if len(out) >= n:
            break
    if fname == "tense":  # balance classes
        a = [x for x in out if x["label"] == "Past"]
        b = [x for x in out if x["label"] == "Pres"]
        m = min(len(a), len(b))
        out = a[:m] + b[:m]
    return out


QUESTIONS = {
    "col": {
        "free": "How many characters are on the last line of the text (the final line, which is cut off), counting letters, spaces and punctuation? Answer with just the number.",
        "forced": "How many characters are on the last line of the text (the final line, which is cut off)? Options: A) 0-19 B) 20-39 C) 40-59 D) 60-79 E) 80-99. Answer with the letter only.",
        "numeric": "Roughly how many tens of characters are on the last line of the text (the final, cut-off line)? Answer with a single digit (e.g. 4 means about forty characters).",
    },
    "tense": {
        "free": "Is the clause at the very end of the text (the one being written when the text stops) in the past tense or the present tense? Answer with one word.",
        "forced": "Is the clause at the very end of the text in the past tense or the present tense? Options: A) past B) present. Answer with the letter only.",
        "numeric": "On a scale from 0 (clearly present tense) to 9 (clearly past tense), rate the tense of the clause at the very end of the text. Answer with a single digit.",
    },
    "next_pos": {
        "free": "What part of speech will the next word of the text be? Answer with one word (for example: noun, verb, adjective, adverb, preposition).",
        "forced": "What part of speech will the next word of the text be? Options: A) noun B) verb C) adjective D) preposition. Answer with the letter only.",
        "numeric": "On a scale from 0 (certainly not) to 9 (certainly), how likely is it that the next word of the text is a noun? Answer with a single digit.",
    },
    "depth": {
        "free": "At the very end of the text, how many square brackets are currently open (opened but not yet closed)? Answer with just the number.",
        "forced": "At the very end of the text, how many square brackets are currently open? Options: A) 1 B) 2 C) 3 D) 4 E) 5 F) 6 G) 7. Answer with the letter only.",
        "numeric": "At the very end of the text, how many square brackets are currently open? Answer with a single digit.",
    },
    "topic": {
        "free": "What is the topic of the text: World, Sports, Business, or Science/Technology? Answer with one word.",
        "forced": "What is the topic of the text? Options: A) World B) Sports C) Business D) Science/Technology. Answer with the letter only.",
        "numeric": "On a scale from 0 (not at all) to 9 (entirely), how much is the text about sports? Answer with a single digit.",
    },
}


def _forced_truth(fname: str, lab) -> int:  # noqa: ANN001
    if fname == "col":
        return min(int(lab) // 20, 4)
    if fname == "tense":
        return 0 if lab == "Past" else 1
    if fname == "next_pos":
        return POS4.index(lab)
    if fname == "depth":
        return int(lab) - 1
    if fname == "topic":
        return TOPICS.index(lab)
    raise ValueError(fname)


def _n_opts(fname: str) -> int:
    return {"col": 5, "tense": 2, "next_pos": 4, "depth": 7, "topic": 4}[fname]


def _parse_free(fname: str, s: str):  # noqa: ANN202
    import re

    t = s.strip().lower()
    if fname in ("col", "depth"):
        m = re.search(r"\d+", t)
        return int(m.group()) if m else None
    if fname == "tense":
        return "Past" if "past" in t else ("Pres" if "present" in t else None)
    if fname == "next_pos":
        for k, v in POS4_NAMES.items():
            if v in t:
                return k
        for w in ("adverb", "determiner", "pronoun", "article", "conjunction"):
            if w in t:
                return w
        return None
    if fname == "topic":
        for k, kw in (("World", "world"), ("Sports", "sport"), ("Business", "business"), ("SciTech", "sci"), ("SciTech", "tech")):
            if kw in t:
                return k
        return None
    return None


@app.function(image=nonj_image, volumes=VOLUMES, gpu="H100", timeout=4 * 3600, memory=131072)
def selfreport(features: list[str], n: int = 300) -> dict:
    import pickle

    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from nonj_lib import load_model, save_json

    VOLUMES["/results"].reload()
    tok, model = load_model()
    tok.padding_side = "left"
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    res_all = {}
    for fname in features:
        dataset = {"col": "wrap", "next_pos": "prose", "tense": "prose", "depth": "brackets", "topic": "topic"}[fname]
        items = pickle.load(open(f"{OUT}/data/{dataset}.pkl", "rb"))
        ctxs = _contexts(fname, items, np.random.default_rng(21), n)

        def prompt(c: dict, fmt: str) -> str:
            msg = f"Here is a piece of text:\n\n<text>\n{c['prefix']}\n</text>\n\n{QUESTIONS[fname][fmt]}"
            return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)

        rows = []
        bs = 8
        for b in range(0, len(ctxs), bs):
            chunk = ctxs[b:b + bs]
            for fmt in ("free", "forced", "numeric"):
                enc = tok([prompt(c, fmt) for c in chunk], return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    if fmt == "free":
                        gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
                        outs = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
                        for c, o in zip(chunk, outs):
                            c["free_raw"], c["free"] = o, _parse_free(fname, o)
                    else:
                        lg = model(**enc).logits[:, -1].float()
                        lp = torch.log_softmax(lg, -1)
                        for k, c in enumerate(chunk):
                            if fmt == "forced":
                                sc = lp[k, letter_ids[: _n_opts(fname)]]
                                c["forced"] = int(sc.argmax())
                                c["forced_mass"] = float(sc.exp().sum())
                            else:
                                pd = lp[k, digit_ids].exp()
                                c["numeric_mass"] = float(pd.sum())
                                pd = pd / pd.sum()
                                c["numeric_ev"] = float((pd * torch.arange(10, device="cuda")).sum())
                                c["numeric_argmax"] = int(pd.argmax())
            rows.extend(chunk)
        # ---- scoring
        r: dict = {"n": len(rows)}
        truth_f = np.array([_forced_truth(fname, c["label"]) for c in rows])
        forced = np.array([c["forced"] for c in rows])
        r["forced_acc"] = float((forced == truth_f).mean())
        r["forced_chance_uniform"] = 1.0 / _n_opts(fname)
        r["forced_chance_majority"] = float(np.bincount(truth_f).max() / len(truth_f))
        r["forced_mass_mean"] = float(np.mean([c["forced_mass"] for c in rows]))
        r["numeric_mass_mean"] = float(np.mean([c["numeric_mass"] for c in rows]))
        rng = np.random.default_rng(0)

        def boot(fn) -> list[float]:  # noqa: ANN001
            v = [fn(rng.integers(0, len(rows), len(rows))) for _ in range(1000)]
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

        r["forced_acc_ci95"] = boot(lambda ix: (forced[ix] == truth_f[ix]).mean())
        free = [c["free"] for c in rows]
        if fname in ("col", "depth"):
            y = np.array([float(c["label"]) for c in rows])
            fv = np.array([np.nan if f is None else float(f) for f in free])
            ok = ~np.isnan(fv)
            r["free_parse_rate"] = float(ok.mean())
            r["free_spearman"] = float(spearmanr(fv[ok], y[ok]).statistic) if ok.sum() > 5 else None
            r["free_exact"] = float((fv[ok] == y[ok]).mean()) if ok.any() else None
            r["free_within2"] = float((np.abs(fv[ok] - y[ok]) <= 2).mean()) if ok.any() else None
            r["free_mae"] = float(np.abs(fv[ok] - y[ok]).mean()) if ok.any() else None
            r["mae_predict_mean_baseline"] = float(np.abs(y - y.mean()).mean())
            r["free_spearman_ci95"] = boot(lambda ix: spearmanr(fv[ix][ok[ix]], y[ix][ok[ix]]).statistic) if ok.sum() > 5 else None
            ev = np.array([c["numeric_ev"] for c in rows])
            tgt = y // 10 if fname == "col" else y
            r["numeric_spearman"] = float(spearmanr(ev, tgt).statistic)
            r["numeric_spearman_ci95"] = boot(lambda ix: spearmanr(ev[ix], tgt[ix]).statistic)
            r["numeric_argmax_acc"] = float((np.array([c["numeric_argmax"] for c in rows]) == tgt).mean())
        else:
            truth = [c["label"] for c in rows]
            fr = np.array([f == t for f, t in zip(free, truth)])
            r["free_acc"] = float(fr.mean())
            r["free_parse_rate"] = float(np.mean([f is not None for f in free]))
            r["free_acc_ci95"] = boot(lambda ix: fr[ix].mean())
            pos_lab = {"tense": "Past", "next_pos": "NOUN", "topic": "Sports"}[fname]
            yb = np.array([t == pos_lab for t in truth])
            ev = np.array([c["numeric_ev"] for c in rows])
            r["numeric_auroc"] = float(roc_auc_score(yb, ev)) if 0 < yb.sum() < len(yb) else None
            r["numeric_auroc_ci95"] = boot(lambda ix: roc_auc_score(yb[ix], ev[ix]) if 0 < yb[ix].sum() < len(ix) else 0.5)
        r["examples"] = [{k: c[k] for k in ("label", "free_raw", "forced", "numeric_ev")} | {"prefix_tail": c["prefix"][-120:]}
                         for c in rows[:12]]
        res_all[fname] = r
        print(fname, {k: v for k, v in r.items() if k != "examples"}, flush=True)
        save_json(r, f"{OUT}/selfreport/{fname}.json")
        with open(f"{OUT}/selfreport/{fname}_rows.pkl", "wb") as f:
            pickle.dump(rows, f)
        VOLUMES["/results"].commit()
    return res_all


@app.local_entrypoint()
def main(features: str = "col,next_pos,tense,depth,topic", n: int = 300) -> None:
    fl = features.split(",")
    groups = [fl[i::2] for i in range(2) if fl[i::2]]
    for r in selfreport.starmap([(g, n) for g in groups], return_exceptions=True):
        print({k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in r.items()} if isinstance(r, dict) else r)
