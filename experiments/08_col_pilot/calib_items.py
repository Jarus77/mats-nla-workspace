"""Calibration item generators (Modal build container with network + spaCy). DECLARED PILOT SPLIT.

lang : papluca/language-identification TEST split (02 used the train split + multilingual_mt;
       07 used test fr/de only), Russian and Polish, len > 60 chars, shuffled with CALIB_SEED;
       30 + 30 items; read token = random index in [8, min(len, 64)); ids truncated there.
tense: fresh wikitext paragraphs (02 cleaner, pool seed CALIB_SEED) excluding 02 wrap/prose, 07
       items_base.json and the 08 col pilot passages; 02's build_prose (spaCy clause tense,
       label only after the clause's finite verb); one read token per passage in [8, 120);
       30 Past + 30 Pres passages.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random

import pilot_items as PI

CALIB_SEED = 9009
N_PER_CLASS = 30


def _sha(t: str) -> str:
    return hashlib.sha1(PI._norm(t).encode()).hexdigest()[:16]


def build_lang(tok) -> list[dict]:  # noqa: ANN001
    from datasets import load_dataset

    ds = load_dataset("papluca/language-identification", split="test")
    rng = random.Random(CALIB_SEED)
    out = []
    for code, name in (("ru", "Russian"), ("pl", "Polish")):
        texts = [ex["text"] for ex in ds if ex["labels"] == code and len(ex["text"]) > 60]
        rng.shuffle(texts)
        k = 0
        for t in texts:
            ids = tok(t, add_special_tokens=False)["input_ids"][:64]
            if len(ids) < 12:
                continue
            j = rng.randrange(8, len(ids))
            out.append({"id": f"calib_lang_{code}_{k:02d}", "feature": "lang", "label": name,
                        "y": 1 if code == "ru" else 0, "ids": ids[: j + 1], "text": tok.decode(ids[: j + 1]),
                        "pos": j, "source_sha1_16": _sha(t), "source_head": PI._norm(t)[:80],
                        "source": "papluca/test"})
            k += 1
            if k == N_PER_CLASS:
                break
    return out


def build_tense(tok, out02: str, out07: str, col_pilot_items: list[dict]) -> list[dict]:  # noqa: ANN001
    import spacy

    s1 = PI.load_02("s1_data", out02)
    nlp = spacy.load("en_core_web_lg")
    pool = s1._wikitext_paragraphs(500, 60000, CALIB_SEED)
    used = PI._used_prefixes(out02, out07)
    by_head: dict[str, list[str]] = {}
    for u in used:
        by_head.setdefault(u[:30], []).append(u)
    col_sha = {it["passage_sha1_16"] for it in col_pilot_items}
    fresh = [p for p in pool[:6000] if not PI._overlaps(p, by_head) and _sha(p) not in col_sha]
    rng = random.Random(CALIB_SEED)
    orig = s1._wikitext_paragraphs
    s1._wikitext_paragraphs = lambda *_a, **_k: fresh[:1500]
    prose = s1.build_prose(tok, nlp, n=1500)
    s1._wikitext_paragraphs = orig
    out, n = [], {"Past": 0, "Pres": 0}
    for k, it in enumerate(prose):
        cand = [(j, t) for j, t in enumerate(it["tense"]) if t in ("Past", "Pres") and 8 <= j < 120]
        need = [c for c in cand if n[c[1]] < N_PER_CLASS]
        if not need:
            continue
        pres = [c for c in need if c[1] == "Pres"]
        j, t = rng.choice(pres if pres else need)
        n[t] += 1
        out.append({"id": f"calib_tense_{len(out):02d}", "feature": "tense", "label": t,
                    "y": 1 if t == "Past" else 0, "ids": it["ids"][: j + 1],
                    "text": it["text"][: it["offsets"][j][1]], "pos": j,
                    "source_sha1_16": _sha(fresh[k]), "source_head": PI._norm(fresh[k])[:80],
                    "source": "wikitext103/train"})
        if all(v == N_PER_CLASS for v in n.values()):
            break
    return out
