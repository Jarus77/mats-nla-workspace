"""Step 1: build token-labelled datasets (CPU). Output: /results/02_nonj_features/data/<name>.pkl

Datasets (all raw text, no chat template, Qwen tokenizer, no BOS):
  wrap     wikitext paragraphs hard-wrapped at width W~U{30..90}: col (chars since last \\n after
           the token), width W, next_is_nl
  prose    natural wikitext paragraphs (<=160 tok): next-word UPOS, current-word UPOS, clause tense
           (Past/Pres, only after the clause's finite verb), dependency depth, next token id
  brackets synthetic nested Python lists: open-bracket depth after each token
  topic    AG News (World/Sports/Business/SciTech): article topic at every position >= 8

    modal run s1_data.py
"""

import modal

from nonj_common import OUT, VOLUMES, nonj_image

app = modal.App("nonj-s1-data")

SEED = 0


def _wikitext_paragraphs(min_chars: int, n: int, seed: int) -> list[str]:
    import random
    import re

    import pandas as pd

    df = pd.read_parquet(f"{OUT}/data/raw_wikitext103_train.parquet")
    paras = []
    for t in df["text"].tolist():
        t = t.strip()
        if len(t) < min_chars or t.startswith("="):
            continue
        # undo wikitext tokenisation artefacts
        t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        t = re.sub(r" ([.,;:!?)\]'])", r"\1", t)
        t = re.sub(r"([(\[]) ", r"\1", t)
        t = t.replace(" n't", "n't").replace(" 's", "'s").replace('" ', '"', 1)
        if "<unk>" in t:
            continue
        paras.append(t)
    rng = random.Random(seed)
    rng.shuffle(paras)
    return paras[:n]


def _tok(tok, text: str, max_tokens: int):  # noqa: ANN001, ANN202
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"][:max_tokens], enc["offset_mapping"][:max_tokens]
    return ids, offs


def build_wrap(tok, n: int = 900) -> list[dict]:  # noqa: ANN001
    import random
    import textwrap

    rng = random.Random(SEED + 1)
    items = []
    for i, p in enumerate(_wikitext_paragraphs(900, n, SEED + 1)):
        W = rng.randint(30, 90)
        text = textwrap.fill(p, width=W, break_long_words=False)[:1100]
        text = text[: text.rfind("\n")] if "\n" in text[-W:] else text  # end mid-line is fine
        ids, offs = _tok(tok, text, 320)
        cols, nl_in, next_nl = [], [], []
        for j, (s, e) in enumerate(offs):
            seg = text[:e]
            last = seg.rfind("\n")
            cols.append(e - (last + 1))
            nl_in.append("\n" in text[s:e])
            nxt = offs[j + 1] if j + 1 < len(offs) else None
            next_nl.append(bool(nxt and "\n" in text[nxt[0] : nxt[1]]))
        # true line lengths for each line (for chars_remaining)
        items.append({"id": f"wrap{i}", "text": text, "ids": ids, "offsets": offs, "width": W,
                      "col": cols, "has_nl": nl_in, "next_nl": next_nl})
    return items


def _clause_tense(t) -> tuple[str | None, int | None]:  # noqa: ANN001
    """Tense of the clause containing spaCy token t and char start of its finite verb."""
    clause_deps = {"ROOT", "ccomp", "advcl", "relcl", "conj", "parataxis", "csubj", "acl", "xcomp"}
    h = t
    for _ in range(40):
        if h.pos_ in ("VERB", "AUX") and (h.dep_ in clause_deps or h.head == h):
            break
        if h.head == h:
            break
        h = h.head
    if h.pos_ not in ("VERB", "AUX"):
        return None, None
    cands = [c for c in h.children if c.dep_ in ("aux", "auxpass") ] + [h]
    for c in cands:
        if "Fin" in c.morph.get("VerbForm"):
            if c.lemma_.lower() in ("will", "shall", "'ll"):
                return "Fut", c.idx
            tense = c.morph.get("Tense")
            if tense:
                return tense[0], c.idx
    return None, None


def build_prose(tok, nlp, n: int = 1600) -> list[dict]:  # noqa: ANN001
    import bisect

    items = []
    paras = _wikitext_paragraphs(500, n, SEED + 2)
    for i, (p, doc) in enumerate(zip(paras, nlp.pipe(paras, batch_size=64))):
        ids, offs = _tok(tok, p, 160)
        starts = [t.idx for t in doc]
        sp = list(doc)

        def sp_at(ch: int) -> int:
            return max(0, bisect.bisect_right(starts, ch) - 1)

        tense_cache = {}
        next_pos, cur_pos, tense, depth = [], [], [], []
        for j, (s, e) in enumerate(offs):
            k = sp_at(max(s, e - 1))
            t = sp[k]
            word_end = t.idx + len(t.text)
            cur_pos.append(t.pos_)
            # next-word POS only if this token ends the spaCy word and the next qwen token starts the next word
            nxt = offs[j + 1] if j + 1 < len(offs) else None
            if e == word_end and k + 1 < len(sp) and nxt is not None and sp_at(nxt[1] - 1) == k + 1 \
                    and p[nxt[0]:nxt[1]].strip() and sp[k + 1].idx >= nxt[0]:
                next_pos.append(sp[k + 1].pos_)
            else:
                next_pos.append(None)
            if k not in tense_cache:
                tense_cache[k] = _clause_tense(t)
            tn, fstart = tense_cache[k]
            tense.append(tn if (tn is not None and fstart is not None and s >= fstart) else None)
            d, h = 0, t
            while h.head != h and d < 50:
                h, d = h.head, d + 1
            depth.append(d)
        items.append({"id": f"prose{i}", "text": p, "ids": ids, "offsets": offs, "next_pos": next_pos,
                      "cur_pos": cur_pos, "tense": tense, "dep_depth": depth})
    return items


def _rand_list(rng, depth: int, maxd: int) -> str:  # noqa: ANN001
    n = rng.randint(1, 4)
    parts = []
    for _ in range(n):
        if depth < maxd and rng.random() < 0.45:
            parts.append(_rand_list(rng, depth + 1, maxd))
        else:
            parts.append(str(rng.randint(0, 99)) if rng.random() < 0.7 else repr(rng.choice(
                ["a", "b", "x", "y", "foo", "bar", "id", "ok"])))
    return "[" + ", ".join(parts) + "]"


def build_brackets(tok, n_docs: int = 320) -> list[dict]:  # noqa: ANN001
    import random

    rng = random.Random(SEED + 3)
    items = []
    for i in range(n_docs):
        lines = []
        for li in range(8):
            s = _rand_list(rng, 1, 7)
            while s.count("[") < 4:
                s = _rand_list(rng, 1, 7)
            lines.append(f"v{li} = {s}")
        text = "\n".join(lines) + "\n"
        ids, offs = _tok(tok, text, 512)
        depth = []
        for s, e in offs:
            seg = text[:e]
            depth.append(seg.count("[") - seg.count("]"))
        items.append({"id": f"br{i}", "text": text, "ids": ids, "offsets": offs, "depth": depth})
    return items


def build_topic(tok, per_class: int = 500) -> list[dict]:  # noqa: ANN001
    import pandas as pd

    df = pd.read_parquet(f"{OUT}/data/raw_ag_news_train.parquet").sample(frac=1.0, random_state=SEED)
    names = ["World", "Sports", "Business", "SciTech"]
    items = []
    for lab in range(4):
        sub = df[df["label"] == lab].head(per_class)
        for k, t in enumerate(sub["text"].tolist()):
            t = t.replace("\\", " ").strip()
            ids, offs = _tok(tok, t, 96)
            items.append({"id": f"topic{lab}_{k}", "text": t, "ids": ids, "offsets": offs, "topic": names[lab]})
    return items


@app.function(image=nonj_image, volumes=VOLUMES, timeout=3 * 3600, memory=32768, cpu=8)
def build() -> dict:
    import collections
    import pickle

    import spacy
    from transformers import AutoTokenizer

    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    nlp = spacy.load("en_core_web_lg")
    out = {}
    for name, fn in (("wrap", lambda: build_wrap(tok)), ("brackets", lambda: build_brackets(tok)),
                     ("topic", lambda: build_topic(tok)), ("prose", lambda: build_prose(tok, nlp))):
        items = fn()
        with open(f"{OUT}/data/{name}.pkl", "wb") as f:
            pickle.dump(items, f)
        ntok = sum(len(it["ids"]) for it in items)
        summ: dict = {"n_items": len(items), "n_tokens": ntok}
        if name == "prose":
            for key in ("next_pos", "cur_pos", "tense"):
                summ[key] = collections.Counter(x for it in items for x in it[key] if x).most_common(14)
        if name == "brackets":
            summ["depth"] = sorted(collections.Counter(x for it in items for x in it["depth"]).items())
        if name == "wrap":
            summ["col_max"] = max(max(it["col"]) for it in items)
        out[name] = summ
        print(name, summ, flush=True)
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main() -> None:
    print(build.remote())
