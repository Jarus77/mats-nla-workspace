"""Feature definitions: which rows of a captured dataset carry which label (pure numpy, no torch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POS_CLASSES = ["NOUN", "ADP", "DET", "VERB", "ADJ", "PROPN", "PRON", "ADV"]
TOPICS = ["World", "Sports", "Business", "SciTech"]
LANG_NAMES = ["Arabic", "Greek", "Polish", "Russian", "Turkish", "Vietnamese"]


@dataclass(frozen=True)
class Feature:
    name: str
    dataset: str
    kind: str  # "cls" | "reg"
    in_workspace_control: bool
    description: str
    steer_pair: tuple | None = None  # (pos_class, neg_class) for cls steering direction


FEATURES = [
    Feature("col", "wrap", "reg", False, "characters since last newline (after current token) in hard-wrapped prose"),
    Feature("next_pos", "prose", "cls", False, "UPOS of the next word (8 classes)", ("ADJ", "NOUN")),
    Feature("tense", "prose", "cls", False, "tense of current clause (Past vs Pres), positions after the finite verb", ("Past", "Pres")),
    Feature("depth", "brackets", "reg", False, "open-bracket nesting depth in nested Python lists"),
    Feature("pred_pos", "prose", "cls", False, "model-PREDICTED UPOS of the next word (argmax POS-lexicon mass of the next-token distribution; >=0.3 lexicon mass, >=0.6 share)", ("ADJ", "NOUN")),
    Feature("col_report", "wrap_report", "reg", True, "IJ1: characters since last newline under an EXPLICIT 'how many characters in the first line' chat question"),
    Feature("lang", "lang", "cls", True, "IJ2: language identity (ar, el, pl, ru, tr, vi; papluca + WorkspaceBench multilingual_mt prompts)", ("Russian", "Polish")),
    Feature("topic", "topic", "cls", True, "AG News article topic (4 classes)", ("Sports", "Business")),
    Feature("nexttok", "prose", "cls", True, "identity of the next token (10 most frequent word tokens)", None),
]
FEATURES_BY_NAME = {f.name: f for f in FEATURES}


def rows_for(feat: Feature, items: list[dict], index: dict, rng: np.random.Generator,
             cap_per_class: int = 4000, cap_reg: int = 40000) -> dict:
    """Return dict(rows, y, groups, types, classes) for the feature; rows index the capture arrays."""
    item, pos, tid, ntid = index["item"], index["pos"], index["tid"], index["ntid"]
    keep = pos >= 4  # skip attention-sink-ish first tokens
    y_raw: list = [None] * len(item)
    if feat.name in ("col", "col_report"):
        for r in np.nonzero(keep)[0]:
            it = items[item[r]]
            c = it["col"][pos[r]]
            if not it["has_nl"][pos[r]] and c is not None and c >= 1:
                y_raw[r] = float(c)
    elif feat.name == "lang":
        for r in np.nonzero(keep)[0]:
            y_raw[r] = items[item[r]]["lang"]
    elif feat.name == "pred_pos":
        cm = index["class_mass"]
        tot = cm.sum(1)
        share = cm.max(1) / np.maximum(tot, 1e-9)
        arg = cm.argmax(1)
        for r in np.nonzero(keep & (tot >= 0.3) & (share >= 0.6))[0]:
            y_raw[r] = POS_CLASSES[arg[r]]
    elif feat.name == "depth":
        for r in np.nonzero(keep)[0]:
            y_raw[r] = float(items[item[r]]["depth"][pos[r]])
    elif feat.name in ("next_pos", "tense"):
        key = feat.name
        for r in np.nonzero(keep)[0]:
            v = items[item[r]][key][pos[r]]
            if v is not None and (key != "tense" or v in ("Past", "Pres")) and (key != "next_pos" or v in POS_CLASSES):
                y_raw[r] = v
    elif feat.name == "topic":
        for r in np.nonzero(pos >= 8)[0]:
            y_raw[r] = items[item[r]]["topic"]
    elif feat.name == "nexttok":
        top = index["nexttok_classes"]
        lut = {t: i for i, t in enumerate(top)}
        for r in np.nonzero(keep & (ntid >= 0))[0]:
            if int(ntid[r]) in lut:
                y_raw[r] = int(ntid[r])
    rows = np.array([r for r in range(len(item)) if y_raw[r] is not None])
    if feat.kind == "reg":
        if len(rows) > cap_reg:
            rows = np.sort(rng.choice(rows, cap_reg, replace=False))
        y = np.array([y_raw[r] for r in rows], dtype=np.float32)
        classes = None
    else:
        labs = np.array([y_raw[r] for r in rows], dtype=object)
        classes = sorted(set(labs.tolist()), key=str)
        sel = []
        for c in classes:
            rc = rows[labs == c]
            sel.append(rc if len(rc) <= cap_per_class else rng.choice(rc, cap_per_class, replace=False))
        rows = np.sort(np.concatenate(sel))
        cmap = {c: i for i, c in enumerate(classes)}
        y = np.array([cmap[y_raw[r]] for r in rows], dtype=np.int64)
    return {"rows": rows, "y": y, "groups": item[rows], "types": tid[rows], "classes": classes}


def nexttok_classes(tok, ntid: np.ndarray, k: int = 10) -> list[int]:  # noqa: ANN001
    """k most frequent next tokens that are alphabetic words with a leading space."""
    vals, counts = np.unique(ntid[ntid >= 0], return_counts=True)
    out = []
    for v in vals[np.argsort(-counts)]:
        s = tok.decode([int(v)])
        if s.startswith(" ") and s[1:].isalpha():
            out.append(int(v))
        if len(out) == k:
            break
    return out
