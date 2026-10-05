"""Vocabulary token sets per feature, "in the form the lens emits" (Fable round 4).

``emitted_sets(fname, vocab, tok, gender_lex)`` -> (class-0 ids, class-1 ids); class-0 may be empty.
Used (a) by the functional gate: does the J-Lens readout put more mass / more top-10 slots on the
class-1 token category when the feature is 1? and (b) by the G4 behaviour metrics (bf_causal).
"""

from __future__ import annotations

from typing import Any

from bf_feats import JS_MARKERS, PY_MARKERS


def single_ids(tok: Any, words: list[str]) -> list[int]:
    out = []
    for w in words:
        e = tok.encode(w, add_special_tokens=False)
        if len(e) == 1:
            out.append(e[0])
    return sorted(set(out))


def gender_sets(gender_lex: dict, min_count: int = 3, purity: float = 0.9) -> tuple[list[int], list[int]]:
    fem, masc = [], []
    for t, (nf, nm) in gender_lex.items():
        n = nf + nm
        if n >= min_count:
            if nf / n >= purity:
                fem.append(int(t))
            elif nm / n >= purity:
                masc.append(int(t))
    return masc, fem


def emitted_sets(fname: str, vocab: list[str], tok: Any, gender_lex: dict | None = None) -> tuple[list[int], list[int]]:
    def where(pred) -> list[int]:  # noqa: ANN001
        return [i for i, s in enumerate(vocab) if pred(s)]

    if fname in ("cap_next", "cap_next_all"):
        return (where(lambda s: len(s) > 1 and s[0] == " " and s[1].isalpha() and s[1].islower()),
                where(lambda s: len(s) > 1 and s[0] == " " and s[1].isalpha() and s[1].isupper()))
    if fname == "code_lang":
        return single_ids(tok, JS_MARKERS), single_ids(tok, PY_MARKERS)
    if fname == "in_quote":
        return (where(lambda s: s.startswith(' "')),
                where(lambda s: '"' in s and not s.startswith(' "')))
    if fname == "gender_next":
        return gender_sets(gender_lex or {})
    if fname == "indent":
        return (where(lambda s: 0 < len(s) <= 4 and set(s) == {" "}),
                where(lambda s: len(s) >= 5 and set(s) == {" "}))
    if fname == "sent_pos":
        return [], where(lambda s: len(s) > 0 and s[0] in ".!?" and not s.startswith(".."))
    if fname == "list_hdr":
        return single_ids(tok, ["2", "3"]), single_ids(tok, ["5", "6", "7"])
    if fname == "list_num":
        return single_ids(tok, ["2", "3", "4"]), single_ids(tok, ["5", "6", "7", "8"])
    raise ValueError(fname)
