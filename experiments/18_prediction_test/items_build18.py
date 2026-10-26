"""Pure text generators for the exp 18 rows (no model, no tokenizer; unit-tested locally). Each returns
items WITHOUT token ids: {id, feature, cluster, value, text, meta}. Synthetic rows take an rng and a count
(values alternate, so counts are balanced); the stance row takes cleaned paragraphs. The read position is
the last character of ``text``; ``text`` never ends in whitespace."""
from __future__ import annotations

import random
import re

from feature_spec18 import FEATURES

SEED = 1818
_WORDS = ["river", "market", "engine", "garden", "signal", "harbour", "lantern", "council", "orbit", "ledger",
          "meadow", "circuit", "canvas", "glacier", "saddle", "beacon", "furnace", "pillar", "harvest", "compass"]
_NAMES = ["alice", "bruno", "chen", "dana", "eli", "farah", "gwen", "hugo", "iris", "jonas"]
_KEYS = ["name", "city", "role", "team", "status", "owner", "region", "level", "label", "group", "kind", "note"]
_STRVALS = ["active", "pending", "retired", "north", "central", "lead", "staff", "writer", "auditor", "Lyon",
            "Prague", "Seville", "Oslo", "Turin"]


def _value(rng: random.Random, feature: str, k: int) -> str:
    v = FEATURES[feature]["values"]
    return v[k % 2]


def _words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n))


# ------------------------------------------------------------------ json_depth
def build_json_depth(rng: random.Random, n: int) -> list[dict]:
    """Read at the closing quote of a string value that is NOT the last key of its object, at depth 3 vs 1.
    Shallow items carry extra sibling keys so prefix lengths overlap across values."""
    out = []
    for k in range(n):
        val = _value(rng, "json_depth", k)
        depth = 3 if val == "deep" else 1
        keys = rng.sample(_KEYS, 9)
        pre = ", ".join(f'"{kk}": "{rng.choice(_STRVALS)}"' for kk in keys[:3])  # 3 completed top-level pairs
        if depth == 3:
            text = ('{' + pre + f', "{keys[3]}": {{"{keys[4]}": "{rng.choice(_STRVALS)}", "{keys[5]}": '
                    f'{{"{keys[6]}": "{rng.choice(_STRVALS)}"')
        else:
            extra = ", ".join(f'"{kk}": "{rng.choice(_STRVALS)}"' for kk in keys[3:5])
            text = '{' + pre + ", " + extra + f', "{keys[6]}": "{rng.choice(_STRVALS)}"'
        out.append({"id": f"json_{k}_d{depth}", "feature": "json_depth", "cluster": f"json{k}", "value": val,
                    "text": text, "meta": {"depth": depth}})
    return out


# ------------------------------------------------------------------ latex_math
def build_latex_math(rng: random.Random, n: int) -> list[dict]:
    out = []
    for k in range(n):
        val = _value(rng, "latex_math", k)
        lead = f"We consider the {_words(rng, 2)} and write the {rng.choice(_WORDS)} term as"
        a, b, c = rng.sample("xyzuvwabc", 3)
        expr = f"${a}_{{{rng.randint(1, 9)}}} + \\frac{{{b}}}{{{c}}} \\cdot \\sqrt{{{a} - {b}}}"
        if val == "inside":
            text = f"In the {_words(rng, 2)} setting, {lead[0].lower() + lead[1:]} {expr}"   # math still open: odd $ count; lead padded to match length
        else:
            text = f"{lead} {expr}$ for every {_words(rng, 3)} in the {rng.choice(_WORDS)}"  # 4+ words after $
        out.append({"id": f"late_{k}_{val}", "feature": "latex_math", "cluster": f"latex{k}", "value": val,
                    "text": text, "meta": {}})
    return out


# ------------------------------------------------------------------ py_string
def build_py_string(rng: random.Random, n: int) -> list[dict]:
    out = []
    for k in range(n):
        val = _value(rng, "py_string", k)
        q = rng.choice(['"', "'"])
        var = rng.choice(_NAMES)
        body = f"{q}{_words(rng, 4)}"
        other = rng.choice([n for n in _NAMES if n != var])
        if val == "inside":
            text = f"{other} = {q}{_words(rng, 2)}{q}\n{var} = {body}"
        else:
            text = f"{var} = {body}{q}\nresult = process({var}, {rng.choice(_WORDS)}"
        out.append({"id": f"pyst_{k}_{val}", "feature": "py_string", "cluster": f"pystr{k}", "value": val,
                    "text": text, "meta": {"quote": q}})
    return out


# ------------------------------------------------------------------ call_arg
def build_call_arg(rng: random.Random, n: int) -> list[dict]:
    out = []
    for k in range(n):
        val = _value(rng, "call_arg", k)
        fn = rng.choice(["render", "merge", "update", "compute", "dispatch", "resolve"])
        args = rng.sample(_NAMES, 3)
        setup = "\n".join(f"{a} = load({rng.choice(_WORDS)!r})" for a in args)
        if val == "first":
            setup += f"\n{rng.choice(_NAMES)} = load({rng.choice(_WORDS)!r})"   # one more setup line: matches the longer call
            text = f"{setup}\nout = {fn}({args[0]}"
        else:
            text = f"{setup}\nout = {fn}({args[0]}, {args[1]}, {args[2]}"
        out.append({"id": f"call_{k}_{val}", "feature": "call_arg", "cluster": f"call{k}", "value": val,
                    "text": text, "meta": {"n_args": 1 if val == "first" else 3}})
    return out


# ------------------------------------------------------------------ md_heading
def build_md_heading(rng: random.Random, n: int) -> list[dict]:
    out = []
    for k in range(n):
        val = _value(rng, "md_heading", k)
        marks = "#" if val == "h1" else "###"
        intro = f"# {_words(rng, 3).title()}\n\n{_words(rng, 12)}.\n\n## {_words(rng, 2).title()}\n\n{_words(rng, 10)}.\n\n"
        if val == "h1":
            intro = f"{_words(rng, 14)}.\n\n{_words(rng, 12)}.\n\n"                 # same length, no earlier headings
        words = rng.randint(4, 8)
        text = intro + f"{marks} {_words(rng, words).title()}"
        out.append({"id": f"mdhd_{k}_{val}", "feature": "md_heading", "cluster": f"mdh{k}", "value": val,
                    "text": text, "meta": {"level": 1 if val == "h1" else 3}})
    return out


# ------------------------------------------------------------------ csv_col
def build_csv_col(rng: random.Random, n: int) -> list[dict]:
    out = []
    for k in range(n):
        val = _value(rng, "csv_col", k)
        heads = rng.sample(["name", "city", "role", "team", "status", "year", "owner"], 5)
        rows = [",".join(rng.choice(_STRVALS) for _ in heads) for _ in range(rng.randint(3, 5))]
        cut = 1 if val == "first" else 4
        last = ",".join([rng.choice(_STRVALS) for _ in range(cut)])
        text = "\n".join([",".join(heads)] + rows + [last])
        out.append({"id": f"csvc_{k}_{val}", "feature": "csv_col", "cluster": f"csv{k}", "value": val,
                    "text": text, "meta": {"col": cut}})
    return out


# ------------------------------------------------------------------ html_depth
def build_html_depth(rng: random.Random, n: int) -> list[dict]:
    out = []
    for k in range(n):
        val = _value(rng, "html_depth", k)
        done = f"<p>{_words(rng, 6)}</p>\n<p>{_words(rng, 5)}</p>\n"
        if val == "shallow":
            text = done + f"<p>{_words(rng, 3)} <b>{_words(rng, 2)}</b> {_words(rng, 2)}"
        else:
            text = done + f"<div><ul><li>{_words(rng, 7)}"
        out.append({"id": f"html_{k}_{val}", "feature": "html_depth", "cluster": f"html{k}", "value": val,
                    "text": text, "meta": {"depth": 1 if val == "shallow" else 3}})
    return out


# ------------------------------------------------------------------ stance_word (paragraphs)
def build_stance_word(paras: list[str], rng: random.Random) -> list[dict]:
    """Prefix a paragraph with 'Unfortunately,' / 'Fortunately,' (alternating) and read 15-30 words in."""
    out = []
    for k, p in enumerate(paras):
        val = FEATURES["stance_word"]["values"][k % 2]
        words = re.findall(r"\S+", p)
        if len(words) < 31:
            continue
        cut = rng.randint(15, 30)
        body = " ".join(words[:cut]).rstrip(".,;:")
        text = f"{val.capitalize()}, {body[0].lower() + body[1:]}"
        out.append({"id": f"stan_{k}_{val[:2]}", "feature": "stance_word", "cluster": f"stance{k}", "value": val,
                    "text": text, "meta": {"para": f"stance{k}", "words_in": cut}})
    return out


def length_band(items: list[dict], lo: int, hi: int) -> list[dict]:
    """Keep items whose token count is within [lo, hi] (used to length-match the paragraph rows)."""
    return [it for it in items if lo <= len(it["ids"]) <= hi]


BUILDERS = {"json_depth": build_json_depth, "latex_math": build_latex_math, "py_string": build_py_string,
            "call_arg": build_call_arg, "md_heading": build_md_heading, "csv_col": build_csv_col,
            "html_depth": build_html_depth}
