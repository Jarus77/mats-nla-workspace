"""Pilot item generator (runs in a Modal container that mounts /results). DECLARED PILOT SPLIT:
every passage here is recorded in pilot_split.json and must be EXCLUDED from the main analysis.

n = 60 col items. Each item = one fresh wikitext paragraph hard-wrapped at a width from the
prereg grid {30, 40, 50, 60, 70} (12 passages per width, cycled), using 02's paragraph cleaner and
textwrap recipe (s1_data.build_wrap; same col definition: characters since the last newline after
the read token). One read position per passage: a random in-line token (not containing a
newline) with token index >= 8, col >= 1, and at least one newline before it (so the line is a
genuine wrapped line). ids are truncated so the read token is the LAST token.

Disjointness: paragraphs come from the cleaned wikitext pool shuffled with PILOT_SEED (8008),
excluding (a) every paragraph in 02's wrap/prose pickles (02's gating/probe/dose data) and
(b) every passage used by 07_template_mining (items_base.json), matched on whitespace-normalised
text prefixes. 02 used seed 1 (first 900 of its shuffle), 07 used seed 7007.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import random
import sys
import textwrap
import types

PILOT_SEED = 8008
N_ITEMS = 60
WIDTHS = (30, 40, 50, 60, 70)
MIN_POS = 8
MAX_TOKENS = 320
E02 = "/root/ext02"


def load_02(name: str, out02: str) -> types.ModuleType:
    """Import a read-only copy of a 02 module with a stub nonj_common (no Modal image build)."""
    if "nonj_common" not in sys.modules:
        stub = types.ModuleType("nonj_common")
        stub.OUT, stub.VOLUMES, stub.nonj_image, stub.net_image = out02, {}, None, None
        sys.modules["nonj_common"] = stub
    spec = importlib.util.spec_from_file_location(f"e02_{name}", f"{E02}/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _norm(t: str) -> str:
    return " ".join(t.split())


def _used_prefixes(out02: str, out07: str) -> list[str]:
    used = []
    for name in ("wrap", "prose"):
        with open(f"{out02}/data/{name}.pkl", "rb") as f:
            used += [_norm(it["text"]) for it in pickle.load(f)]
    try:
        with open(f"{out07}/items_base.json") as f:
            used += [_norm(it["text"]) for it in json.load(f) if it.get("text")]
    except FileNotFoundError:
        pass
    return [u for u in used if len(u) >= 30]


def _overlaps(p: str, used_by_head: dict[str, list[str]]) -> bool:
    n = _norm(p)
    for u in used_by_head.get(n[:30], []):
        if n.startswith(u[: len(n)]) or u.startswith(n[: len(u)]):
            return True
    return False


def _wrap_item(tok, para: str, width: int, rng: random.Random) -> dict | None:  # noqa: ANN001
    text = textwrap.fill(para, width=width, break_long_words=False)[:1100]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"][:MAX_TOKENS], enc["offset_mapping"][:MAX_TOKENS]
    cand = []
    for j, (s, e) in enumerate(offs):
        last = text[:e].rfind("\n")
        col = e - (last + 1)
        if j >= MIN_POS and last >= 0 and "\n" not in text[s:e] and col >= 1:
            cand.append((j, col))
    if not cand:
        return None
    j, col = rng.choice(cand)
    end = offs[j][1]
    nxt = offs[j + 1] if j + 1 < len(offs) else None
    line_start = text[:end].rfind("\n") + 1
    nl_after = text.find("\n", end)
    line_len = (nl_after if nl_after >= 0 else len(text)) - line_start
    return {"ids": ids[: j + 1], "text": text[:end], "col": int(col), "width": width, "pos": j,
            "line_len": int(line_len), "chars_remaining": int(line_len - col),
            "next_is_nl": bool(nxt and "\n" in text[nxt[0]:nxt[1]]),
            "line_index": text[:end].count("\n")}


def build(tok, out02: str, out07: str) -> list[dict]:  # noqa: ANN001
    s1 = load_02("s1_data", out02)
    pool = s1._wikitext_paragraphs(900, 60000, PILOT_SEED)
    used = _used_prefixes(out02, out07)
    by_head: dict[str, list[str]] = {}
    for u in used:
        by_head.setdefault(u[:30], []).append(u)
    rng = random.Random(PILOT_SEED)
    items, k = [], 0
    for para in pool:
        if len(items) == N_ITEMS:
            break
        if _overlaps(para, by_head):
            continue
        width = WIDTHS[len(items) % len(WIDTHS)]
        it = _wrap_item(tok, para, width, rng)
        if it is None:
            continue
        pid = hashlib.sha1(_norm(para).encode()).hexdigest()[:16]
        items.append({"id": f"pilot_col_{len(items):02d}", "passage_sha1_16": pid,
                      "passage_head": _norm(para)[:80], **it})
        k += 1
    if len(items) != N_ITEMS:
        raise RuntimeError(f"only {len(items)} pilot items")
    return items
