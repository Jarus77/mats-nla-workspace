"""Frozen candidate lists per item: dictated concepts, the binding partner's, seeded draws from
the rest of the bank; built over the whole bank and pinned by the golden."""

import hashlib
import random
from collections.abc import Sequence
from typing import Any

from .prompts import N_OPTIONS, SEED


def _rng(key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest(), 16))


def partner_of(name: str) -> str | None:
    """The item this one is a binding pair with: ``d-fox-a`` <-> ``d-fox-b``, ``b-*-ab`` <->
    ``b-*-ba``. Their concepts differ only in which word binds to which, so the partner's
    concepts are the hardest distractors and are always offered."""
    for a, b in (("-ab", "-ba"), ("-a", "-b")):
        if name.endswith(a):
            return name[: -len(a)] + b
        if name.endswith(b):
            return name[: -len(b)] + a
    return None


def option_sets(items: Sequence[dict[str, Any]]) -> dict[str, tuple[list[str], list[int]]]:
    """``id -> (options, gold indices)``: the item's concepts, the partner's concepts it does not
    share, then seeded draws from other items' concepts up to :data:`N_OPTIONS`, shuffled with a
    seed keyed by the item name. Controls (no concepts) get six draws and no gold."""
    by_name = {it["name"]: it for it in items}
    out: dict[str, tuple[list[str], list[int]]] = {}
    for it in items:
        name = str(it["name"])
        own = [str(c) for c in it.get("concepts") or []]
        partner = by_name.get(partner_of(name) or "")
        extra = [str(c) for c in (partner or {}).get("concepts") or [] if c not in own]
        taken = set(own) | set(extra)
        pool = sorted(
            {
                str(c)
                for other in items
                if other is not it and other is not partner
                for c in other.get("concepts") or []
            }
            - taken
        )
        n_draw = max(0, N_OPTIONS - len(own) - len(extra))
        if n_draw > len(pool):
            raise ValueError(f"{name}: the bank has too few other concepts to fill the options")
        draws = _rng(f"{name}|draw").sample(pool, n_draw)
        options = own + extra + draws
        _rng(f"{name}|order").shuffle(options)
        out[it["id"]] = (options, [options.index(c) for c in own])
    return out
