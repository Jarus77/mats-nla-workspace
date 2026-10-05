"""Frozen bank files: loading, the per-item target rule, and the readout-id rule."""

import json
import re
from pathlib import Path
from typing import Any

_LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def label_of(name: str) -> str:
    """The readout id of a bank item: its name with every character outside ``[A-Za-z0-9._-]``
    replaced by ``_`` (the rule every readout producer applies when a label becomes a file
    name, so ``br-proc-arith-2*3`` reads as ``br-proc-arith-2_3``)."""
    return _LABEL_UNSAFE.sub("_", name)


def load_bank(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(header, items)`` of a ``lens-eval-*.json`` bank; the header is every key but ``items``.
    Each item gets ``id = label_of(name)``; two names that share a label are an error."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    items = d["items"]
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise ValueError(f"{path}: 'items' must be a list of objects")
    seen: dict[str, str] = {}
    for it in items:
        label = label_of(it["name"])
        if label in seen:
            raise ValueError(
                f"{path}: items {seen[label]!r} and {it['name']!r} share label {label!r}"
            )
        seen[label] = it["name"]
        it["id"] = label
    header = {k: v for k, v in d.items() if k != "items"}
    return header, items


def item_targets(item: dict[str, Any]) -> list[str]:
    """Match strings for a single-token bank item, most curated source first: the headline
    units' ``match`` lists (all units when none is headline), then ``intermediates``; ``target``
    only when nothing else names the concept."""
    units = list(item.get("units") or [])
    chosen = [u for u in units if u.get("headline")] or units
    out: list[str] = []
    for u in chosen:
        for m in u.get("match", []):
            if m not in out:
                out.append(m)
    fallback = [] if out else [item.get("target")]
    for m in item.get("intermediates") or fallback:
        if m and m not in out:
            out.append(str(m))
    return out


def exact_targets(targets: list[str]) -> list[str]:
    """Drop targets that are a component of a longer sibling ("moon" beside "full moon"), so
    the judge is asked about whole concepts only."""
    return [t for t in targets if not any(_component_of(t, u) for u in targets)]


def _word_seq(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def _component_of(t: str, u: str) -> bool:
    tw, uw = _word_seq(t), _word_seq(u)
    if not tw or not uw:
        tl, ul = t.lower(), u.lower()
        return tl != ul and tl in ul
    if len(tw) >= len(uw):
        return False
    return any(uw[i : i + len(tw)] == tw for i in range(len(uw) - len(tw) + 1))
