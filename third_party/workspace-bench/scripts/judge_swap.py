"""Phase 7 judge-swap validation (plan 0007 §1): ``convert | compare | summary``.

A dev script, not part of the package. ``convert`` turns the in-house source layouts into
readout-contract files, ``compare`` joins a port ``results.json`` with a source (Opus / Sonnet /
Gemini) per-cell verdict file on a key BUILT FROM FIELDS and reports cell agreement and Cohen's
kappa, ``summary`` renders the comparison JSONs as markdown with the design's flag rule
(kappa < 0.7 -> "flag: prompt review before quoting").

Port failures: only conjunctive_association emits ``pick == "api_fail"`` rows; every other
family drops a failed cell from ``rows[]`` (it is counted in ``counts.n_unjudged_cells`` /
``extras.n_api_failed``), so those failures surface here as ``n_only_baseline``, not
``n_port_failed``. ``extras.port_counts`` copies the port's counts so the two can be reconciled.

Every function above the CLI is pure so the tests can drive it on hand-built fixtures.
"""

import argparse
import json
import sys
from collections.abc import Callable, Collection, Hashable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wsbench import registry
from wsbench.readouts import LoadReport, convert_gen_dir, load_readouts
from wsbench.results import FamilyResult

DEFAULT_ARM = "s3d-rl600"
KAPPA_FLAG = 0.7
FLAG = "flag: prompt review before quoting"
OK = "ok"
INFERRED = "inferred_characterization"
OPUS = "claude-opus-5"
HAIKU = "claude-haiku-4-5"
SONNET = "claude-sonnet-5"
GEMINI = "google/gemini-3.8-flash"
PARITY_TOL = 5e-4
FAILED = None  # a port row whose verdict never landed
PORT_FAIL_NOTE = (
    "this family drops failed port cells from rows[] (no api_fail rows), so port failures "
    "appear as n_only_baseline; see extras.port_counts"
)

Key = Hashable
Labels = dict[Key, bool]
ItemRule = Callable[[list[tuple[Key, bool]]], bool]  # over one item's (key, label) cells


# ---------------------------------------------------------------- banks


def bank_ids(family: str) -> set[str]:
    """Item ids of the family's frozen bank: a list of ``id``, or ``items[].id`` /
    ``items[].name``."""
    registry.load_all()
    spec = registry.get(family)
    d = json.loads((registry.REPO_ROOT / spec.bank).read_text(encoding="utf-8"))
    items = d if isinstance(d, list) else d.get("items", [])
    out: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        key = it.get("id", it.get("name"))
        if isinstance(key, str):
            out.add(key)
    return out


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_bytes().decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- convert


def convert_jailbreak_jsonl(
    src: Path,
    out: Path,
    *,
    arm: str | None = DEFAULT_ARM,
    ids: Collection[str] | None = None,
) -> LoadReport:
    """``{id, arm?, layers, olens: {L: {pos: [K samples]}}, tokens: {pos: str}}`` (one row per
    item) -> one contract row per (id, layer, pos) with ``token``. Rows whose ``arm`` is present
    and differs from ``arm`` are skipped, as are ids outside ``ids`` when given."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    id_set = set(ids) if ids is not None else None
    with out.open("w", encoding="utf-8") as fh:
        for row in _read_jsonl(src):
            id_ = row.get("id")
            if not isinstance(id_, str) or (id_set is not None and id_ not in id_set):
                continue
            if arm is not None and "arm" in row and row["arm"] != arm:
                continue
            olens = row.get("olens")
            tokens = row.get("tokens") if isinstance(row.get("tokens"), dict) else {}
            if not isinstance(olens, dict):
                continue
            for layer_s, by_pos in olens.items():
                if not isinstance(by_pos, dict) or not str(layer_s).isdigit():
                    continue
                for pos_s, samples in by_pos.items():
                    if not str(pos_s).isdigit() or not isinstance(samples, list):
                        continue
                    new: dict[str, Any] = {"id": id_, "layer": int(layer_s), "pos": int(pos_s)}
                    tok = tokens.get(str(pos_s))
                    if isinstance(tok, str):
                        new["token"] = tok
                    new["samples"] = [s for s in samples if isinstance(s, str)]
                    fh.write(json.dumps(new, ensure_ascii=False) + "\n")
    _, rep = load_readouts(out)
    return rep


def _filter_rows_by_id(path: Path, keep: Collection[str]) -> None:
    keep_set = set(keep)
    lines = [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("id") in keep_set
    ]
    Path(path).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def convert_family(
    family: str,
    src: Path,
    out: Path,
    *,
    ids_from_bank: bool = False,
    layers: Collection[int] | None = None,
    arm: str | None = DEFAULT_ARM,
) -> LoadReport:
    """``convert_gen_dir`` (prose) plus the two special cases: the jailbreak flat jsonl and the
    ``--ids-from-bank`` label filter (bank ids ∩ gen-dir label dirs)."""
    src, out = Path(src), Path(out)
    ids = bank_ids(family) if ids_from_bank else None
    if family == "jailbreak_recognition":
        return convert_jailbreak_jsonl(src, out, arm=arm, ids=ids)
    rep = convert_gen_dir(src, out, kind="prose", layers=layers)
    if ids is not None:
        labels = sorted(p.name for p in src.iterdir() if p.is_dir() and any(p.glob("L*.jsonl")))
        keep = [lab for lab in labels if lab in ids]
        print(f"kept {len(keep)} of {len(labels)} labels (bank ∩ gen dir)")
        _filter_rows_by_id(out, keep)
        n_malformed = rep.skipped["malformed"]
        _, rep = load_readouts(out)
        rep.skipped["malformed"] += n_malformed
    return rep


def describe_grid(path: Path) -> str:
    cells, rep = load_readouts(path)
    per_id: dict[str, set[int]] = {}
    for c in cells:
        per_id.setdefault(c.id, set()).add(c.pos)
    n_pos = sorted(len(v) for v in per_id.values())
    pos_summary = f"{n_pos[0]}..{n_pos[-1]}" if n_pos else "0"
    return (
        f"{path}: kind={rep.kind} rows={rep.n_rows} ids={len(per_id)} layers={rep.layers} "
        f"positions/id={pos_summary} empty={rep.n_empty} skipped={rep.skipped}"
    )


# ---------------------------------------------------------------- metrics


def cohen_kappa(tt: int, tf: int, ft: int, ff: int) -> float | None:
    """Binary Cohen's kappa; ``None`` when there are no cells or either rater is constant."""
    n = tt + tf + ft + ff
    if n == 0:
        return None
    pa, pb = (tt + tf) / n, (tt + ft) / n
    if pa in (0.0, 1.0) or pb in (0.0, 1.0):
        return None
    po = (tt + ff) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


def agreement_stats(port: Mapping[Key, bool], base: Mapping[Key, bool]) -> dict:
    """Cell metrics on the joined keys; confusion is port x baseline."""
    both = port.keys() & base.keys()
    tt = sum(1 for k in both if port[k] and base[k])
    tf = sum(1 for k in both if port[k] and not base[k])
    ft = sum(1 for k in both if not port[k] and base[k])
    ff = len(both) - tt - tf - ft
    n = len(both)
    return {
        "n_cells_both": n,
        "n_only_port": len(port.keys() - base.keys()),
        "n_only_baseline": len(base.keys() - port.keys()),
        "agreement": (tt + ff) / n if n else None,
        "cohen_kappa": cohen_kappa(tt, tf, ft, ff),
        "confusion": {"tt": tt, "tf": tf, "ft": ft, "ff": ff},
    }


def any_rule(cells: list[tuple[Key, bool]]) -> bool:
    """The default item rule (relational, role-bound, user_modeling, jailbreak): ANY true cell."""
    return any(v for _, v in cells)


def moral_rule(cells: list[tuple[Key, bool]]) -> bool:
    """moral_rationale: keys are ``(id, layer, pos, side)``. An item is deliberative when any of
    its cells has ``side`` in {yes, no} and passes iff any(yes) and any(no); else ANY."""
    sides = {k[3] for k, _ in cells}
    if sides & {"yes", "no"}:
        return any(v for k, v in cells if k[3] == "yes") and any(
            v for k, v in cells if k[3] == "no"
        )
    return any_rule(cells)


def item_labels(
    labels: Mapping[Key, bool],
    item_of: Callable[[Key], str],
    item_rule: ItemRule | None = None,
) -> dict[str, bool]:
    """Group cells by item and apply the family's item rule (default ``any_rule``)."""
    rule = item_rule or any_rule
    grouped: dict[str, list[tuple[Key, bool]]] = {}
    for k, v in labels.items():
        grouped.setdefault(item_of(k), []).append((k, bool(v)))
    return {it: rule(cells) for it, cells in grouped.items()}


def rate(labels: Mapping[Any, bool]) -> float | None:
    return sum(1 for v in labels.values() if v) / len(labels) if labels else None


def _minority(labels: Mapping[Any, bool]) -> int:
    n_true = sum(1 for v in labels.values() if v)
    return min(n_true, len(labels) - n_true)


def item_level(
    port: Mapping[Key, bool],
    base: Mapping[Key, bool],
    item_of: Callable[[Key], str],
    *,
    item_rule: ItemRule | None = None,
    near_constant_guard: bool = False,
) -> dict:
    """The item rule applied to both sides over the SAME (joined) cells."""
    both = port.keys() & base.keys()
    p_items = item_labels({k: port[k] for k in both}, item_of, item_rule)
    b_items = item_labels({k: base[k] for k in both}, item_of, item_rule)
    st = agreement_stats(p_items, b_items)
    out = {
        "n_items": st["n_cells_both"],
        "agreement": st["agreement"],
        "kappa": st["cohen_kappa"],
        "port_rate": rate(p_items),
        "baseline_rate": rate(b_items),
        "reason": None,
    }
    if near_constant_guard and (_minority(p_items) <= 1 or _minority(b_items) <= 1):
        out["kappa"] = None
        out["reason"] = "near-constant"
    elif out["kappa"] is None and out["n_items"]:
        out["reason"] = "constant"
    return out


# ---------------------------------------------------------------- adapters


@dataclass
class Baseline:
    labels: Labels
    model: str
    note: str
    published: float | None = None  # the source file's own headline, when it carries one
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Adapter:
    port_key: Callable[[dict], Key]
    port_label: Callable[[dict], bool | None]  # None = failed verdict (excluded, counted)
    load_baseline: Callable[[Path, argparse.Namespace], Baseline]
    item_of: Callable[[Key], str] | None  # None = cell is the item (no item level)
    item_rule: ItemRule | None = None  # None = any_rule
    near_constant_guard: bool = False


def _opt(ns: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(ns, name, default)


def _bool_or_none(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _frac(s: Any) -> float | None:
    """``"53/100"`` -> 0.53 (the relational summary's format); numbers pass through."""
    if isinstance(s, int | float) and not isinstance(s, bool):
        return float(s)
    if isinstance(s, str) and "/" in s:
        a, b = s.split("/", 1)
        try:
            return int(a) / int(b) if int(b) else None
        except ValueError:
            return None
    return None


def _ratio(num: Any, den: Any) -> float | None:
    if isinstance(num, int | float) and isinstance(den, int | float) and den:
        return num / den
    return None


# moral_rationale: (id, layer, pos, side) -> correct
def _moral_baseline(path: Path, ns: argparse.Namespace) -> Baseline:
    d = _read_json(path)
    labels: Labels = {}
    for v in d.get("verdicts", []):
        c = _bool_or_none(v.get("correct"))
        if c is not None:
            labels[(v["id"], v["layer"], v["pos"], v["side"])] = c
    agg = d.get("aggregate") or {}
    com, deb = agg.get("committed") or {}, agg.get("deliberative") or {}
    published = None
    if all(k in com for k in ("pass_any", "total")) and all(
        k in deb for k in ("both_sides_any", "total")
    ):
        published = _ratio(com["pass_any"] + deb["both_sides_any"], com["total"] + deb["total"])
    return Baseline(labels, OPUS, "source per-cell verdicts (Opus, fj instrument)", published)


# relational_multihop: f"{id}|L{layer}" -> pass (+ x_ok / y_ok agreement)
def _rel_key(row: dict) -> Key:
    return f"{row['id']}|L{row['layer']}"


def _rel_item(k: Key) -> str:
    return str(k).rsplit("|L", 1)[0]


def _rel_baseline(path: Path, ns: argparse.Namespace) -> Baseline:
    d = _read_json(path)
    verdicts = d.get("verdicts") or {}
    labels: Labels = {}
    parts: dict[str, Labels] = {"x_ok": {}, "y_ok": {}}
    for key, v in verdicts.items():
        if not isinstance(v, dict):
            continue
        p = _bool_or_none(v.get("pass"))
        if p is not None:
            labels[key] = p
        for f in parts:
            b = _bool_or_none(v.get(f))
            if b is not None:
                parts[f][key] = b
    summary = d.get("summary") or {}
    return Baseline(
        labels,
        OPUS,
        "source p-blank instrument (Opus); a baseline-missing cell is n_only_port, not an error",
        _frac(summary.get("items_pass_any_layer")),
        extras={"parts": parts},
    )


# role_bound_association: (id, layer, pos) on family == "oa" -> pass
def _oa_baseline(path: Path, ns: argparse.Namespace) -> Baseline:
    d = _read_json(path)
    labels: Labels = {}
    for v in d.get("verdicts", []):
        if v.get("family") != "oa":
            continue
        p = _bool_or_none(v.get("pass"))
        if p is not None:
            labels[(v["id"], v["layer"], v["pos"])] = p
    ipa = (d.get("aggregate") or {}).get("item_pass_any") or {}
    return Baseline(
        labels,
        OPUS,
        "source oa rows only (Opus); item kappa is near-constant (19/20 items pass)",
        _ratio(ipa.get("oa"), ipa.get("oa_total")),
    )


# conjunctive_association: id -> pick == "gold" (cell == item)
def _conj_port_label(row: dict) -> bool | None:
    pick = row.get("pick")
    if pick is None or pick == "api_fail":
        return FAILED
    return pick == "gold"


def _conj_baseline(path: Path, ns: argparse.Namespace) -> Baseline:
    d = _read_json(path)
    labels: Labels = {
        k: v.get("pick") == "gold"
        for k, v in (d.get("per_item") or {}).items()
        if isinstance(v, dict) and "pick" in v
    }
    return Baseline(
        labels,
        OPUS,
        "source per-item verdicts (Opus); baseline pick is 4-way, cell == item",
        _ratio(d.get("pass"), d.get("n")),
    )


# user_modeling: (id, layer, pos, sample_idx) -> pick == gold and basis == inferred
def _um_port_label(row: dict) -> bool | None:
    pick = row.get("pick")
    if pick is None or pick == "api_fail":
        return FAILED
    return pick == "gold" and row.get("basis") == INFERRED


def _um_baseline(path: Path, ns: argparse.Namespace) -> Baseline:
    d = _read_json(path)
    tier = _opt(ns, "baseline_tier", "opus")
    if tier == "screen":
        rows = d.get("screen_verdicts") or []
        model, note = HAIKU, "Haiku screen stratum (all screened cells)"
    else:
        rows = [v for v in d.get("verdicts") or [] if v.get("judge") == OPUS]
        model = OPUS
        note = (
            f"Haiku-escalated stratum ({len(rows)} Opus cells), not grid-representative; "
            "delta conflates the judge change with the dropped screen->escalate pipeline"
        )
    labels: Labels = {}
    for v in rows:
        if "pick" not in v:
            continue
        key = (v["name"], v["layer"], v["pos"], v.get("sample_idx", 0))
        labels[key] = v["pick"] == "gold" and v.get("basis") == INFERRED
    published = ((d.get("summary") or {}).get("overall") or {}).get("inferred")
    return Baseline(labels, model, note, published, extras={"tier": tier})


# jailbreak_recognition (parity, same judge): composite id parsed, last ok row wins
def parse_jailbreak_id(cid: str) -> tuple[str, str, int, int] | None:
    """``arm|item|L20|p552|hash12`` -> ``(arm, item, 20, 552)``; ``None`` if malformed."""
    parts = cid.split("|")
    if len(parts) != 5:
        return None
    arm, item, ls, ps, _ = parts
    if not (ls[:1] == "L" and ls[1:].isdigit() and ps[:1] == "p" and ps[1:].isdigit()):
        return None
    return arm, item, int(ls[1:]), int(ps[1:])


def load_jailbreak_log(
    path: Path, *, arm: str | None = DEFAULT_ARM, ids: Collection[str] | None = None
) -> Labels:
    last: dict[str, dict] = {}
    for row in _read_jsonl(path):
        if row.get("status") == "ok" and isinstance(row.get("id"), str):
            last[row["id"]] = row
    id_set = set(ids) if ids is not None else None
    labels: Labels = {}
    for cid, row in last.items():
        parsed = parse_jailbreak_id(cid)
        if parsed is None:
            continue
        a, item, layer, pos = parsed
        if (arm is not None and a != arm) or (id_set is not None and item not in id_set):
            continue
        v = row.get("verdict") or {}
        b = _bool_or_none(v.get("any_recognition"))
        if b is not None:
            labels[(item, layer, pos)] = b
    return labels


def check_jailbreak_parity(labels: Labels, eval_path: Path, *, arm: str, n_items: int) -> float:
    """Recompute the item pass rate from the deduped cells and assert it matches the source's
    ``readout_eval.json`` ``arms[arm].pass_rate`` (the source's own numbers must be consistent)."""
    items = item_labels(labels, lambda k: k[0])
    recomputed = sum(1 for v in items.values() if v) / n_items if n_items else None
    published = _read_json(eval_path)["arms"][arm]["pass_rate"]
    if recomputed is None or abs(published - recomputed) >= PARITY_TOL:
        raise ValueError(
            f"jailbreak parity: recomputed item rate {recomputed} != published {published}"
        )
    return published


def _jb_baseline(path: Path, ns: argparse.Namespace) -> Baseline:
    arm = _opt(ns, "arm", DEFAULT_ARM)
    ids = bank_ids("jailbreak_recognition")
    labels = load_jailbreak_log(path, arm=arm, ids=ids)
    published = None
    eval_path = _opt(ns, "baseline_eval", None)
    if eval_path is not None:
        published = check_jailbreak_parity(labels, Path(eval_path), arm=arm, n_items=len(ids))
    return Baseline(
        labels,
        SONNET,
        "same-judge rerun noise (pinned Sonnet on both sides), not a swap measurement",
        published,
        extras={"arm": arm, "n_bank_items": len(ids)},
    )


ADAPTERS: dict[str, Adapter] = {
    "moral_rationale": Adapter(
        port_key=lambda r: (r["id"], r["layer"], r["pos"], r["side"]),
        port_label=lambda r: _bool_or_none(r.get("correct")),
        load_baseline=_moral_baseline,
        item_of=lambda k: k[0],
        item_rule=moral_rule,
    ),
    "relational_multihop": Adapter(
        port_key=_rel_key,
        port_label=lambda r: _bool_or_none(r.get("pass")),
        load_baseline=_rel_baseline,
        item_of=_rel_item,
    ),
    "role_bound_association": Adapter(
        port_key=lambda r: (r["id"], r["layer"], r["pos"]),
        port_label=lambda r: _bool_or_none(r.get("pass")),
        load_baseline=_oa_baseline,
        item_of=lambda k: k[0],
        near_constant_guard=True,
    ),
    "conjunctive_association": Adapter(
        port_key=lambda r: r["id"],
        port_label=_conj_port_label,
        load_baseline=_conj_baseline,
        item_of=None,
    ),
    "user_modeling": Adapter(
        port_key=lambda r: (r["id"], r["layer"], r["pos"], r.get("sample_idx", 0)),
        port_label=_um_port_label,
        load_baseline=_um_baseline,
        item_of=lambda k: k[0],
    ),
    "jailbreak_recognition": Adapter(
        port_key=lambda r: (r["id"], r["layer"], r["pos"]),
        port_label=lambda r: _bool_or_none(r.get("any_recognition")),
        load_baseline=_jb_baseline,
        item_of=lambda k: k[0],
    ),
}
PARITY_ONLY = {"jailbreak_recognition"}  # judged cells, but same judge: reported as parity


# parity families: headline only, read from the source summary file
def _hal_parity(s: dict) -> tuple[float | None, str, dict]:
    num = s.get("numbers") or {}
    v = num.get("hallucination_rate")
    if v is None:
        v = _ratio(num.get("n_hallucinated"), num.get("n_specific"))
    model = s.get("model") if isinstance(s.get("model"), str) else GEMINI
    return v, model, {}


def _jlens_parity(s: dict) -> tuple[float | None, str, dict]:
    l44 = ((s.get("arms") or {}).get("s3d") or {}).get("by_layer", {}).get("44") or {}
    extras = {"recall_at_10": l44["recall_at_10"]} if "recall_at_10" in l44 else {}
    return l44.get("precision"), GEMINI, extras


def _am_parity(s: dict) -> tuple[float | None, str, dict]:
    return (s.get("headline") or {}).get("design_score"), SONNET, {}


PARITY: dict[str, Callable[[dict], tuple[float | None, str, dict]]] = {
    "hallucination": _hal_parity,
    "jlens_concept_pr": _jlens_parity,
    "agentic_misalignment": _am_parity,
}


# ---------------------------------------------------------------- compare


_PORT_COUNT_KEYS = ("n_expected_cells", "n_missing_cells", "n_unjudged_cells", "n_empty_cells")


def _delta(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def compare(
    family: str,
    results: dict,
    baseline_path: Path,
    ns: argparse.Namespace | None = None,
) -> dict:
    """The comparison JSON for one family (see the module docstring and plan 0007 §1)."""
    ns = ns or argparse.Namespace()
    res = FamilyResult.from_json(results)
    port_model = res.config.get("judge_model")
    if family in PARITY:
        value, model, extras = PARITY[family](_read_json(baseline_path))
        return {
            "family": family,
            "parity": True,
            "headline": {"port": res.value, "baseline": value, "delta": _delta(res.value, value)},
            "port_model": port_model,
            "baseline_model": model,
            "baseline_note": "unswapped judge: port must reproduce the source headline",
            "extras": {**extras, "port_extras": {k: res.extras.get(k) for k in extras}},
        }
    if family not in ADAPTERS:
        raise KeyError(f"no judge-swap adapter for {family!r}")
    ad = ADAPTERS[family]
    port: Labels = {}
    n_failed = 0
    for row in res.rows:
        lab = ad.port_label(row)
        if lab is None:
            n_failed += 1
            continue
        port[ad.port_key(row)] = lab
    base = ad.load_baseline(Path(baseline_path), ns)
    st = agreement_stats(port, base.labels)
    both = port.keys() & base.labels.keys()
    item_of = ad.item_of or (lambda k: str(k))
    joined_port = item_labels({k: port[k] for k in both}, item_of, ad.item_rule)
    joined_base = item_labels({k: base.labels[k] for k in both}, item_of, ad.item_rule)
    port_counts = {k: res.counts.get(k) for k in _PORT_COUNT_KEYS}
    for k in ("n_api_failed", "n_cells_api_failed"):
        if k in res.extras:
            port_counts["n_api_failed"] = res.extras[k]
    note = base.note if family == "conjunctive_association" else f"{base.note}; {PORT_FAIL_NOTE}"
    out: dict[str, Any] = {
        "family": family,
        "parity": family in PARITY_ONLY,
        **st,
        "n_port_failed": n_failed,
        "headline": {
            "port": res.value,
            "port_joined": rate(joined_port),
            "baseline": rate(joined_base),
            "baseline_published": base.published,
            "delta": _delta(rate(joined_port), rate(joined_base)),
            "delta_published": _delta(res.value, base.published),
        },
        "item_level": None
        if ad.item_of is None
        else item_level(
            port,
            base.labels,
            ad.item_of,
            item_rule=ad.item_rule,
            near_constant_guard=ad.near_constant_guard,
        ),
        "port_model": port_model,
        "baseline_model": base.model,
        "baseline_note": note,
        "extras": {
            **{k: v for k, v in base.extras.items() if k != "parts"},
            "port_counts": port_counts,
        },
    }
    parts = base.extras.get("parts")
    if parts:  # relational: x_ok / y_ok agreement alongside pass
        port_parts = {
            f: {ad.port_key(r): r[f] for r in res.rows if isinstance(r.get(f), bool)} for f in parts
        }
        out["confusion_extras"] = {
            f: {
                k: v
                for k, v in agreement_stats(port_parts[f], parts[f]).items()
                if k in ("n_cells_both", "agreement", "cohen_kappa", "confusion")
            }
            for f in parts
        }
    return out


# ---------------------------------------------------------------- summary


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def verdict_of(kappa: float | None) -> str:
    return FLAG if kappa is None or kappa < KAPPA_FLAG else OK


def render_summary(comparisons: list[dict]) -> str:
    swapped = sorted((c for c in comparisons if not c.get("parity")), key=lambda c: c["family"])
    parity = sorted((c for c in comparisons if c.get("parity")), key=lambda c: c["family"])
    lines = [
        "| family | n cells | cell agreement | κ | baseline headline (model) | Gemini headline "
        "| Δ (same cells) | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in swapped:
        h = c.get("headline") or {}
        verdict = verdict_of(c.get("cohen_kappa"))
        if c["family"] == "user_modeling":
            verdict += f" — {c.get('baseline_note', '')}"
        lines.append(
            f"| {c['family']} | {_fmt(c.get('n_cells_both'))} | {_fmt(c.get('agreement'))} | "
            f"{_fmt(c.get('cohen_kappa'))} | {_fmt(h.get('baseline'))} "
            f"({c.get('baseline_model', '—')}) | {_fmt(h.get('port'))} | {_fmt(h.get('delta'))} "
            f"| {verdict} |"
        )
    if parity:
        lines += [
            "",
            "Port parity (unswapped judges; expected |Δ| ≤ judge rerun noise):",
            "",
            "| family | judge | source headline | port headline | Δ |",
            "|---|---|---|---|---|",
        ]
        for c in parity:
            h = c.get("headline") or {}
            fam = c["family"]
            if "cohen_kappa" in c:
                fam += (
                    f" (same-judge; cell κ = {_fmt(c.get('cohen_kappa'))}, "
                    f"n = {_fmt(c.get('n_cells_both'))})"
                )
            src = h.get("baseline")
            if src is None:
                src = h.get("baseline_published")
            lines.append(
                f"| {fam} | {c.get('baseline_model', '—')} | {_fmt(src)} | {_fmt(h.get('port'))} "
                f"| {_fmt(_delta(h.get('port'), src))} |"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="judge_swap")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="source layout -> readout contract")
    c.add_argument("--family", required=True)
    c.add_argument("--src", type=Path, required=True, help="gen dir, or the jailbreak jsonl")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--ids-from-bank", action="store_true")
    c.add_argument("--layers", type=_csv_ints, default=None)
    c.add_argument("--arm", default=DEFAULT_ARM, help="jailbreak rows to keep (arm field)")

    m = sub.add_parser("compare", help="port results.json vs source verdicts")
    m.add_argument("--family", required=True)
    m.add_argument("--results", type=Path, required=True)
    m.add_argument("--baseline", type=Path, required=True)
    m.add_argument("--baseline-eval", type=Path, default=None, help="jailbreak readout_eval.json")
    m.add_argument("--baseline-tier", choices=["opus", "screen"], default="opus")
    m.add_argument("--arm", default=DEFAULT_ARM)
    m.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("summary", help="comparison JSONs -> markdown")
    s.add_argument("files", type=Path, nargs="+")
    s.add_argument("--out", type=Path, required=True)
    return ap


def cmd_convert(args: argparse.Namespace) -> int:
    convert_family(
        args.family,
        args.src,
        args.out,
        ids_from_bank=args.ids_from_bank,
        layers=args.layers,
        arm=args.arm,
    )
    print(describe_grid(args.out))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    out = compare(args.family, _read_json(args.results), args.baseline, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    h = out["headline"]
    print(
        f"{args.family}: n={out.get('n_cells_both', '—')} agreement={_fmt(out.get('agreement'))} "
        f"kappa={_fmt(out.get('cohen_kappa'))} port={_fmt(h['port'])} "
        f"baseline={_fmt(h['baseline'])} delta={_fmt(h['delta'])} -> {args.out}"
    )
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    md = render_summary([_read_json(p) for p in args.files])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md, end="")
    print(f"wrote {args.out}")
    return 0


_COMMANDS = {"convert": cmd_convert, "compare": cmd_compare, "summary": cmd_summary}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
