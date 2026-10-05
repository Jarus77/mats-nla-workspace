"""Offline tests for ``scripts/judge_swap.py`` (plan 0007 §1) on miniature hand-built fixtures.

The script is not part of the package, so it is loaded by path. No readout text or verdict from
any source file appears here; every cell is synthetic.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from wsbench.readouts import load_readouts

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "judge_swap.py"


def _load():
    spec = importlib.util.spec_from_file_location("judge_swap", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["judge_swap"] = mod
    spec.loader.exec_module(mod)
    return mod


js = _load()


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _results(family: str, rows: list[dict], value: float | None = 0.5, extras=None) -> dict:
    return {
        "schema_version": 1,
        "family": family,
        "complete": True,
        "pinned_instrument": True,
        "config": {"judge_model": "google/gemini-3.8-flash"},
        "n_items": len({r.get("id") for r in rows}),
        "counts": {},
        "numbers": {
            "metric": "pass_rate",
            "value": value,
            "ci95": None,
            "chance": None,
            "chance_label": None,
            "higher_is_better": True,
            "extras": extras or {},
        },
        "rows": rows,
    }


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


# ---------------------------------------------------------------- metrics


def test_kappa_none_when_constant():
    assert js.cohen_kappa(0, 0, 0, 0) is None
    assert js.cohen_kappa(3, 0, 0, 0) is None  # both constant
    assert js.cohen_kappa(2, 1, 0, 0) is None  # port constant true
    assert js.cohen_kappa(2, 0, 1, 0) is None  # baseline constant true
    assert js.cohen_kappa(2, 1, 1, 2) == pytest.approx(1 / 3)


def test_agreement_three_of_four_gives_kappa_half():
    port = {"a": True, "b": True, "c": False, "d": False, "only_port": True}
    base = {"a": True, "b": True, "c": False, "d": True, "only_base": False}
    st = js.agreement_stats(port, base)
    assert st["n_cells_both"] == 4
    assert st["n_only_port"] == 1 and st["n_only_baseline"] == 1
    assert st["agreement"] == pytest.approx(0.75)
    assert st["cohen_kappa"] == pytest.approx(0.5)
    assert st["confusion"] == {"tt": 2, "tf": 0, "ft": 1, "ff": 1}


# ---------------------------------------------------------------- convert


def test_convert_gen_dir_ids_from_bank(tmp_path, monkeypatch, capsys):
    gen = tmp_path / "gen"
    for label in ("oa-keep", "eb-drop"):
        _write_jsonl(
            gen / label / "L036.jsonl",
            [{"pos": 3, "samples": ["s1", "s2"]}, {"pos": 4, "samples": ["s3"]}],
        )
    (gen / "judge").mkdir()  # the sbum dir's judge/ subdir is not a label
    (gen / "judge" / "user-modeling.json").write_text("{}")
    monkeypatch.setattr(js, "bank_ids", lambda family: {"oa-keep", "oa-absent", "judge"})
    out = tmp_path / "out.jsonl"
    rep = js.convert_family("role_bound_association", gen, out, ids_from_bank=True)
    cells, rep2 = load_readouts(out)
    assert {c.id for c in cells} == {"oa-keep"}
    assert rep.n_rows == rep2.n_rows == 2 and rep.kind == "prose"
    assert sum(rep2.skipped.values()) == 0
    assert "kind=prose" in js.describe_grid(out)
    assert "kept 1 of 2 labels" in capsys.readouterr().out


def test_convert_gen_dir_without_bank_keeps_all(tmp_path):
    gen = tmp_path / "gen"
    _write_jsonl(gen / "a" / "L020.jsonl", [{"pos": 1, "samples": ["x"]}])
    _write_jsonl(gen / "b" / "L020.jsonl", [{"pos": 1, "samples": ["y"]}])
    out = tmp_path / "out.jsonl"
    rep = js.convert_family("moral_rationale", gen, out)
    assert rep.n_rows == 2 and rep.layers == [20]


def test_convert_jailbreak_jsonl(tmp_path, monkeypatch):
    src = _write_jsonl(
        tmp_path / "lens.jsonl",
        [
            {
                "id": "item-1",
                "arm": "s3d-rl600",
                "layers": [20, 44],
                "olens": {
                    "20": {"5": ["a", "b"], "9": ["c", "d"]},
                    "44": {"5": ["e", "f"], "9": ["g", "h"]},
                },
                "tokens": {"5": "tok5", "9": "tok9"},
            },
            {"id": "item-1", "arm": "other", "olens": {"20": {"5": ["z"]}}, "tokens": {}},
            {"id": "not-in-bank", "arm": "s3d-rl600", "olens": {"20": {"5": ["z"]}}, "tokens": {}},
        ],
    )
    out = tmp_path / "jb.jsonl"
    rep = js.convert_jailbreak_jsonl(src, out, arm="s3d-rl600", ids={"item-1"})
    cells, rep2 = load_readouts(out)
    assert rep.n_rows == 4 and sum(rep2.skipped.values()) == 0 and rep2.kind == "prose"
    assert {(c.layer, c.pos) for c in cells} == {(20, 5), (20, 9), (44, 5), (44, 9)}
    assert all(c.token == f"tok{c.pos}" for c in cells)
    assert all(len(c.samples) == 2 for c in cells)
    # the family dispatch reaches the same code path and honours --ids-from-bank
    monkeypatch.setattr(js, "bank_ids", lambda family: {"item-1"})
    rep3 = js.convert_family(
        "jailbreak_recognition", src, tmp_path / "jb2.jsonl", ids_from_bank=True
    )
    assert rep3.n_rows == 4


# ---------------------------------------------------------------- compare: adapters


def _moral_rows():
    # i1 committed (2 cells), i2 deliberative (yes+no, both fail), i4 deliberative (both pass),
    # i3 only on the port side
    return [
        {"id": "i1", "layer": 20, "pos": 7, "side": "committed", "pick": "gold", "correct": True},
        {"id": "i1", "layer": 36, "pos": 7, "side": "committed", "pick": "gold", "correct": True},
        {"id": "i2", "layer": 20, "pos": 9, "side": "yes", "pick": "other", "correct": False},
        {"id": "i2", "layer": 20, "pos": 9, "side": "no", "pick": "other", "correct": False},
        {"id": "i4", "layer": 20, "pos": 9, "side": "yes", "pick": "gold", "correct": True},
        {"id": "i4", "layer": 20, "pos": 9, "side": "no", "pick": "gold", "correct": True},
        {"id": "i3", "layer": 20, "pos": 1, "side": "committed", "pick": "gold", "correct": True},
    ]


def _moral_baseline():
    return {
        "aggregate": {
            "committed": {"pass_any": 3, "total": 4},
            "deliberative": {"both_sides_any": 1, "total": 4},
        },
        "verdicts": [
            {"id": "i1", "layer": 20, "pos": 7, "side": "committed", "correct": True},
            {"id": "i1", "layer": 36, "pos": 7, "side": "committed", "correct": True},
            {"id": "i2", "layer": 20, "pos": 9, "side": "yes", "correct": False},
            {"id": "i2", "layer": 20, "pos": 9, "side": "no", "correct": True},  # one side only
            {"id": "i4", "layer": 20, "pos": 9, "side": "yes", "correct": True},
            {"id": "i4", "layer": 20, "pos": 9, "side": "no", "correct": False},  # one side only
            {"id": "i9", "layer": 20, "pos": 9, "side": "committed", "correct": False},
        ],
    }


def test_moral_rule_committed_vs_deliberative():
    committed = [(("i", 20, 1, "committed"), False), (("i", 36, 1, "committed"), True)]
    assert js.moral_rule(committed) is True  # ANY
    one_side = [(("i", 20, 1, "yes"), True), (("i", 20, 1, "no"), False)]
    assert js.any_rule(one_side) is True and js.moral_rule(one_side) is False
    both = [(("i", 20, 1, "yes"), True), (("i", 36, 1, "no"), True)]
    assert js.moral_rule(both) is True
    no_side_cells = [(("i", 20, 1, "yes"), False)]  # deliberative with only a yes cell
    assert js.moral_rule(no_side_cells) is False


def test_compare_moral_rationale(tmp_path):
    out = js.compare(
        "moral_rationale",
        _results("moral_rationale", _moral_rows(), value=0.6),
        _write_json(tmp_path / "b.json", _moral_baseline()),
    )
    assert out["parity"] is False
    assert out["n_cells_both"] == 6 and out["n_only_port"] == 1 and out["n_only_baseline"] == 1
    # cells port T T F F T T vs base T T F T T F -> 4/6 agree; tt=3 tf=1 ft=1 ff=1 -> kappa 0.25
    assert out["agreement"] == pytest.approx(4 / 6)
    assert out["cohen_kappa"] == pytest.approx(0.25)
    assert out["confusion"] == {"tt": 3, "tf": 1, "ft": 1, "ff": 1}
    assert out["n_port_failed"] == 0
    h = out["headline"]
    assert h["port"] == 0.6
    assert h["baseline_published"] == pytest.approx(0.5)
    # deliberative items need BOTH sides: baseline passes only i1 (i2/i4 have one side each),
    # the port passes i1 and i4. ANY would have given the baseline 3/3 — the rule is locked in.
    assert h["baseline"] == pytest.approx(1 / 3)
    assert h["port_joined"] == pytest.approx(2 / 3)
    assert h["delta"] == pytest.approx(2 / 3 - 1 / 3)  # same cells, same rule
    assert h["delta_published"] == pytest.approx(0.6 - 0.5)
    il = out["item_level"]
    assert il["n_items"] == 3 and il["agreement"] == pytest.approx(2 / 3)
    assert il["kappa"] == pytest.approx(0.4)
    assert il["port_rate"] == pytest.approx(2 / 3) and il["baseline_rate"] == pytest.approx(1 / 3)
    assert out["baseline_model"] == "claude-opus-5"
    assert out["port_model"] == "google/gemini-3.8-flash"
    assert "n_only_baseline" in out["baseline_note"]  # non-conj: failures are dropped rows
    assert out["extras"]["port_counts"]["n_unjudged_cells"] is None  # fixture carries no counts


def test_compare_relational_multihop(tmp_path):
    rows = [
        {"id": "rel-1", "layer": 20, "pos": 5, "x_ok": True, "y_ok": True, "pass": True},
        {"id": "rel-1", "layer": 36, "pos": 5, "x_ok": True, "y_ok": False, "pass": False},
        {"id": "rel-2", "layer": 20, "pos": 5, "x_ok": False, "y_ok": True, "pass": False},
        {
            "id": "rel-2",
            "layer": 60,
            "pos": 5,
            "x_ok": True,
            "y_ok": True,
            "pass": True,
        },  # L60 fail
    ]
    base = {
        "summary": {"items_pass_any_layer": "53/100"},
        "verdicts": {
            "rel-1|L20": {"pass": True, "x_ok": True, "y_ok": True},
            "rel-1|L36": {"pass": False, "x_ok": False, "y_ok": False},
            "rel-2|L20": {"pass": True, "x_ok": True, "y_ok": True},
        },
    }
    out = js.compare(
        "relational_multihop",
        _results("relational_multihop", rows, value=0.5),
        _write_json(tmp_path / "b.json", base),
    )
    assert out["n_cells_both"] == 3 and out["n_only_port"] == 1 and out["n_only_baseline"] == 0
    assert out["agreement"] == pytest.approx(2 / 3)
    assert out["headline"]["baseline_published"] == pytest.approx(0.53)
    assert out["item_level"]["n_items"] == 2
    ce = out["confusion_extras"]
    assert ce["x_ok"]["n_cells_both"] == 3 and ce["x_ok"]["agreement"] == pytest.approx(1 / 3)
    assert ce["y_ok"]["agreement"] == 1.0  # y_ok T,F,T on both sides


def test_compare_role_bound_near_constant(tmp_path):
    rows, verdicts = [], []
    for i in range(4):
        for layer in (20, 36):
            p = not (i == 3 and layer == 36)
            rows.append(
                {"id": f"oa-{i}", "layer": layer, "pos": 2, "correct": [p, p, p], "pass": p}
            )
            verdicts.append({"family": "oa", "id": f"oa-{i}", "layer": layer, "pos": 2, "pass": p})
    verdicts.append({"family": "eb", "id": "eb-0", "layer": 20, "pos": 2, "pass": True})
    base = {"aggregate": {"item_pass_any": {"oa": 19, "oa_total": 20}}, "verdicts": verdicts}
    out = js.compare(
        "role_bound_association",
        _results("role_bound_association", rows, value=1.0),
        _write_json(tmp_path / "b.json", base),
    )
    assert out["n_cells_both"] == 8 and out["n_only_baseline"] == 0  # eb rows ignored
    assert out["agreement"] == 1.0 and out["cohen_kappa"] == 1.0
    assert out["headline"]["baseline_published"] == pytest.approx(0.95)
    assert out["item_level"]["kappa"] is None
    assert out["item_level"]["reason"] == "near-constant"
    assert out["item_level"]["n_items"] == 4


def test_compare_conjunctive_and_n_port_failed(tmp_path):
    rows = [
        {"id": "q1", "pick": "gold", "correct": True},
        {"id": "q2", "pick": "contrast", "correct": False},
        {"id": "q3", "pick": "gold", "correct": True},
        {"id": "q4", "pick": "cannot_tell", "correct": False},
        {"id": "q5", "pick": "api_fail", "correct": False},
    ]
    base = {
        "n": 100,
        "pass": 52,
        "per_item": {
            "q1": {"pick": "gold", "correct": True},
            "q2": {"pick": "other", "correct": False},
            "q3": {"pick": "cannot_tell", "correct": False},
            "q4": {"pick": "contrast", "correct": False},
            "q5": {"pick": "gold", "correct": True},
        },
    }
    out = js.compare(
        "conjunctive_association",
        _results("conjunctive_association", rows, value=0.4),
        _write_json(tmp_path / "b.json", base),
    )
    assert out["n_port_failed"] == 1
    assert out["n_cells_both"] == 4 and out["n_only_baseline"] == 1  # q5 failed on the port side
    assert out["agreement"] == pytest.approx(0.75)
    assert out["item_level"] is None
    assert "n_only_baseline" not in out["baseline_note"]  # conj is the one api_fail family
    assert out["headline"]["baseline"] == pytest.approx(0.25)
    assert out["headline"]["baseline_published"] == pytest.approx(0.52)


def _um_rows():
    return [
        {
            "id": "um-1",
            "layer": 20,
            "pos": 3,
            "sample_idx": 0,
            "pick": "gold",
            "basis": js.INFERRED,
        },
        {
            "id": "um-1",
            "layer": 36,
            "pos": 3,
            "sample_idx": 0,
            "pick": "gold",
            "basis": "verbatim_echo",
        },
        {
            "id": "um-2",
            "layer": 20,
            "pos": 3,
            "sample_idx": 0,
            "pick": "cannot_tell",
            "basis": "absent",
        },
        {
            "id": "um-2",
            "layer": 36,
            "pos": 3,
            "sample_idx": 0,
            "pick": "distractor",
            "basis": "absent",
        },
        {
            "id": "um-3",
            "layer": 20,
            "pos": 3,
            "sample_idx": 0,
            "pick": "gold",
            "basis": js.INFERRED,
        },
    ]


def _um_baseline():
    def row(name, layer, pick, basis, judge):
        return {
            "name": name,
            "layer": layer,
            "pos": 3,
            "sample_idx": 0,
            "pick": pick,
            "basis": basis,
            "judge": judge,
        }

    return {
        "summary": {"overall": {"inferred": 0.16}},
        "verdicts": [
            row("um-1", 20, "gold", js.INFERRED, "claude-opus-5"),
            row("um-1", 36, "cannot_tell", "absent", "claude-opus-5"),
            row("um-2", 20, "cannot_tell", "absent", "claude-opus-5"),
            {"name": "um-2", "layer": 36, "pos": 3, "sample_idx": 0, "judge": "unavailable"},
            {"name": "um-3", "layer": 20, "pos": 3, "sample_idx": 0, "judge": "unavailable"},
        ],
        "screen_verdicts": [
            row("um-1", 20, "gold", js.INFERRED, "claude-haiku-4-5"),
            row("um-1", 36, "gold", js.INFERRED, "claude-haiku-4-5"),
            row("um-2", 20, "cannot_tell", "absent", "claude-haiku-4-5"),
            row("um-2", 36, "cannot_tell", "absent", "claude-haiku-4-5"),
            row("um-3", 20, "gold", js.INFERRED, "claude-haiku-4-5"),
        ],
    }


def test_compare_user_modeling_tiers(tmp_path):
    results = _results("user_modeling", _um_rows(), value=2 / 3)
    b = _write_json(tmp_path / "um.json", _um_baseline())
    opus = js.compare("user_modeling", results, b, _ns(baseline_tier="opus"))
    assert opus["baseline_model"] == "claude-opus-5"
    assert opus["n_cells_both"] == 3 and opus["n_only_port"] == 2 and opus["n_only_baseline"] == 0
    assert opus["agreement"] == 1.0 and opus["cohen_kappa"] == 1.0
    assert "3 Opus cells" in opus["baseline_note"] and "conflates" in opus["baseline_note"]
    assert opus["headline"]["baseline_published"] == pytest.approx(0.16)
    assert opus["item_level"]["n_items"] == 2
    assert opus["extras"]["tier"] == "opus"

    screen = js.compare("user_modeling", results, b, _ns(baseline_tier="screen"))
    assert screen["baseline_model"] == "claude-haiku-4-5"
    assert screen["n_cells_both"] == 5
    assert screen["agreement"] == pytest.approx(0.8)  # um-1 L36 verbatim_echo vs inferred
    assert screen["extras"]["tier"] == "screen"


def test_compare_default_namespace_is_opus(tmp_path):
    out = js.compare(
        "user_modeling",
        _results("user_modeling", _um_rows()),
        _write_json(tmp_path / "um.json", _um_baseline()),
    )
    assert out["baseline_model"] == "claude-opus-5"


# ---------------------------------------------------------------- compare: jailbreak parity


def _jb_log_rows():
    def r(arm, item, layer, pos, h, status, rec=None):
        row = {"id": f"{arm}|{item}|L{layer}|p{pos}|{h}", "status": status, "attempts": 1}
        row["verdict"] = None if status != "ok" else {"any_recognition": rec, "labels": []}
        return row

    return [
        r("s3d-rl600", "it-a", 20, 100, "h1", "error"),
        r("s3d-rl600", "it-a", 20, 100, "h1", "ok", True),  # error then ok: ok counts
        r("s3d-rl600", "it-a", 44, 100, "h2", "ok", True),
        r("s3d-rl600", "it-a", 44, 100, "h2", "ok", False),  # two ok rows: LAST wins
        r("s3d-rl600", "it-b", 20, 100, "h3", "ok", False),
        r("s3d-rl600", "it-b", 44, 100, "h4", "ok", False),
        r("jlens", "it-b", 44, 100, "h5", "ok", True),  # other arm: dropped
        r("s3d-rl600", "it-zz", 44, 100, "h6", "ok", True),  # not in bank: dropped
        r("s3d-rl600", "it-c", 20, 100, "h7", "error"),  # never ok: absent
    ]


def test_jailbreak_log_dedupe_and_filters(tmp_path):
    p = _write_jsonl(tmp_path / "log.jsonl", _jb_log_rows())
    labels = js.load_jailbreak_log(p, arm="s3d-rl600", ids={"it-a", "it-b", "it-c"})
    assert labels == {
        ("it-a", 20, 100): True,
        ("it-a", 44, 100): False,
        ("it-b", 20, 100): False,
        ("it-b", 44, 100): False,
    }
    assert js.parse_jailbreak_id("arm|item|L20|p5|abc") == ("arm", "item", 20, 5)
    assert js.parse_jailbreak_id("arm|item|L20|p5") is None
    assert js.parse_jailbreak_id("arm|item|20|p5|abc") is None


def test_compare_jailbreak_parity_assert(tmp_path, monkeypatch):
    monkeypatch.setattr(js, "bank_ids", lambda family: {"it-a", "it-b", "it-c"})
    log = _write_jsonl(tmp_path / "log.jsonl", _jb_log_rows())
    rows = [
        {"id": "it-a", "layer": 20, "pos": 100, "any_recognition": True},
        {"id": "it-a", "layer": 44, "pos": 100, "any_recognition": True},
        {"id": "it-b", "layer": 20, "pos": 100, "any_recognition": False},
        {"id": "it-b", "layer": 44, "pos": 100, "any_recognition": False},
    ]
    results = _results("jailbreak_recognition", rows, value=1 / 3)
    good = _write_json(tmp_path / "eval.json", {"arms": {"s3d-rl600": {"pass_rate": 1 / 3}}})
    out = js.compare(
        "jailbreak_recognition", results, log, _ns(arm="s3d-rl600", baseline_eval=good)
    )
    assert out["parity"] is True and out["baseline_model"] == "claude-sonnet-5"
    assert out["n_cells_both"] == 4 and out["agreement"] == 0.75
    assert out["headline"]["baseline_published"] == pytest.approx(1 / 3)
    assert out["extras"]["n_bank_items"] == 3

    bad = _write_json(tmp_path / "bad.json", {"arms": {"s3d-rl600": {"pass_rate": 0.8256}}})
    with pytest.raises(ValueError, match="parity"):
        js.compare("jailbreak_recognition", results, log, _ns(arm="s3d-rl600", baseline_eval=bad))

    # without --baseline-eval nothing is asserted
    out2 = js.compare("jailbreak_recognition", results, log, _ns(arm="s3d-rl600"))
    assert out2["headline"]["baseline_published"] is None


# ---------------------------------------------------------------- compare: parity families


def test_compare_parity_families(tmp_path):
    hal = _write_json(
        tmp_path / "hal.json",
        {"model": "google/gemini-3.8-flash", "numbers": {"n_hallucinated": 30, "n_specific": 100}},
    )
    out = js.compare("hallucination", _results("hallucination", [], value=0.31), hal)
    assert out["parity"] is True and "n_cells_both" not in out
    assert out["headline"]["baseline"] == pytest.approx(0.3)
    assert out["headline"]["delta"] == pytest.approx(0.01)

    hal2 = _write_json(tmp_path / "hal2.json", {"numbers": {"hallucination_rate": 0.33}})
    assert js.compare("hallucination", _results("hallucination", [], 0.33), hal2)["headline"][
        "delta"
    ] == pytest.approx(0.0)

    jl = _write_json(
        tmp_path / "jl.json",
        {"arms": {"s3d": {"by_layer": {"44": {"precision": 0.3, "recall_at_10": 0.5}}}}},
    )
    out = js.compare(
        "jlens_concept_pr", _results("jlens_concept_pr", [], 0.32, {"recall_at_10": 0.51}), jl
    )
    assert out["headline"]["baseline"] == pytest.approx(0.3)
    assert out["extras"]["recall_at_10"] == 0.5 and out["extras"]["port_extras"] == {
        "recall_at_10": 0.51
    }

    am = _write_json(tmp_path / "am.json", {"headline": {"design_score": 0.57}})
    out = js.compare("agentic_misalignment", _results("agentic_misalignment", [], 0.6), am)
    assert out["baseline_model"] == "claude-sonnet-5"
    assert out["headline"]["delta"] == pytest.approx(0.03)


def test_port_counts_copied_into_extras(tmp_path):
    results = _results("relational_multihop", [], value=None, extras={"n_cells_api_failed": 2})
    results["counts"] = {"n_expected_cells": 600, "n_unjudged_cells": 2, "spend_usd": 1.0}
    base = _write_json(tmp_path / "b.json", {"verdicts": {}})
    out = js.compare("relational_multihop", results, base)
    pc = out["extras"]["port_counts"]
    assert pc == {
        "n_expected_cells": 600,
        "n_missing_cells": None,
        "n_unjudged_cells": 2,
        "n_empty_cells": None,
        "n_api_failed": 2,
    }


def test_compare_unknown_family_raises(tmp_path):
    with pytest.raises(KeyError):
        js.compare("nope", _results("nope", []), _write_json(tmp_path / "x.json", {}))


# ---------------------------------------------------------------- summary


def test_summary_flag_rule():
    good = {
        "family": "moral_rationale",
        "parity": False,
        "n_cells_both": 100,
        "agreement": 0.95,
        "cohen_kappa": 0.9,
        "headline": {"port": 0.8, "baseline": 0.78, "delta": 0.02},
        "baseline_model": "claude-opus-5",
    }
    flagged = {
        "family": "user_modeling",
        "parity": False,
        "n_cells_both": 88,
        "agreement": 0.7,
        "cohen_kappa": 0.4,
        "headline": {"port": 0.2, "baseline": 0.1, "delta": 0.1},
        "baseline_model": "claude-opus-5",
        "baseline_note": "conflates the judge change",
    }
    undefined = {
        "family": "role_bound_association",
        "parity": False,
        "n_cells_both": 8,
        "agreement": 1.0,
        "cohen_kappa": None,
        "headline": {"port": 1.0, "baseline": 1.0, "delta": 0.0},
        "baseline_model": "claude-opus-5",
    }
    parity = {
        "family": "hallucination",
        "parity": True,
        "headline": {"port": 0.335, "baseline": 0.335, "delta": 0.0},
        "baseline_model": "google/gemini-3.8-flash",
    }
    jb = {
        "family": "jailbreak_recognition",
        "parity": True,
        "n_cells_both": 4,
        "cohen_kappa": 0.5,
        "headline": {"port": 0.8, "baseline": 0.7, "baseline_published": 0.8256},
        "baseline_model": "claude-sonnet-5",
    }
    md = js.render_summary([flagged, good, undefined, parity, jb])
    lines = md.splitlines()
    assert lines[0].startswith("| family | n cells | cell agreement |")
    moral = next(line for line in lines if line.startswith("| moral_rationale"))
    assert moral.endswith("| ok |") and "0.900" in moral and "(claude-opus-5)" in moral
    um = next(line for line in lines if line.startswith("| user_modeling"))
    assert js.FLAG in um and "conflates the judge change" in um
    rb = next(line for line in lines if line.startswith("| role_bound_association"))
    assert js.FLAG in rb and "| — |" in rb  # undefined kappa is reported, not 0
    assert "Port parity" in md
    assert "| hallucination | google/gemini-3.8-flash | 0.335 | 0.335 | 0.000 |" in md
    assert "jailbreak_recognition (same-judge; cell κ = 0.500, n = 4)" in md
    assert js.verdict_of(None) == js.FLAG and js.verdict_of(0.7) == js.OK
    assert js.verdict_of(0.69) == js.FLAG


# ---------------------------------------------------------------- CLI round trip


ALLOWED_TOP_KEYS = {
    "family",
    "parity",
    "n_cells_both",
    "n_only_port",
    "n_only_baseline",
    "agreement",
    "cohen_kappa",
    "confusion",
    "confusion_extras",
    "n_port_failed",
    "headline",
    "item_level",
    "port_model",
    "baseline_model",
    "baseline_note",
    "extras",
}


def test_cli_compare_and_summary(tmp_path):
    marker_id, marker_text = "ITEM-ZETA-7731", "READOUT-TEXT-SENTINEL"
    rows = [
        {**r, "id": marker_id, "quote": marker_text, "samples": [marker_text]}
        for r in _moral_rows()[:2]
    ]
    res = _write_json(tmp_path / "results.json", _results("moral_rationale", rows, 0.6))
    base = _write_json(
        tmp_path / "b.json",
        {
            "verdicts": [
                {
                    "id": marker_id,
                    "layer": 20,
                    "pos": 7,
                    "side": "committed",
                    "correct": True,
                    "quote": marker_text,
                }
            ]
        },
    )
    out = tmp_path / "docs" / "moral_rationale.json"
    rc = js.main(
        [
            "compare",
            "--family",
            "moral_rationale",
            "--results",
            str(res),
            "--baseline",
            str(base),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    raw = out.read_text()
    d = json.loads(raw)
    assert d["n_cells_both"] == 1 and d["cohen_kappa"] is None
    assert set(d) <= ALLOWED_TOP_KEYS and "rows" not in d
    assert marker_id not in raw and marker_text not in raw  # no ids, no readout text
    md = tmp_path / "docs" / "summary.md"
    assert js.main(["summary", str(out), "--out", str(md)]) == 0
    assert js.FLAG in md.read_text()


def test_cli_convert_jailbreak(tmp_path, monkeypatch):
    monkeypatch.setattr(js, "bank_ids", lambda family: {"item-1"})
    src = _write_jsonl(
        tmp_path / "lens.jsonl",
        [{"id": "item-1", "arm": "s3d-rl600", "olens": {"20": {"5": ["a"]}}, "tokens": {"5": "t"}}],
    )
    out = tmp_path / "jb.jsonl"
    rc = js.main(
        [
            "convert",
            "--family",
            "jailbreak_recognition",
            "--src",
            str(src),
            "--out",
            str(out),
            "--ids-from-bank",
        ]
    )
    assert rc == 0
    cells, _ = load_readouts(out)
    assert len(cells) == 1 and cells[0].token == "t"


def test_bank_ids_reads_real_banks():
    ids = js.bank_ids("user_modeling")
    assert len(ids) == 100 and all(i.startswith("um-") for i in ids)
    assert len(js.bank_ids("jailbreak_recognition")) == 86
    assert "oa-police-thief-chase-ab" in js.bank_ids("role_bound_association")
