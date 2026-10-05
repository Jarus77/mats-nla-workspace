import json

import pytest

from wsbench.results import (
    SCHEMA_VERSION,
    FamilyResult,
    bootstrap_ci,
    completeness,
    macro,
    markdown_table,
    read_results,
    write_results,
)


def _res(family="fam", metric="pass_rate", value=0.5, complete=True, **kw):
    base = {
        "family": family,
        "metric": metric,
        "value": value,
        "ci95": (0.4, 0.6) if value is not None else None,
        "n_items": 10,
        "higher_is_better": True,
        "chance": 0.1,
        "chance_label": "1/10",
        "complete": complete,
        "pinned_instrument": True,
        "config": {"judge_model": "google/gemini-3.8-flash", "prompt_version": "v1"},
        "counts": {
            "n_expected_cells": 10,
            "n_missing_cells": 0,
            "n_unjudged_cells": 0,
            "n_empty_cells": 0,
            "skipped_rows": 0,
            "spend_usd": 0.0,
        },
    }
    base.update(kw)
    return FamilyResult(**base)


def test_to_json_exact_keys_and_round_trip(tmp_path):
    r = _res(extras={"k": 1}, rows=[{"cell": "a"}])
    d = r.to_json()
    assert set(d) == {
        "schema_version",
        "family",
        "complete",
        "pinned_instrument",
        "config",
        "n_items",
        "counts",
        "numbers",
        "rows",
    }
    assert d["schema_version"] == SCHEMA_VERSION
    assert set(d["counts"]) == {
        "n_expected_cells",
        "n_missing_cells",
        "n_unjudged_cells",
        "n_empty_cells",
        "skipped_rows",
        "spend_usd",
    }
    assert set(d["numbers"]) == {
        "metric",
        "value",
        "ci95",
        "chance",
        "chance_label",
        "higher_is_better",
        "extras",
    }
    back = FamilyResult.from_json(json.loads(json.dumps(d)))
    assert back == r
    assert isinstance(back.ci95, tuple)
    p = write_results(tmp_path, r)
    assert p == tmp_path / "results.json"
    assert read_results(tmp_path) == r


def test_from_json_rejects_other_schema():
    d = _res().to_json()
    d["schema_version"] = 99
    with pytest.raises(ValueError):
        FamilyResult.from_json(d)


def test_bootstrap_ci():
    vals = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    a = bootstrap_ci(vals, seed=0)
    assert a == bootstrap_ci(vals, seed=0)
    assert a != bootstrap_ci(vals, seed=1)
    assert 0.0 <= a[0] <= 0.625 <= a[1] <= 1.0
    assert bootstrap_ci([0.3]) == (0.3, 0.3)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_completeness_each_clause():
    ok = {
        "pinned": True,
        "subset": False,
        "n_expected": 100,
        "n_missing": 0,
        "n_unjudged": 0,
        "n_empty": 5,
    }
    assert completeness(**ok) is True
    assert completeness(**{**ok, "pinned": False}) is False
    assert completeness(**{**ok, "subset": True}) is False
    assert completeness(**{**ok, "n_missing": 1}) is False
    assert completeness(**{**ok, "n_unjudged": 1}) is False
    assert completeness(**{**ok, "n_empty": 6}) is False


def test_macro_exclusions():
    rs = [
        _res("a", value=0.2),
        _res("b", value=0.4),
        _res("c", complete=False),
        _res("d", metric="precision"),
        _res("e", value=None),
    ]
    m = macro(rs)
    assert m["value"] == pytest.approx(0.3)
    assert m["families"] == ["a", "b"]
    assert m["excluded"] == [
        {"family": "c", "reason": "incomplete"},
        {"family": "d", "reason": "metric"},
        {"family": "e", "reason": "no value"},
    ]
    assert macro([])["value"] is None


def test_markdown_table():
    rs = [_res("a"), _res("b", value=None, complete=False)]
    md = markdown_table(rs, macro(rs))
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert lines[0].startswith(
        "| family | metric | value | 95% CI | n | chance | judge | pinned | complete |"
    )
    assert len(lines) == 2 + 3  # header + separator + 2 results + macro
    assert lines[2].startswith("| a | pass_rate | 0.500 |")
    assert "google/gemini-3.8-flash" in lines[2]
    assert "macro" in lines[-1]
    assert len([ln for ln in markdown_table(rs, None).splitlines() if ln.startswith("|")]) == 4
