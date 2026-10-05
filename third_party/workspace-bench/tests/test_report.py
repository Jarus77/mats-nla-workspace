"""``markdown_table`` formatting and ``wsbench report --json``."""

import json

from wsbench.cli import main
from wsbench.results import FamilyResult, macro, markdown_table, write_results


def _res(family="fam", metric="pass_rate", value=0.5, complete=True, **kw) -> FamilyResult:
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


def test_lower_is_better_and_no_readouts_cells():
    rs = [
        _res("hal", metric="hallucination_rate", higher_is_better=False),
        _res("um", extras={"n_items_without_readouts": 3}),
        _res("zero", extras={"n_items_without_readouts": 0}),
    ]
    md = markdown_table(rs, None)
    rows = {ln.split("|")[1].strip(): ln for ln in md.splitlines() if ln.startswith("|")}
    assert "| hallucination_rate (lower is better) |" in rows["hal"]
    assert "| 10 (3 no readouts) |" in rows["um"]
    assert "| 10 |" in rows["zero"] and "no readouts" not in rows["zero"]
    assert "| pass_rate |" in rows["um"]
    assert md.endswith("\n")


def test_macro_row_and_not_in_macro_footnotes():
    rs = [
        _res("a"),
        _res("hal", metric="hallucination_rate", higher_is_better=False),
        _res("b", complete=False),
        _res("c", value=None),
    ]
    m = macro(rs)
    md = markdown_table(rs, m)
    table = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert len(table) == 2 + 4 + 1
    assert "excluded: hal (metric), b (incomplete), c (no value)" in table[-1]
    assert "| 1 families |" in table[-1]
    foot = md.split(table[-1], 1)[1]
    assert "Not in macro:" in foot
    assert "- hal: metric (hallucination_rate)" in foot
    assert "- b: incomplete" in foot and "- c: no value" in foot
    # no exclusions -> no footnote; no macro row -> no footnote
    assert "Not in macro" not in markdown_table([_res("a")], macro([_res("a")]))
    assert "Not in macro" not in markdown_table(rs, None)


def test_report_json(tmp_path, capsys):
    write_results(tmp_path / "a", _res("a"))
    write_results(tmp_path / "hal", _res("hal", metric="hallucination_rate", value=0.2))
    assert main(["report", f"dir={tmp_path}", "json=True"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert set(d) == {"families", "macro", "floors"}
    assert [f["family"] for f in d["families"]] == ["a", "hal"]
    assert d["families"][0]["numbers"]["metric"] == "pass_rate"
    assert d["macro"]["value"] == 0.5 and d["macro"]["families"] == ["a"]
    assert d["macro"]["excluded"] == [{"family": "hal", "reason": "metric"}]
    summary = (tmp_path / "summary.md").read_text()
    assert "| a |" in summary and "- hal: metric (hallucination_rate)" in summary

    assert main(["report", f"dir={tmp_path}"]) == 0
    out = capsys.readouterr().out
    assert "| a |" in out and "Not in macro:" in out and "wrote" in out
