import dataclasses
import json
from pathlib import Path

import pytest

from wsbench import llm, registry
from wsbench.cli import main
from wsbench.judge_config import JudgeConfig
from wsbench.readouts import load_readouts
from wsbench.registry import EvalSpec, JudgeArgs, register
from wsbench.results import FamilyResult, completeness, read_results

GEMINI = "google/gemini-3.8-flash"


def _stub_run(args: JudgeArgs) -> FamilyResult:
    cells, rep = load_readouts(args.readouts, layers=args.layers)
    subset = args.items is not None or args.limit > 0
    n = len(cells)
    return FamilyResult(
        family="stub",
        metric="pass_rate",
        value=1.0 if n else None,
        ci95=(1.0, 1.0) if n else None,
        n_items=n,
        higher_is_better=True,
        chance=0.5,
        chance_label="coin",
        complete=completeness(
            pinned=args.judge.pinned,
            subset=subset,
            n_expected=n,
            n_missing=0,
            n_unjudged=0,
            n_empty=rep.n_empty,
        ),
        pinned_instrument=args.judge.pinned,
        config={
            "judge_model": args.judge.model,
            "prompt_version": "v1",
            "reasoning": args.judge.reasoning,
            "layers": args.layers,
            "dry_run": args.dry_run,
            "concurrency": args.concurrency,
            "rpm": args.rpm,
            "allow_missing": args.allow_missing,
            "items": args.items,
            "limit": args.limit,
        },
        counts={
            "n_expected_cells": n,
            "n_missing_cells": 0,
            "n_unjudged_cells": 0,
            "n_empty_cells": rep.n_empty,
            "skipped_rows": sum(rep.skipped.values()),
            "spend_usd": 0.0,
        },
        rows=[{"key": c.key} for c in cells],
    )


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    # ``run`` preflights every judge model up front (phase 6); tests make no network calls
    monkeypatch.setattr(llm, "preflight", lambda model, reasoning: None)
    return register(
        EvalSpec(
            name="stub",
            title="Stub",
            group="logic",
            bank=Path("evals/stub/items.json"),
            judge=JudgeConfig(model=GEMINI),
            metric="pass_rate",
            higher_is_better=True,
            run=_stub_run,
        )
    )


@pytest.fixture
def readouts(tmp_path):
    p = tmp_path / "stub.jsonl"
    rows = [
        {"id": "a", "layer": 36, "pos": 1, "samples": ["x"]},
        {"id": "b", "layer": 36, "pos": 1, "samples": ["y"]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_list_empty_registry(capsys):
    registry.FAMILIES.clear()  # the real families are already imported; load_all re-adds none
    assert main(["list"]) == 0
    assert "no families registered yet" in capsys.readouterr().out


def test_list_with_stub(stub, capsys, tmp_path, monkeypatch):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "stub" in out and "logic" in out and GEMINI in out and "v1" in out and "?" in out


def test_judge_unknown_exit_2(stub, capsys):
    assert main(["judge", "family=nope", "readouts=x", "out=y"]) == 2
    err = capsys.readouterr().err
    assert "nope" in err and "stub" in err


def test_judge_writes_results_and_pinned(stub, readouts, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["judge", "family=stub", f"readouts={readouts}", f"out={out}"]) == 0
    r = read_results(out)
    assert r.family == "stub" and r.pinned_instrument and r.complete and r.n_items == 2
    assert r.config["judge_model"] == GEMINI and r.config["reasoning"] == {"effort": "minimal"}
    assert (
        r.config["dry_run"] is False and r.config["concurrency"] == 64 and r.config["rpm"] == 240.0
    )
    printed = capsys.readouterr().out
    assert "1.0" in printed

    assert (
        main(
            [
                "judge",
                "family=stub",
                f"readouts={readouts}",
                f"out={out}",
                "judge_model=claude-x",
            ]
        )
        == 0
    )
    r = read_results(out)
    assert r.pinned_instrument is False and r.complete is False
    assert r.config["judge_model"] == "claude-x" and r.config["reasoning"] is None


def test_judge_flags_pass_through(stub, readouts, tmp_path):
    out = tmp_path / "out"
    argv = [
        "judge",
        "family=stub",
        f"readouts={readouts}",
        f"out={out}",
        "layers=20,36",
        "items=a,b",
        "limit=5",
        "allow_missing=True",
        "concurrency=3",
        "rpm=10",
        "dry_run=True",
    ]
    assert main(argv) == 0
    r = read_results(out)
    c = r.config
    assert c["layers"] == [20, 36] and c["items"] == ["a", "b"] and c["limit"] == 5
    assert (
        c["allow_missing"] is True
        and c["concurrency"] == 3
        and c["rpm"] == 10.0
        and c["dry_run"] is True
    )
    assert r.complete is False  # subset flags


def test_judge_default_out(stub, readouts, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["judge", "family=stub", f"readouts={readouts}"]) == 0
    assert (tmp_path / "outputs" / "stub" / "stub" / "results.json").exists()


def test_judge_config_error_exit_3(stub, readouts, tmp_path, monkeypatch):
    from wsbench.llm import JudgeConfigError

    def boom(args):
        raise JudgeConfigError("no key")

    monkeypatch.setitem(registry.FAMILIES, "stub", dataclasses.replace(stub, run=boom))
    assert main(["judge", "family=stub", f"readouts={readouts}", f"out={tmp_path / 'o'}"]) == 3


def test_run_and_report(stub, readouts, tmp_path, capsys):
    other = register(
        EvalSpec(
            name="other",
            title="Other",
            group="safety",
            bank=Path("evals/other/items.json"),
            judge=JudgeConfig(),
            metric="pass_rate",
            higher_is_better=True,
            run=_stub_run,
        )
    )
    assert other.name == "other"
    root = readouts.parent
    out = tmp_path / "run_out"
    assert main(["run", "all=True", f"readouts_root={root}", f"out={out}"]) == 0
    printed = capsys.readouterr().out
    assert "other" in printed and "skip" in printed.lower()
    assert (out / "stub" / "results.json").exists()
    assert not (out / "other").exists()
    summary = (out / "summary.md").read_text()
    assert "| stub |" in summary and "macro" in summary

    assert main(["run", "families=stub", f"readouts_root={root}", f"out={out}"]) == 0
    assert main(["run", "families=nope", f"readouts_root={root}", f"out={out}"]) == 2

    (out / "summary.md").unlink()
    assert main(["report", f"dir={out}"]) == 0
    printed = capsys.readouterr().out
    assert "| stub |" in printed and "macro" in printed
    assert (out / "summary.md").exists()


def test_run_requires_exactly_one_selection(stub, tmp_path, capsys):
    assert main(["run", f"readouts_root={tmp_path}", f"out={tmp_path / 'o'}"]) == 2
    assert "exactly one" in capsys.readouterr().err
    argv = ["run", "all=True", "families=stub", f"readouts_root={tmp_path}", f"out={tmp_path}"]
    assert main(argv) == 2


def test_convert_gen_dir(tmp_path, capsys):
    g = tmp_path / "gen" / "item1"
    g.mkdir(parents=True)
    (g / "L036.jsonl").write_text(json.dumps({"pos": 2, "samples": ["s"], "scores": [1.0]}) + "\n")
    (g / "L020.jsonl").write_text(json.dumps({"pos": 2, "samples": ["t"]}) + "\n")
    out = tmp_path / "c.jsonl"
    assert (
        main(
            [
                "convert-gen-dir",
                f"gen_dir={tmp_path / 'gen'}",
                f"out={out}",
                "kind=tokens",
                "layers=36",
            ]
        )
        == 0
    )
    cells, rep = load_readouts(out)
    assert (
        len(cells) == 1
        and cells[0].tokens == ("s",)
        and cells[0].scores == (1.0,)
        and cells[0].layer == 36
    )
    assert "tokens" in capsys.readouterr().out
    assert main(["convert-gen-dir", f"gen_dir={tmp_path / 'gen'}", f"out={out}", "kind=prose"]) == 0
    cells, rep = load_readouts(out)
    assert len(cells) == 2 and rep.kind == "prose"


def test_opt_and_aux_models_reach_the_family(readouts, tmp_path, monkeypatch):
    seen: dict = {}

    def run(args: JudgeArgs) -> FamilyResult:
        seen["extra"] = args.extra
        seen["aux"] = dict(args.aux_models)
        return _stub_run(args)

    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    register(
        EvalSpec(
            name="stub2",
            title="Stub2",
            group="logic",
            bank=Path("evals/stub/items.json"),
            judge=JudgeConfig(aux_models={"summarizer": "m/s"}),
            metric="pass_rate",
            higher_is_better=True,
            run=run,
        )
    )
    out = tmp_path / "o"
    argv = ["judge", "family=stub2", f"readouts={readouts}", f"out={out}"]
    assert main([*argv, "opts=char_cap=5,x=a=b"]) == 0
    assert seen == {"extra": {"char_cap": "5", "x": "a=b"}, "aux": {"summarizer": "m/s"}}
    assert main(argv) == 0 and seen["extra"] == {}
    with pytest.raises(SystemExit):
        main([*argv, "opts=novalue"])


def test_no_command_and_unknown_command_exit_2(capsys):
    assert main([]) == 2
    assert "usage: wsbench" in capsys.readouterr().err
    assert main(["nope"]) == 2
    assert "usage: wsbench" in capsys.readouterr().err


def test_missing_required_and_unknown_key_exit_2(stub, capsys):
    assert main(["judge", "family=stub"]) == 2
    assert "readouts" in capsys.readouterr().err
    assert main(["judge", "family=stub", "readouts=x", "bogus=1"]) == 2
    assert "bogus" in capsys.readouterr().err


def test_show_and_help(stub, capsys):
    argv = ["judge", "family=stub", "readouts=x", "layers=20,36", "dry_run=false", "--show"]
    assert main(argv) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["family"] == "stub" and shown["layers"] == [20, 36] and shown["limit"] == 0
    assert shown["items"] is None and shown["dry_run"] is False and shown["opt"] == []
    assert main(["--help"]) == 0
    assert "usage: wsbench" in capsys.readouterr().out
    assert main(["run", "-h"]) == 0
    out = capsys.readouterr().out
    assert "readouts_root=<required>" in out and "family_workers=3" in out


@pytest.mark.parametrize(
    "override",
    [
        "dry_run=no",
        "allow_missing=off",
        "layers=True",
        "layers=,",
        "limit=1.9",
        "limit=None",
        "rpm=None",
        "readouts=None",
        "items=True",
    ],
)
def test_bad_values_exit_2(stub, capsys, override):
    argv = ["judge", "family=stub", "readouts=x", "out=y", override]
    if override.startswith("readouts="):
        argv.remove("readouts=x")
    assert main(argv) == 2
    assert "judge:" in capsys.readouterr().err


def test_string_bools_and_numeric_items(stub, readouts, tmp_path):
    out = tmp_path / "out"
    argv = ["judge", "family=stub", f"readouts={readouts}", f"out={out}", "dry_run=false"]
    assert main([*argv, "items=1"]) == 0
    c = read_results(out).config
    assert c["dry_run"] is False and c["items"] == ["1"]
    assert main([*argv[:-1], "dry_run=true", "items=1,2", "layers=36"]) == 0
    c = read_results(out).config
    assert c["dry_run"] is True and c["items"] == ["1", "2"] and c["layers"] == [36]
