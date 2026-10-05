"""``wsbench run --all``: fail-soft per family, one preflight per model, concurrency."""

import json
import threading
from pathlib import Path

import pytest

from wsbench import llm, mcjudge, registry
from wsbench.cli import main
from wsbench.judge_config import JudgeConfig
from wsbench.llm import JudgeConfigError
from wsbench.registry import EvalSpec, JudgeArgs, register
from wsbench.results import FamilyResult

GEMINI = "google/gemini-3.8-flash"


def _ok(args: JudgeArgs, family: str = "ok") -> FamilyResult:
    return FamilyResult(
        family=family,
        metric="pass_rate",
        value=0.75,
        ci95=(0.5, 1.0),
        n_items=4,
        higher_is_better=True,
        chance=0.2,
        chance_label="1/5",
        complete=args.judge.pinned,
        pinned_instrument=args.judge.pinned,
        config={"judge_model": args.judge.model, "prompt_version": "v1"},
        counts={
            "n_expected_cells": 4,
            "n_missing_cells": 0,
            "n_unjudged_cells": 0,
            "n_empty_cells": 0,
            "skipped_rows": 0,
            "spend_usd": 1.5,
        },
        extras={"n_items_without_readouts": 1},
    )


def _spec(name: str, run, model: str = GEMINI) -> EvalSpec:
    return register(
        EvalSpec(
            name=name,
            title=name.title(),
            group="logic",
            bank=Path(f"evals/{name}/items.json"),
            judge=JudgeConfig(model=model),
            metric="pass_rate",
            higher_is_better=True,
            run=run,
        )
    )


@pytest.fixture
def offline(monkeypatch):
    """No network: ``llm.preflight`` records its calls; a client would fail the test."""
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.setattr(llm, "_make_client", lambda r, k: pytest.fail("no client in tests"))
    seen: list[tuple[str, dict | None]] = []
    monkeypatch.setattr(llm, "preflight", lambda model, reasoning: seen.append((model, reasoning)))
    registry.FAMILIES.clear()
    return seen


def _readouts(root: Path, *names: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    row = json.dumps({"id": "a", "layer": 36, "pos": 1, "samples": ["x"]}) + "\n"
    for n in names:
        (root / f"{n}.jsonl").write_text(row)


def test_fail_soft_statuses_run_json_and_summary(offline, tmp_path, capsys):
    def boom(args):
        raise JudgeConfigError("no key for claude")

    def nocells(args):
        raise SystemExit("no cells in scope")

    _spec("bad", boom)
    _spec("empty", nocells)
    _spec("good", _ok)
    _spec("absent", _ok)
    root, out = tmp_path / "r", tmp_path / "o"
    _readouts(root, "bad", "empty", "good")
    code = main(["run", "all=True", f"readouts_root={root}", f"out={out}"])
    assert code == 3  # a JudgeConfigError somewhere wins over the SystemExit
    printed = capsys.readouterr().out
    assert "absent: skipped (no readouts at" in printed
    assert (out / "good" / "results.json").exists()
    assert not (out / "bad" / "results.json").exists()

    summary = (out / "summary.md").read_text()
    assert "| ok |" in summary and "macro" in summary
    assert "## skipped / failed" in summary
    assert "- bad: failed — no key for claude" in summary
    assert "- empty: failed — no cells in scope" in summary
    assert "- absent: skipped — no readouts at" in summary

    run = json.loads((out / "run.json").read_text())
    assert set(run) == {"started", "finished", "families", "judge_overrides", "readouts_root"}
    assert run["judge_overrides"] == {"flag": None, "env": None}
    assert run["readouts_root"] == str(root)
    fams = run["families"]
    assert fams["bad"] == {"status": "failed", "error": "no key for claude"}
    assert fams["empty"] == {"status": "failed", "error": "no cells in scope"}
    assert fams["absent"]["status"] == "skipped" and "no readouts" in fams["absent"]["error"]
    good = fams["good"]
    assert good["status"] == "ok" and good["results_path"] == str(out / "good" / "results.json")
    assert good["value"] == 0.75 and good["complete"] is True and good["pinned_instrument"] is True
    assert good["spend_usd"] == 1.5 and "n_calls" not in good and "error" not in good


def test_system_exit_only_is_exit_2_and_int_codes_render(offline, tmp_path):
    def nocells(args):
        raise SystemExit(2)

    _spec("empty", nocells)
    _spec("good", _ok)
    root, out = tmp_path / "r", tmp_path / "o"
    _readouts(root, "empty", "good")
    assert main(["run", "all=True", f"readouts_root={root}", f"out={out}"]) == 2
    run = json.loads((out / "run.json").read_text())
    assert run["families"]["empty"] == {"status": "failed", "error": "exit 2"}
    assert main(["run", "families=good", f"readouts_root={root}", f"out={out}"]) == 0


def test_bad_opt_is_one_error_before_any_family_runs(offline, tmp_path):
    ran: list[str] = []

    def run(args):
        ran.append("x")
        return _ok(args)

    _spec("good", run)
    root, out = tmp_path / "r", tmp_path / "o"
    _readouts(root, "good")
    with pytest.raises(SystemExit):
        main(["run", "all=True", f"readouts_root={root}", f"out={out}", "opts=novalue"])
    assert ran == [] and offline == [] and not (out / "run.json").exists()


def test_preflight_once_per_distinct_model_and_seeds_the_shared_set(offline, tmp_path):
    inner: list[str] = []

    def run(args):
        # a family's own Preflighter finds the model already preflighted -> no second call
        mcjudge.Preflighter(args.dry_run).ensure(args.judge)
        inner.append(args.judge.model)
        return _ok(args)

    _spec("g1", run)
    _spec("g2", run)
    _spec("c1", run, model="claude-sonnet-5")
    _spec("nofile", run, model="claude-other")  # skipped -> never preflighted
    root, out = tmp_path / "r", tmp_path / "o"
    _readouts(root, "g1", "g2", "c1")
    assert main(["run", "all=True", f"readouts_root={root}", f"out={out}"]) == 0
    assert sorted(offline) == [("claude-sonnet-5", None), (GEMINI, {"effort": "minimal"})]
    assert len(inner) == 3
    assert mcjudge.preflight_key(GEMINI, {"effort": "minimal"}) in mcjudge._PREFLIGHTED


def test_preflight_failure_aborts_before_any_thread(offline, tmp_path, monkeypatch, capsys):
    def preflight(model, reasoning):
        raise JudgeConfigError("ANTHROPIC_API_KEY is missing")

    monkeypatch.setattr(llm, "preflight", preflight)
    ran: list[str] = []

    def run(args):
        ran.append(args.judge.model)
        return _ok(args)

    _spec("c1", run, model="claude-sonnet-5")
    root, out = tmp_path / "r", tmp_path / "o"
    _readouts(root, "c1")
    assert main(["run", "all=True", f"readouts_root={root}", f"out={out}"]) == 3
    assert ran == []
    assert "ANTHROPIC_API_KEY is missing" in capsys.readouterr().err


def test_judge_override_flag_is_recorded_and_resolved_once(offline, tmp_path, monkeypatch):
    _spec("g1", _ok)
    monkeypatch.setenv("WSBENCH_JUDGE_MODEL", "env-model")
    root, out = tmp_path / "r", tmp_path / "o"
    _readouts(root, "g1")
    argv = ["run", "all=True", f"readouts_root={root}", f"out={out}", "judge_model=x/y"]
    assert main(argv) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["judge_overrides"] == {"flag": "x/y", "env": "env-model"}
    assert run["families"]["g1"]["pinned_instrument"] is False
    assert offline == [("x/y", {"effort": "minimal"})]


def test_dry_run_never_preflights_and_runs_one_at_a_time(offline, tmp_path):
    lock = threading.Lock()
    active = {"n": 0, "max": 0}

    def run(args):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        threading.Event().wait(0.02)
        with lock:
            active["n"] -= 1
        return _ok(args)

    for n in ("a", "b", "c"):
        _spec(n, run)
    root, out = tmp_path / "r", tmp_path / "o"
    _readouts(root, "a", "b", "c")
    argv = ["run", "all=True", f"readouts_root={root}", f"out={out}", "dry_run=True"]
    assert main([*argv, "family_workers=3"]) == 0
    assert offline == [] and active["max"] == 1


def test_families_run_concurrently_under_family_workers(offline, tmp_path):
    barrier = threading.Barrier(3, timeout=5)

    def run(args):
        barrier.wait()  # BrokenBarrierError (timeout) propagates and fails the test
        return _ok(args)

    for n in ("a", "b", "c"):
        _spec(n, run)
    root, out = tmp_path / "r", tmp_path / "o"
    _readouts(root, "a", "b", "c")
    argv = ["run", "all=True", f"readouts_root={root}", f"out={out}"]
    assert main([*argv, "family_workers=3"]) == 0
    run_json = json.loads((out / "run.json").read_text())
    assert {v["status"] for v in run_json["families"].values()} == {"ok"}


def test_run_json_flag_prints_json(offline, tmp_path, capsys):
    _spec("g1", _ok)
    root, out = tmp_path / "r", tmp_path / "o"
    _readouts(root, "g1", "zz")
    argv = ["run", "all=True", f"readouts_root={root}", f"out={out}", "json=True"]
    assert main(argv) == 0
    d = json.loads(capsys.readouterr().out.strip().splitlines()[-1])  # one JSON line, last
    assert set(d) >= {"families", "macro", "statuses"}
    assert d["statuses"]["g1"]["status"] == "ok"
    assert (out / "summary.md").exists() and (out / "run.json").exists()


def test_keyboard_interrupt_writes_manifest_and_exits_130(offline, tmp_path, monkeypatch):
    from wsbench import runner

    _spec("g1", _ok)
    _spec("g2", _ok)
    root, out = tmp_path / "r", tmp_path / "o"
    _readouts(root, "g1", "g2")

    class Interrupting(runner.ThreadPoolExecutor):
        def submit(self, fn, *a, **kw):
            if getattr(self, "_n", 0) == 1:
                raise KeyboardInterrupt
            self._n = getattr(self, "_n", 0) + 1
            return super().submit(fn, *a, **kw)

    monkeypatch.setattr(runner, "ThreadPoolExecutor", Interrupting)
    assert main(["run", "all=True", f"readouts_root={root}", f"out={out}"]) == 130
    run_json = json.loads((out / "run.json").read_text())
    st = {k: v["status"] for k, v in run_json["families"].items()}
    assert st["g1"] == "ok" and st["g2"] == "skipped"
    assert run_json["families"]["g2"]["error"] == "interrupted"
    assert "## skipped / failed" in (out / "summary.md").read_text()
