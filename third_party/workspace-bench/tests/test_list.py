"""``wsbench list`` renders metric, calls/arm and credit for every registered family."""

from pathlib import Path

from wsbench import registry
from wsbench.cli import main
from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

HEAD = ("family", "group", "n items", "metric", "judge model", "prompt version")
HEAD += ("calls/arm (approx)", "credit")


def test_list_columns_for_every_family(capsys):
    registry.load_all()
    specs = sorted(registry.FAMILIES.values(), key=lambda s: s.name)
    assert len(specs) >= 9
    assert main(["list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    for h in HEAD:
        assert h in lines[0], h
    assert lines[0].index("metric") < lines[0].index("judge model") < lines[0].index("credit")
    assert len(lines) == 1 + len(specs)
    for s, ln in zip(specs, lines[1:], strict=True):
        assert ln.startswith(s.name)
        assert s.metric in ln and s.calls_per_arm and s.calls_per_arm in ln, s.name
        assert (s.sources or "—") in ln, s.name
    in_house = sum(1 for s in specs if not s.sources)
    assert sum("—" in ln for ln in lines[1:]) == in_house


def test_list_defaults_render_placeholders(capsys, monkeypatch):
    registry.FAMILIES.clear()
    register(
        EvalSpec(
            name="stub",
            title="Stub",
            group="logic",
            bank=Path("evals/stub/items.json"),
            judge=JudgeConfig(),
            metric="pass_rate",
            higher_is_better=True,
            run=lambda a: None,
        )
    )
    assert main(["list"]) == 0
    row = capsys.readouterr().out.splitlines()[1]
    assert row.startswith("stub") and "pass_rate" in row and row.count("?") == 2 and "—" in row
