"""poetry on the shared basic-family code."""

from pathlib import Path

from wsbench import registry
from wsbench.banks import item_targets, load_bank
from wsbench.basic.prompts import PROMPT_VERSION, SYSTEM
from wsbench.cli import main
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples/readouts/poetry.jsonl"


def test_bank_and_targets():
    header, items = load_bank(REPO / "evals/poetry/items.json")
    assert header["family"] == "poetry" and len(items) == 100
    first = next(it for it in items if it["name"] == "couplet-ahead-head")
    assert item_targets(first) == ["led"]


def test_registered_spec_and_readme():
    from wsbench.evals.poetry import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)
    assert SPEC.group == "basic" and SPEC.judge.prompt_version == PROMPT_VERSION
    readme = (REPO / "evals/poetry/README.md").read_text(encoding="utf-8")
    assert SYSTEM in readme and PROMPT_VERSION in readme


def test_dry_run_on_the_toy_file(tmp_path, capsys, monkeypatch):
    from wsbench.evals.poetry import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    out = tmp_path / "out"
    argv = ["judge", "family=poetry", f"readouts={EXAMPLE}", f"out={out}", "dry_run=True"]
    assert main(argv) == 0
    r = read_results(out)
    assert r.value is None and r.extras["n_items_without_readouts"] == 98
    assert "judge prompt for" in capsys.readouterr().out
