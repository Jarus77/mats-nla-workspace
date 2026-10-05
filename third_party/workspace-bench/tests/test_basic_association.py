"""The association family end to end, offline: bank, judge with a fake client, cache, rules."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from wsbench import llm, registry
from wsbench.banks import exact_targets, item_targets, label_of, load_bank
from wsbench.basic.judge import quote_verified
from wsbench.basic.prompts import PROMPT_VERSION, SYSTEM
from wsbench.cli import main
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples/readouts/association.jsonl"


def _reply(expressed: bool, target: str = "", quote: str = "", cost: float = 0.001) -> Any:
    content = json.dumps({"expressed": expressed, "target": target, "quote": quote})
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(cost=cost),
    )


class FakeJudge:
    """Answers YES with a quote when a trigger phrase is in the user prompt, else NO."""

    def __init__(
        self,
        phrases: dict[str, str],
        fail_on: str | None = None,
        quotes: dict[str, str] | None = None,
    ):
        self.phrases = (
            phrases  # trigger phrase -> target; the quote is the trigger unless overridden
        )
        self.quotes = quotes or {}
        self.fail_on = fail_on
        self.calls: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw: Any) -> Any:
        user = kw["messages"][1]["content"]
        self.calls.append(user)
        if self.fail_on and self.fail_on in user:
            raise RuntimeError("boom")
        for trigger, target in self.phrases.items():
            if trigger in user:
                return _reply(True, target, self.quotes.get(trigger, trigger))
        return _reply(False)

    async def close(self) -> None:
        return None


@pytest.fixture
def family(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.setattr(llm, "_backoff", lambda attempt, status: 0.0)

    async def no_pace(rpm: float) -> None:
        return None

    monkeypatch.setattr(llm, "_pace", no_pace)
    monkeypatch.setattr(llm, "preflight", lambda model, reasoning: None)
    from wsbench.evals.association import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)
    return SPEC


def _install(monkeypatch, fake: FakeJudge) -> FakeJudge:
    monkeypatch.setattr(llm, "_make_client", lambda route, key: fake)
    return fake


def rows_by_id(r) -> dict[str, dict]:
    return {row["id"]: row for row in r.rows}


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
    return path


def test_bank_targets_and_labels(tmp_path):
    header, items = load_bank(REPO / "evals/association/items.json")
    assert header["family"] == "association" and len(items) == 100
    assert item_targets(items[0]) == ["carnaval"] and items[0]["id"] == items[0]["name"]
    assert item_targets({"name": "x", "target": "Fe", "intermediates": ["iron"]}) == ["iron"]
    assert item_targets({"name": "x", "target": "Fe"}) == ["Fe"]
    units = [{"headline": True, "match": ["a"]}, {"match": ["b"]}]
    assert item_targets({"name": "x", "units": units}) == ["a"]
    assert exact_targets(["moon", "full moon"]) == ["full moon"]
    assert label_of("br-proc-arith-2*3") == "br-proc-arith-2_3" and label_of("chess") == "chess"
    clash = tmp_path / "bank.json"
    clash.write_text(json.dumps({"family": "x", "items": [{"name": "a*b"}, {"name": "a_b"}]}))
    with pytest.raises(ValueError, match="share label"):
        load_bank(clash)


def test_registered_spec(family):
    assert family.group == "basic" and family.metric == "pass_rate"
    assert family.judge.model == "google/gemini-3.8-flash"
    assert family.judge.prompt_version == PROMPT_VERSION and family.calls_per_arm == "≈ 1.1k"


def test_readme_carries_the_verbatim_prompt():
    readme = (REPO / "evals/association/README.md").read_text(encoding="utf-8")
    assert SYSTEM in readme and PROMPT_VERSION in readme


def test_quote_gate_is_per_sample():
    samples = ["The sport of **soccer**", "She’s not lying"]  # noqa: RUF001
    assert quote_verified("soccer", samples)
    assert quote_verified("the sport of soccer", samples)
    assert quote_verified("She\\u2019s not lying", samples)
    assert not quote_verified("", samples)
    assert not quote_verified("tennis", samples)
    assert not quote_verified("new delhi", ["new", "delhi"])  # never fused across samples
    assert quote_verified("new delhi", ["new delhi"])


def test_dry_run_makes_no_calls_and_scores_nothing(family, tmp_path, monkeypatch, capsys):
    fake = _install(monkeypatch, FakeJudge({}))
    out = tmp_path / "out"
    argv = ["judge", "family=association", f"readouts={EXAMPLE}", f"out={out}", "dry_run=True"]
    assert main(argv) == 0
    assert fake.calls == []
    r = read_results(out)
    assert r.value is None and r.complete is False and r.n_items == 100
    assert r.counts["n_missing_cells"] == 192  # the toy file covers 4 of 100 items
    assert r.extras["n_items_without_readouts"] == 96
    assert "judge prompt for" in capsys.readouterr().out


def test_missing_cells_exit_2_without_allow_missing(family, tmp_path, monkeypatch, capsys):
    _install(monkeypatch, FakeJudge({}))
    out = tmp_path / "out"
    argv = ["judge", "family=association", f"readouts={EXAMPLE}", f"out={out}"]
    with pytest.raises(SystemExit) as ei:
        main(argv)
    assert ei.value.code == 2 and "allow_missing" in capsys.readouterr().err
    assert main([*argv, "allow_missing=True"]) == 0
    r = read_results(out)
    assert r.complete is False and r.extras["n_items_undecided"] == 96


def test_empty_readouts_are_never_complete(family, tmp_path, monkeypatch):
    _install(monkeypatch, FakeJudge({}))
    empty = _write(tmp_path / "empty.jsonl", [])
    out = tmp_path / "out"
    argv = ["judge", "family=association", f"readouts={empty}", f"out={out}"]
    assert main(argv) == 0
    r = read_results(out)
    assert r.value is None and r.complete is False and r.counts["n_expected_cells"] == 0
    assert r.extras["n_items_without_readouts"] == 100


def test_judge_verdicts_and_cells(family, tmp_path, monkeypatch):
    fake = _install(monkeypatch, FakeJudge({"Carnival in Rio": "carnaval", "chess": "chess"}))
    out = tmp_path / "out"
    argv = [
        "judge",
        "family=association",
        f"readouts={EXAMPLE}",
        f"out={out}",
        "items=pt-carnaval,chess,bf-paris,nasa",
    ]
    assert main(argv) == 0
    r = read_results(out)
    assert r.n_items == 4 and r.value == 0.5 and r.pinned_instrument
    assert r.complete is False  # items= subset
    assert r.counts["n_expected_cells"] == 8 and r.counts["n_unjudged_cells"] == 0
    assert r.counts["n_empty_cells"] == 1 and r.counts["spend_usd"] > 0
    assert len(fake.calls) == 7 and all(c.startswith("targets: ") for c in fake.calls)
    rows = rows_by_id(r)
    assert rows["pt-carnaval"]["pass"] and rows["pt-carnaval"]["earliest_layer"] == 20
    assert rows["chess"]["pass"] and rows["chess"]["layers"]["20"]["quote"] == "chess"
    assert rows["bf-paris"]["pass"] is False and rows["nasa"]["pass"] is False
    assert rows["nasa"]["layers"]["20"] == {
        "expressed": False,
        "target": "",
        "quote": "",
        "quote_ok": False,
    }
    assert r.extras["n_calls"] == 7 and r.extras["n_items_decided"] == 4


def test_unverifiable_quote_voids_a_positive(family, tmp_path, monkeypatch):
    fake = FakeJudge({"Two players": "chess"}, quotes={"Two players": "grandmaster"})
    _install(monkeypatch, fake)  # YES on this cell, quoting a word the sample does not contain
    out = tmp_path / "out"
    argv = ["judge", "family=association", f"readouts={EXAMPLE}", f"out={out}", "items=chess"]
    assert main(argv) == 0
    r = read_results(out)
    layer = r.rows[0]["layers"]["36"]
    assert r.value == 0.0 and layer["quote_ok"] is False and layer["expressed"] is False


def test_failed_calls_are_unjudged_and_retried_from_cache(family, tmp_path, monkeypatch):
    out = tmp_path / "out"
    argv = ["judge", "family=association", f"readouts={EXAMPLE}", f"out={out}", "items=chess"]
    fake = _install(monkeypatch, FakeJudge({"chess": "chess"}, fail_on="Two players"))
    assert main(argv) == 0
    r = read_results(out)
    assert r.counts["n_unjudged_cells"] == 1 and r.value == 1.0 and r.complete is False
    assert len(fake.calls) == 2  # a bare RuntimeError is fatal for that cell, not retried
    cache_rows = (out / "cells.jsonl").read_text().splitlines()
    assert len(cache_rows) == 2 and any('"result": null' in c for c in cache_rows)
    fake = _install(monkeypatch, FakeJudge({"chess": "chess"}))
    assert main(argv) == 0
    r = read_results(out)
    assert r.counts["n_unjudged_cells"] == 0 and r.extras["n_calls"] == 2
    assert len(fake.calls) == 1  # only the failed cell is re-asked


def test_undecided_items_leave_the_denominator(family, tmp_path, monkeypatch):
    # chess: L20 fails, L36 is a judged negative -> undecided; nasa: judged negative -> fails
    _install(monkeypatch, FakeJudge({}, fail_on="quiet game"))
    out = tmp_path / "out"
    argv = ["judge", "family=association", f"readouts={EXAMPLE}", f"out={out}", "items=chess,nasa"]
    assert main(argv) == 0
    r = read_results(out)
    rows = rows_by_id(r)
    assert rows["chess"]["pass"] is None and rows["chess"]["n_unjudged"] == 1
    assert rows["nasa"]["pass"] is False
    assert r.value == 0.0 and r.extras["n_items_decided"] == 1
    assert r.extras["n_items_undecided"] == 1 and r.counts["n_unjudged_cells"] == 1
    # a missing layer with no positive is undecided too, never a fail
    rows_ = [{"id": "chess", "layer": 20, "pos": 5, "samples": ["Two players sit in silence."]}]
    partial = _write(tmp_path / "partial.jsonl", rows_)
    _install(monkeypatch, FakeJudge({}))
    argv = ["judge", "family=association", f"readouts={partial}", f"out={out / 'p'}", "items=chess"]
    assert main([*argv, "layers=20,36", "allow_missing=True"]) == 0
    row = rows_by_id(read_results(out / "p"))["chess"]
    assert row["pass"] is None and row["n_missing"] == 1


def test_all_calls_failing_exits_3(family, tmp_path, monkeypatch):
    _install(monkeypatch, FakeJudge({}, fail_on="targets"))
    out = tmp_path / "out"
    argv = ["judge", "family=association", f"readouts={EXAMPLE}", f"out={out}", "items=chess"]
    assert main(argv) == 3


def test_layers_and_limit_restrict_the_grid(family, tmp_path, monkeypatch):
    fake = _install(monkeypatch, FakeJudge({"chess": "chess"}))
    out = tmp_path / "out"
    argv = ["judge", "family=association", f"readouts={EXAMPLE}", f"out={out}"]
    assert main([*argv, "limit=2", "layers=20", "allow_missing=True"]) == 0
    r = read_results(out)
    assert r.n_items == 2 and r.config["layers"] == [20]
    assert r.counts["n_expected_cells"] == 2 and r.counts["n_missing_cells"] == 0
    assert len(fake.calls) == 2 and r.value == 0.5


def test_prompt_version_bump_re_asks_and_keeps_old_rows(family, tmp_path, monkeypatch):
    from wsbench.basic import judge as judge_mod

    fake = _install(monkeypatch, FakeJudge({"chess": "chess"}))
    out = tmp_path / "out"
    argv = ["judge", "family=association", f"readouts={EXAMPLE}", f"out={out}", "items=chess"]
    assert main(argv) == 0 and len(fake.calls) == 2
    assert main(argv) == 0 and len(fake.calls) == 2  # fully cached
    monkeypatch.setattr(judge_mod, "PROMPT_VERSION", "bank-test-bump")
    assert main(argv) == 0 and len(fake.calls) == 4  # re-asked under the new version
    assert len((out / "cells.jsonl").read_text().splitlines()) == 4


def test_tokens_kind_readouts(family, tmp_path, monkeypatch):
    fake = _install(monkeypatch, FakeJudge({"Ġchess": "chess"}))
    rows = [
        {"id": "chess", "layer": 20, "pos": 5, "tokens": ["Ġgame", "Ġchess", "Ġclock"]},
        {"id": "chess", "layer": 36, "pos": 5, "tokens": ["Ġsilence", "Ġboard"]},
    ]
    readouts = _write(tmp_path / "tok.jsonl", rows)
    out = tmp_path / "out"
    argv = ["judge", "family=association", f"readouts={readouts}", f"out={out}", "items=chess"]
    assert main(argv) == 0
    r = read_results(out)
    assert r.config["kind"] == "tokens" and r.value == 1.0 and r.rows[0]["earliest_layer"] == 20
    assert "VOCABULARY TOKENS" in fake.calls[0]


def test_run_command_scores_the_family(family, tmp_path, monkeypatch):
    _install(monkeypatch, FakeJudge({"chess": "chess"}))
    root = tmp_path / "readouts"
    root.mkdir()
    (root / "association.jsonl").write_bytes(EXAMPLE.read_bytes())
    out = tmp_path / "run"
    argv = ["run", "families=association", f"readouts_root={root}", f"out={out}", "limit=2"]
    assert main([*argv, "allow_missing=True"]) == 0
    assert (out / "association" / "results.json").exists()
    assert "| association |" in (out / "summary.md").read_text()
