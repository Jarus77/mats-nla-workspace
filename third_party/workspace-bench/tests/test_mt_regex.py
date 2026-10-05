"""The multi-token regex scorer: the port is pinned to the SOURCE repo's verdicts (the goldens in
``tests/golden/mt_regex_*.json``), and ``run_regex`` applies the contract cell by cell."""

import importlib
import json
from pathlib import Path

import pytest

from wsbench import llm, registry
from wsbench.banks import load_bank
from wsbench.cli import main
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.multitoken import family as fam
from wsbench.multitoken import regex
from wsbench.multitoken.regex import SCORER_VERSION, SUMMARIZED_SCORER_VERSION, ScoredUnit
from wsbench.registry import JudgeArgs
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
MT = [
    "basic_readout_mt",
    "multihop_mt",
    "multilingual_mt",
    "typo_mt",
    "multilingual_multihop",
    "multilingual_typo",
]
GOLDEN = json.loads((REPO / "tests/golden/mt_regex_matching.json").read_text(encoding="utf-8"))
UNITS = json.loads((REPO / "tests/golden/mt_regex_units.json").read_text(encoding="utf-8"))


def test_matcher_verdicts_match_the_source():
    for pair in GOLDEN["pairs"]:
        got = regex.unicode_word_matcher(pair["form"])(pair["sample"])
        assert got is pair["hit"], (pair["form"], pair["sample"], pair["hit"])
    forms = {"en": ["Battle of Ain Jalut", "Ayn Jalut"], "zh": ["阿因贾鲁特战役", "艾因贾卢特"]}
    for sample, langs in GOLDEN["hit_forms"].items():
        assert regex.hit_forms([sample], forms) == langs, sample


def test_rule_branches_by_hand():
    m = regex.unicode_word_matcher
    assert m("Bohr magneton")("the **Bohr-magneton** here") and not m("Bohr magneton")("magnetons")
    assert m("kente")("kente cloth") and not m("kente")("kentes")
    assert m("Mexico")("México") and m("Glauber's salt")("Glauber\u2019s salt")
    assert m("沙漏")("这是沙漏") and m("손기정")("영웅 손기정입니다")  # substring for CJK / Hangul
    assert m("Байконур")("Байконур в") and not m("Байконур")("Байконуре")  # folded boundary
    assert m("435")("answer: 435") and m("435")("= 435") and m("435")("**435**")
    assert not m("435")("the 435 people") and not m("435")("4350") and not m("435")("435 + 1 =")
    assert m("58")("五十八") and not m("58")("五十八个") and m("435")("结果：四百三十五")  # noqa: RUF001
    assert regex.parse_cjk_numeral("四十九") == 49 and regex.parse_cjk_numeral("七十二个") is None
    # one top-k token never holds a two-word form; a phrase or label token can
    assert not m("Aghlabid dynasty")(" Aghlabid") and m("Aghlabid dynasty")("Aghlabid dynasty")
    assert regex.extract_phrase("<|im_start|>assistant\nx</think>") == "x"


def test_hit_forms_never_joins_samples():
    forms = {"en": ["New Delhi"]}
    assert regex.hit_forms(["New", "Delhi"], forms) == []
    assert regex.hit_forms(["in New Delhi today"], forms) == ["en"]


def test_grid_verdict_matches_the_source():
    forms = {"en": ["Battle of Ain Jalut", "Ayn Jalut"], "zh": ["阿因贾鲁特战役", "艾因贾卢特"]}
    lang = {"en": ["Arabic"], "zh": ["阿拉伯语"], "ar": ["عين جالوت", "موقعة عين جالوت"]}
    units = [ScoredUnit("readout", True, forms), ScoredUnit("language", True, lang)]
    grid = {
        20: ["nothing here"],
        24: ["Ayn Jalut was in 1260"],
        28: ["battle of ain jalut, in Arabic"],
        32: ["阿因贾鲁特战役 عين جالوت"],
    }
    got = regex.item_result({k: regex.layer_unit_hits(v, units) for k, v in grid.items()}, units)
    assert got == GOLDEN["grid"]
    assert got["passing_layers"] == [28, 32] and got["first_lang"] == {
        "readout": "en",
        "language": "en",
    }


@pytest.mark.parametrize("name", MT)
def test_scored_units_match_the_source_for_every_item(name):
    header, items = load_bank(REPO / "evals" / name / "items.json")
    contract = regex.contract_for(header, header["family"])
    assert contract.conjunctive_units and contract.multi_token
    assert contract.include_target == (name == "multihop_mt")
    for it in items:
        got = [u.to_json() for u in regex.scored_units(it, contract)]
        assert got == UNITS[name][it["name"]], it["name"]
        required = [u for u in got if u["required"]]
        assert required, it["name"]
        for u in got:
            if it["units"] and any(
                r["role"] == u["role"] and r.get("multi_token", True) for r in it["units"]
            ):
                ptl = it["probe_token_lens"]["units"][u["role"]]
                for lang, fs in u["forms"].items():
                    idx = [
                        it_forms.index(f) for f in fs for it_forms in [_forms(it, u["role"], lang)]
                    ]
                    assert all(ptl[lang][i] > 1 for i in idx), (it["name"], u["role"], lang)


def _forms(item, role, lang):
    return next(u for u in item["units"] if u["role"] == role)["forms"][lang]


def test_scored_units_refuses_a_required_unit_with_no_multi_token_form():
    item = {
        "name": "x",
        "units": [{"role": "r", "required": True, "forms": {"en": ["one"]}}],
        "probe_token_lens": {"units": {"r": {"en": [1]}}, "target": 1},
    }
    contract = regex.BankContract(True, False, True)
    with pytest.raises(ValueError):
        regex.scored_units(item, contract)
    item["units"][0]["multi_token"] = False  # a single-token unit stays creditable
    assert regex.scored_units(item, contract)[0].forms == {"en": ["one"]}


def _rows(items, layers, text_of):
    for it in items:
        for layer in layers:
            yield {"id": it["id"], "layer": layer, "pos": 7, **text_of(it, layer)}


def _args(path, out, **kw):
    base = {
        "readouts": path,
        "out": out,
        "judge": resolve(JudgeConfig(prompt_version=SCORER_VERSION)),
        "layers": None,
        "items": None,
        "limit": 0,
        "allow_missing": False,
        "concurrency": 1,
        "rpm": 1.0,
        "dry_run": False,
    }
    base.update(kw)
    return JudgeArgs(**base)


def test_run_regex_prose(tmp_path):
    """typo_mt: A hits at one layer (pass); B blank everywhere (empty, a negative); C misses;
    D has a layer missing and no pass (undecided); no LLM call, spend 0, pinned, not complete
    (subset)."""
    _h, items = load_bank(REPO / "evals/typo_mt/items.json")
    four = items[:4]
    a, b, c, d = (it["id"] for it in four)
    form = {it["id"]: it["units"][0]["forms"]["en"][0] for it in four}
    layers = [20, 24]

    def text_of(it, layer):
        if it["id"] == a:
            return {"samples": [f"the word is {form[a]} here"] if layer == 24 else ["nothing"]}
        if it["id"] == b:
            return {"samples": ["", "  "]}
        return {"samples": ["some other prose"]}

    rows = [r for r in _rows(four, layers, text_of) if not (r["id"] == d and r["layer"] == 24)]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    r = fam.run_regex(
        _args(path, tmp_path / "out", items=[a, b, c, d], allow_missing=True), name="typo_mt"
    )
    by = {row["id"]: row for row in r.rows}
    assert by[a]["pass"] is True and by[a]["earliest_layer"] == 24 and by[a]["any_hit"]
    assert by[b]["pass"] is False and by[b]["layers"]["20"]["empty"]
    assert by[c]["pass"] is False and by[d]["pass"] is None and by[d]["missing_layers"] == [24]
    assert r.value == pytest.approx(1 / 3) and r.n_items == 4
    assert r.counts == {
        "n_expected_cells": 8,
        "n_missing_cells": 1,
        "n_unjudged_cells": 0,
        "n_empty_cells": 2,
        "skipped_rows": 0,
        "spend_usd": 0.0,
    }
    assert r.pinned_instrument and not r.complete  # a subset run is never complete
    assert r.config["judge_model"] == "regex" and r.config["prompt_version"] == SCORER_VERSION
    assert r.extras["n_calls"] == 0 and r.extras["unit_any_layer"] == {"correction": 1 / 3}


def test_run_regex_tokens_and_the_language_unit(tmp_path, fake_llm, monkeypatch):
    """multilingual_mt (concept + language, both required): a top-k bag of single tokens hits the
    language name but never the two-word concept -> fail; a label-style token holding the whole
    phrase passes; the concept alone in a prose cell is any_hit but not a pass. The summarizer is
    stubbed with an unrelated interpretation, so the raw bag alone decides here."""
    monkeypatch.setattr(llm, "preflight", lambda model, reasoning: None)
    fake_llm(lambda system, user: {"interpretation": "noise about something else entirely"})
    _h, items = load_bank(REPO / "evals/multilingual_mt/items.json")
    it = next(i for i in items if len(i["units"][0]["forms"]["en"][0].split()) >= 2)
    concept = it["units"][0]["forms"]["en"][0]
    lang_name = it["units"][1]["forms"]["en"][0]
    path = tmp_path / "t.jsonl"
    rows = [
        {
            "id": it["id"],
            "layer": 20,
            "pos": 3,
            "tokens": [f" {lang_name}", *(f" {w}" for w in concept.split())],
        },
        {
            "id": it["id"],
            "layer": 24,
            "pos": 3,
            "tokens": [f"mentions of {concept} in {lang_name}"],
        },
    ]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    r = fam.run_regex(_args(path, tmp_path / "out", items=[it["id"]]), name="multilingual_mt")
    row = r.rows[0]
    assert row["layers"]["20"]["hits"] == {"concept": [], "language": ["en"]}
    assert row["layers"]["20"]["source"] == {"concept": [], "language": ["bag"]}
    assert row["pass"] is True and row["passing_layers"] == [24]
    assert r.extras["passing_layers_by_source"] == {"bag_only": 1}
    assert r.config["prompt_version"] == SUMMARIZED_SCORER_VERSION
    assert r.config["judge_model"] == "google/gemini-3.8-flash" and r.pinned_instrument
    prose = tmp_path / "p.jsonl"
    prose.write_text(
        json.dumps({"id": it["id"], "layer": 20, "pos": 3, "samples": [concept]}) + "\n"
    )
    r2 = fam.run_regex(_args(prose, tmp_path / "out2", items=[it["id"]]), name="multilingual_mt")
    assert (
        r2.rows[0]["pass"] is False and r2.rows[0]["any_hit"] and r2.extras["any_hit_rate"] == 1.0
    )


def test_run_regex_carries_a_previous_mc_number(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "results.json").write_text(
        json.dumps(
            {
                "config": {
                    "prompt_version": "mc-2026-09-16",
                    "judge_model": "google/gemini-3.8-flash",
                },
                "numbers": {"value": 0.42, "extras": {"n_items_decided": 100}},
            }
        )
    )
    _h, items = load_bank(REPO / "evals/typo_mt/items.json")
    it = items[0]
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"id": it["id"], "layer": 20, "pos": 1, "samples": ["x"]}) + "\n")
    r = fam.run_regex(_args(path, out, items=[it["id"]]), name="typo_mt")
    assert r.extras["mc"] == {
        "value": 0.42,
        "prompt_version": "mc-2026-09-16",
        "judge_model": "google/gemini-3.8-flash",
        "n_items_decided": 100,
    }


def test_dispatch_and_dry_run(tmp_path, capsys):
    _h, items = load_bank(REPO / "evals/typo_mt/items.json")
    it = items[0]
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"id": it["id"], "layer": 20, "pos": 1, "samples": ["x"]}) + "\n")
    r = fam.run_family(_args(path, tmp_path / "o", items=[it["id"]], dry_run=True), name="typo_mt")
    assert r.value is None and r.rows[0]["pass"] is None
    assert "[regex] typo_mt: scorer mt-regex-2026-09-23" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        fam.run_family(_args(path, tmp_path / "o", extra={"judge": "llm"}), name="typo_mt")


@pytest.mark.parametrize("name", MT)
def test_cli_dry_run_on_the_toy_file_makes_no_call(name, tmp_path, capsys, monkeypatch):
    spec = importlib.import_module(f"wsbench.evals.{name}").SPEC
    registry.FAMILIES.setdefault(spec.name, spec)  # conftest resets the registry per test
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")  # no key: a judge call would fail loudly
    out = tmp_path / "out"
    example = REPO / "examples/readouts" / f"{name}.jsonl"
    assert (
        main(["judge", f"family={name}", f"readouts={example}", f"out={out}", "dry_run=True"]) == 0
    )
    r = read_results(out)
    assert r.value is None and r.config["judge_model"] == "regex"
    assert "[regex]" in capsys.readouterr().out
    assert spec.scorer == "regex" and spec.judge.prompt_version == SCORER_VERSION


@pytest.mark.parametrize("name", MT)
def test_cli_scores_the_toy_file_offline(name, tmp_path, monkeypatch):
    """The toy files: the first item names the gold form at one layer (a pass), the second
    never does and is blank at one layer (a negative) — 0.5, offline, no key needed."""
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    spec = importlib.import_module(f"wsbench.evals.{name}").SPEC
    registry.FAMILIES.setdefault(spec.name, spec)
    out = tmp_path / "out"
    example = REPO / "examples/readouts" / f"{name}.jsonl"
    assert (
        main(["judge", f"family={name}", f"readouts={example}", f"out={out}", "allow_missing=True"])
        == 0
    )
    r = read_results(out)
    assert r.value == 0.5 and r.counts["spend_usd"] == 0.0 and r.pinned_instrument
    assert [row["pass"] for row in r.rows if row["pass"] is not None] == [True, False]
    assert r.counts["n_empty_cells"] == 1 and r.extras["n_calls"] == 0


def _token_rows(item, layers, tokens):
    return [
        {"id": item["id"], "layer": layer, "pos": 3, "tokens": list(tokens)} for layer in layers
    ]


def test_tokens_are_summarized_then_matched(tmp_path, fake_llm, monkeypatch):
    """The summary carries a two-word concept no single token can hold: a pass through the summary
    (source `summary` for the concept, the bag for the language name); the instrument is the
    summarizer model, spend counts its calls; a second run is all cache hits and makes no call."""
    _h, items = load_bank(REPO / "evals/multilingual_mt/items.json")
    it = next(i for i in items if len(i["units"][0]["forms"]["en"][0].split()) >= 2)
    concept = it["units"][0]["forms"]["en"][0]
    lang_name = it["units"][1]["forms"]["en"][0]
    monkeypatch.setattr(llm, "preflight", lambda model, reasoning: None)
    fake = fake_llm(
        lambda system, user: {
            "interpretation": f"The tokens point to {concept} in a {lang_name} text."
        }
    )
    path = tmp_path / "t.jsonl"
    rows = _token_rows(it, [20, 24], [f" {lang_name}", " noise", " more"])
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    out = tmp_path / "out"
    r = fam.run_regex(_args(path, out, items=[it["id"]]), name="multilingual_mt")
    row = r.rows[0]
    assert row["pass"] is True and row["passing_layers"] == [20, 24]
    assert row["layers"]["20"]["source"] == {"concept": ["summary"], "language": ["bag", "summary"]}
    # the summary alone carries both units (the bag alone only the language): "summary_only"
    assert r.extras["passing_layers_by_source"] == {"summary_only": 2}
    assert r.extras["n_summaries"] == 2 and r.extras["n_summaries_cached"] == 0
    assert r.extras["n_calls"] == 2 and len(fake.calls) == 2
    assert r.config["prompt_version"] == SUMMARIZED_SCORER_VERSION
    assert r.config["summarizer"] == "interp-v1" and r.config["judge_model"].startswith("google/")
    assert r.pinned_instrument and r.counts["n_unjudged_cells"] == 0
    # cached: the same bags in the same out dir -> no call, same verdicts
    r2 = fam.run_regex(_args(path, out, items=[it["id"]]), name="multilingual_mt")
    assert len(fake.calls) == 2 and r2.extras["n_calls"] == 0
    assert r2.extras["n_summaries_cached"] == 2 and r2.rows[0]["pass"] is True
    assert r2.counts["spend_usd"] == 0.0


def test_a_failed_summary_leaves_the_item_undecided(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(llm, "preflight", lambda model, reasoning: None)
    fake_llm(lambda system, user: None)  # every summarizer call fails
    _h, items = load_bank(REPO / "evals/typo_mt/items.json")
    it = items[0]
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(_token_rows(it, [20], [" some", " tokens"])[0]) + "\n")
    r = fam.run_regex(_args(path, tmp_path / "out", items=[it["id"]]), name="typo_mt")
    assert r.rows[0]["pass"] is None and r.rows[0]["unsummarized_layers"] == [20]
    assert r.counts["n_unjudged_cells"] == 1 and r.extras["n_summaries_failed"] == 1
    assert r.value is None and not r.complete


def test_an_unpinned_summarizer_unpins_the_token_instrument(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(llm, "preflight", lambda model, reasoning: None)
    fake_llm(lambda system, user: {"interpretation": "nothing relevant"})
    _h, items = load_bank(REPO / "evals/typo_mt/items.json")
    it = items[0]
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(_token_rows(it, [20], [" x"])[0]) + "\n")
    judge = resolve(JudgeConfig(prompt_version=SCORER_VERSION), flag="some/other-model")
    r = fam.run_regex(_args(path, tmp_path / "out", items=[it["id"]], judge=judge), name="typo_mt")
    assert r.config["judge_model"] == "some/other-model" and not r.pinned_instrument
    prose = tmp_path / "p.jsonl"
    prose.write_text(json.dumps({"id": it["id"], "layer": 20, "pos": 3, "samples": ["x"]}) + "\n")
    r2 = fam.run_regex(
        _args(prose, tmp_path / "out2", items=[it["id"]], judge=judge), name="typo_mt"
    )
    assert r2.config["judge_model"] == "regex" and r2.pinned_instrument  # prose ignores the flag
