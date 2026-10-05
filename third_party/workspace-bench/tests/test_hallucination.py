"""hallucination: frozen prompt, verdict classes, revocation, numbers, grid, resume (no network).

Ported from ``hallucination-bench/tests/{test_judge,test_cli,test_sites,test_data}.py`` onto the
wsbench entrypoint (``judge.run`` with the shared fake client).
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from conftest import write_jsonl

from wsbench.evals.hallucination import judge as hc
from wsbench.evals.hallucination import prompts as hp
from wsbench.evals.hallucination import score as hs
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.mcjudge import load_bank

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evals/hallucination"
TOY = ROOT / "examples/readouts/hallucination.jsonl"
TOPK = ROOT / "examples/readouts/hallucination.tokens.jsonl"
TOY_ITEMS = ["chat-dailydialog-0000", "chat-dailydialog-0001"]


def _raw(*samples: tuple[str, list[tuple[str, str]]]) -> dict[str, Any]:
    """samples = (kind, [(span_text, type), ...]) per readout, in idx order."""
    return {
        "samples": [
            {
                "idx": i,
                "kind": kind,
                "spans": [{"text": t, "type": ty, "why": "x"} for t, ty in spans],
                "rationale": "r",
            }
            for i, (kind, spans) in enumerate(samples)
        ]
    }


# ------------------------------------------------------------------ frozen prompt / bank


def test_prompt_literals_equal_the_committed_json() -> None:
    frozen = json.loads((DATA / "judge_prompt_v5c_chat.json").read_text())
    assert frozen == hp.FROZEN
    assert frozen["system"] == hp.JUDGE_SYSTEM
    assert frozen["user"] == hp.JUDGE_USER
    assert frozen["schema"] == hp.JUDGE_SCHEMA
    assert frozen["version"] == hp.PROMPT_VERSION == "v5c-chat"
    assert "{{text, type, why}}" in hp.JUDGE_USER  # the escaped literal, rendered by .format


def test_judge_prompt_marks_the_site_and_has_no_facts() -> None:
    item = {
        "prompt": "Where's your better half?",
        "response": "I am an AI, my better half is code.",
    }
    site = {"char": len("I am an AI,")}
    system, user = hp.judge_prompt(item, site, ["a", "b"])
    assert system == hp.JUDGE_SYSTEM
    assert f"I am an AI,{hp.READ_MARK} my better half" in user
    assert '<readout idx="1">\nb\n</readout>' in user
    assert "Verified facts" not in user
    assert "Below are 2 independent readouts" in user
    assert "{text, type, why}" in user and "{{" not in user


def test_render_marked_response_uses_the_frozen_offset() -> None:
    resp = "Once, upon a time."
    assert hp.render_marked_response(resp, 5) == "Once," + hp.READ_MARK + " upon a time."
    assert hp.render_marked_response(resp, 999).endswith("." + hp.READ_MARK)
    assert hp.render_marked_response(resp, -3).startswith(hp.READ_MARK)
    assert (
        hp.render_marked_response("Élan vital, naïve.", 11)
        == "Élan vital," + hp.READ_MARK + " naïve."
    )


def test_decode_token_byte_level_bpe() -> None:
    assert hc.decode_token("æĹ¥æľ¬èªŀ") == "日本語"
    assert hc.decode_token("Ġlanguages") == " languages"
    assert hc.decode_token(" AI") == " AI"


def test_bank_sites_place_the_marker_after_the_site_token() -> None:
    items = load_bank("hallucination")["items"]
    assert len(items) == 149
    for it in items:
        assert chr(0xFFFD) not in it["response"]
        assert 1 <= len(it["sites"]) <= 8
        for s in it["sites"]:
            assert it["response"][: s["char"]].endswith(s["token"])


def test_bank_and_capture_rows_agree() -> None:
    bank = json.loads((DATA / "items.json").read_text())
    items = bank["items"]
    rows = {r["id"]: r for r in json.loads((DATA / "capture_rows.json").read_text())}
    assert len(items) == bank["meta"]["n_items"] == 149
    assert sum(len(it["sites"]) for it in items) == bank["meta"]["n_sites"] == 1123
    assert Counter(it["source"] for it in items) == {"lmsys": 74, "dailydialog": 75}
    assert {it["id"] for it in items} == set(rows)
    for it in items:
        r = rows[it["id"]]
        assert r["source"] == it["source"] and r["prompt_len"] == it["response_start"]
        assert r["read_positions"] == [s["pos"] for s in it["sites"]]
        for s in it["sites"]:
            assert it["response_start"] <= s["pos"] < len(r["input_ids"])
            assert s["kind"] in hc.SITE_KINDS and s["kind"] != "eos"


# ------------------------------------------------------------------ verdict classes


def test_classes_wrong_beats_off_topic_and_kind() -> None:
    samples = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa lambda mu"]
    raw = _raw(
        ("specific", [("beta gamma", "off_topic"), ("alpha", "wrong")]),
        ("generic", [("epsilon", "wrong")]),
        ("empty", [("theta iota", "off_topic")]),
        ("specific", []),
    )
    v = hc.parse_verdict(raw, samples)
    assert v is not None
    assert [x.cls for x in v] == ["hallucinated", "hallucinated", "generic", "consistent"]


def test_off_topic_only_on_a_specific_readout() -> None:
    v = hc.parse_verdict(_raw(("specific", [("stray code", "off_topic")])), ["some stray code"])
    assert v is not None and v[0].cls == "off_topic" and v[0].off_topic == ["stray code"]


def test_generic_kind_with_no_spans_and_unknown_kind() -> None:
    v = hc.parse_verdict(_raw(("generic", []), ("weird", [])), ["a b c", "d e f"])
    assert v is not None
    assert [x.cls for x in v] == ["generic", "consistent"]
    assert v[1].kind == "specific"


def test_unverifiable_span_is_dropped_and_counted() -> None:
    v = hc.parse_verdict(_raw(("specific", [("not in readout", "wrong")])), ["the readout"])
    assert v is not None
    assert v[0].cls == "consistent" and v[0].n_unverified == 1 and v[0].wrong == []


def test_unknown_span_type_fails_closed_as_wrong() -> None:
    v = hc.parse_verdict(_raw(("specific", [("the readout", "mystery")])), ["the readout"])
    assert v is not None and v[0].cls == "hallucinated"


def test_missing_idx_is_unjudged_and_bad_raw_is_none() -> None:
    v = hc.parse_verdict(_raw(("specific", [])), ["a b c", "d e f"])
    assert v is not None and [x.cls for x in v] == ["consistent", "unjudged"]
    assert hc.parse_verdict(None, ["x"]) is None
    assert hc.parse_verdict({"nope": 1}, ["x"]) is None
    assert not hc.fully_judged(_raw(("specific", [])), ["a b c", "d e f"])
    assert hc.fully_judged(_raw(("specific", [])), ["a b c"])


# ------------------------------------------------------------------ revocation


def test_shared_wrong_span_revokes_both() -> None:
    samples = ["the Moon formed a crust", "then THE  MOON formed a crust today"]
    raw = _raw(
        ("specific", [("the Moon formed a crust", "wrong")]),
        ("specific", [("THE  MOON formed a crust", "wrong")]),
    )
    v = hc.parse_verdict(raw, samples)
    assert v is not None and [x.revoked for x in v] == [True, True]


def test_an_extra_unshared_span_is_not_revoked() -> None:
    samples = ["the Moon formed a crust and KBr", "the Moon formed a crust"]
    raw = _raw(
        ("specific", [("the Moon formed a crust", "wrong"), ("KBr", "wrong")]),
        ("specific", [("the Moon formed a crust", "wrong")]),
    )
    v = hc.parse_verdict(raw, samples)
    assert v is not None and [x.revoked for x in v] == [False, True]


def test_unverified_or_off_topic_copies_never_corroborate() -> None:
    samples = ["the Moon formed a crust", "nothing like that", "the Moon formed a crust"]
    raw = _raw(
        ("specific", [("the Moon formed a crust", "wrong")]),
        ("specific", [("the Moon formed a crust", "wrong")]),  # not verbatim in readout 1
        ("specific", [("the Moon formed a crust", "off_topic")]),
    )
    v = hc.parse_verdict(raw, samples)
    assert v is not None
    assert v[0].cls == "hallucinated" and not v[0].revoked
    assert v[1].n_unverified == 1


def test_k1_is_never_revoked() -> None:
    v = hc.parse_verdict(_raw(("specific", [("made up", "wrong")])), ["made up stuff"])
    assert v is not None and v[0].cls == "hallucinated" and not v[0].revoked


# ------------------------------------------------------------------ numbers


def _rv(cls: str, revoked: bool = False) -> hc.ReadoutVerdict:
    return hc.ReadoutVerdict(0, "specific", [], [], 0, cls, revoked)


def test_numbers_arithmetic() -> None:
    cells = [
        hc.CellRecord(
            "a", 20, 1, "clause", 3, [_rv("hallucinated", True), _rv("consistent"), _rv("generic")]
        ),
        hc.CellRecord("a", 44, 1, "clause", 2, [_rv("off_topic"), _rv("hallucinated")]),
        hc.CellRecord("b", 20, 5, "sentence", 3, None),  # unjudged
        hc.CellRecord(
            "b", 44, 5, "sentence", 3, [_rv("unjudged"), _rv("consistent"), _rv("consistent")]
        ),
    ]
    n = hs.numbers(cells)
    assert n["n_cells"] == 4 and n["n_unjudged_cells"] == 1 and n["n_short_cells"] == 1
    assert n["n_readouts"] == 7  # judged readouts only
    assert n["n_specific"] == 6 and n["n_hallucinated"] == 2 and n["n_revoked"] == 1
    assert n["hallucination_rate"] == pytest.approx(2 / 6)
    assert n["hallucination_rate_revoked"] == pytest.approx(1 / 6)
    assert n["off_topic_rate"] == pytest.approx(1 / 6)
    assert n["assert_share"] == pytest.approx(6 / 7)
    assert set(n["by_layer"]) == {"20", "44"}
    assert n["by_layer"]["20"]["hallucination_rate"] == pytest.approx(1 / 2)
    assert set(n["by_site_kind"]) == {"clause", "sentence"}


def test_zero_denominator_gives_none_not_nan() -> None:
    n = hs.numbers([hc.CellRecord("a", 20, 1, "clause", 1, [_rv("generic")])], k=1)
    assert n["hallucination_rate"] is None and n["ci95"] is None
    assert n["n_short_cells"] == 0


def test_bootstrap_is_deterministic_and_brackets_the_rate() -> None:
    cells = [
        hc.CellRecord(f"i{i}", 20, 1, "clause", 3, [_rv("hallucinated" if i % 3 else "consistent")])
        for i in range(30)
    ]
    ci = hs.bootstrap_rate_ci(cells, seed=0)
    assert ci == hs.bootstrap_rate_ci(cells, seed=0)
    assert ci is not None
    rate = hs.numbers(cells)["hallucination_rate"]
    assert ci[0] <= rate <= ci[1]
    assert hs.N_BOOT == 2000


# ------------------------------------------------------------------ the family run


def _readout_texts(user: str) -> list[str]:
    return [
        r.split("\n", 1)[1].rsplit("\n</readout>", 1)[0] for r in user.split('<readout idx="')[1:]
    ]


def _verify_answer(texts: list[str]) -> dict[str, Any]:
    """Stage-2 answer per readout: idx 0 -> a true claim, an unverifiable one and a paraphrase
    (not verbatim, dropped); idx 1 -> one claim inside its established-false span (skipped);
    idx 2 -> a disputed claim and an off-topic one."""

    def claims(j: int, t: str) -> list[tuple[str, str]]:
        if j == 0:
            return [(t[:12], "true"), (t[-8:], "unverifiable"), ("not in the readout", "true")]
        if j == 1:
            return [(t[:12], "true")]
        return [(t[:12], "disputed"), (t[-8:], "off_topic")]

    return {
        "samples": [
            {
                "idx": j,
                "kind": "specific",
                "claims": [{"quote": q, "status": st, "why": "toy"} for q, st in claims(j, t)],
                "rationale": "",
            }
            for j, t in enumerate(texts)
        ]
    }


def _responder(system: str, user: str) -> dict[str, Any] | None:
    """Judge calls: flags readout 1's full text as wrong in every cell. Verify calls: see
    ``_verify_answer``. Summary calls: a fixed readout. Preflight: {"a": 1}."""
    if user.startswith("Return"):
        return {"a": 1}
    if user.startswith("TOKEN READOUTS:"):
        return {"interpretation": "The model is about to say it is an AI with no spouse."}
    texts = _readout_texts(user)
    if system == hp.VERIFY_SYSTEM:
        return _verify_answer(texts)
    assert system == hp.JUDGE_SYSTEM
    return {
        "samples": [
            {
                "idx": j,
                "kind": "specific",
                "spans": [{"text": t, "type": "wrong", "why": "toy"}] if j == 1 else [],
                "rationale": "",
            }
            for j, t in enumerate(texts)
        ]
    }


def _judge_calls(fake) -> list[str]:
    return [u for s, u in fake.calls if s == hp.JUDGE_SYSTEM]


def _verify_calls(fake) -> list[str]:
    return [u for s, u in fake.calls if s == hp.VERIFY_SYSTEM]


def test_missing_cells_stop_the_run_unless_allowed(tmp_path, fake_llm, mk_args):
    fake = fake_llm(_responder)
    with pytest.raises(SystemExit) as e:
        hc.run(mk_args(TOY, items=TOY_ITEMS))
    assert e.value.code == 2 and fake.calls == []


def test_end_to_end_and_resume(tmp_path, fake_llm, mk_args):
    fake = fake_llm(_responder)
    args = mk_args(TOY, items=TOY_ITEMS, allow_missing=True)
    res = hc.run(args)
    assert res.family == "hallucination" and res.metric == "hallucination_rate"
    assert res.higher_is_better is False and res.chance is None
    assert res.chance_label == hs.CHANCE_LABEL
    assert len(_judge_calls(fake)) == 4 and fake.calls[0][1] == 'Return {"a":1}'
    c = res.counts
    assert c["n_expected_cells"] == 16 and c["n_missing_cells"] == 16 - 4
    assert c["n_unjudged_cells"] == 0 and c["n_empty_cells"] == 0
    assert c["skipped_rows"] == 1 and c["spend_usd"] > 0  # the pos-999 row is off-site
    assert res.complete is False and res.pinned_instrument is True  # subset + missing
    e = res.extras
    assert e["n_cells"] == 4 and e["n_readouts"] == 12 and e["n_specific"] == 12
    # readout 1 is flagged in every cell; never revoked since only readout 1 is ever flagged
    assert e["n_hallucinated"] == 4 and e["n_revoked"] == 0
    assert res.value == pytest.approx(4 / 12) and res.ci95 is not None
    assert e["hallucination_rate_revoked"] == pytest.approx(4 / 12)
    assert e["assert_share"] == 1.0 and e["off_topic_rate"] == 0.0
    assert set(e) >= {
        "hallucination_rate_revoked", "assert_share", "off_topic_rate", "n_specific",
        "n_readouts", "by_layer", "by_site_kind", "n_unverified_spans",
    }  # fmt: skip
    assert "hallucination_rate" not in e and set(e["by_layer"]) == {"36"}
    assert res.n_items == 2
    assert res.config["prompt_version"] == "v5c-chat" and res.config["kind"] == "prose"
    # stage 2: one verify call per span-judged cell, its wrong spans given as established false
    vc = _verify_calls(fake)
    assert len(vc) == 4 and all(hp.READ_MARK in u for u in vc)
    assert all("Established FALSE spans of readout 1:\n  - " in u for u in vc)
    assert all("Established FALSE spans of readout 0:\n  (none)" in u for u in vc)
    assert (
        res.config["verify"] is True and res.config["verify_prompt_version"] == "v5c-chat-verify-v1"
    )
    assert e["n_cells_claims_unjudged"] == 0 and e["n_readouts_claims_judged"] == 12
    # per cell: idx0 -> 1 true + 1 unverifiable (+1 dropped paraphrase); idx1 -> its claim
    # overlaps the false span (skipped, false = 1); idx2 -> 1 disputed (= unverifiable) + 1 off
    assert e["n_claims_true"] == 4 and e["n_claims_false"] == 4
    assert e["n_claims_unverifiable"] == 8 and e["n_claims_disputed"] == 4
    assert e["n_claims_off_topic"] == 4 and e["n_unverified_claim_quotes"] == 4
    assert e["verifiable_share"] == pytest.approx(8 / 16)
    assert e["false_share_of_verifiable"] == pytest.approx(4 / 8)
    assert e["unverifiable_claims_per_readout"] == pytest.approx(8 / 12)
    assert e["by_layer"]["36"]["n_claims_unverifiable"] == 8
    assert sum(b["n_claims_true"] for b in e["by_site_kind"].values()) == 4
    cached = [json.loads(line) for line in (args.out / "cells.jsonl").read_text().splitlines()]
    assert sorted(r["key"] for r in cached if r["key"].endswith(":V")) == sorted(
        r["key"] + ":V" for r in cached if not r["key"].endswith(":V")
    )
    assert res.config["k"] == 3 and res.config["reasoning"] == {"effort": "minimal"}
    assert res.config["judge_model"] == "google/gemini-3.8-flash"
    row = next(r for r in res.rows if r["pos"] == 23)
    assert set(row) == {"key", "id", "layer", "pos", "site_kind", "samples", "verdict"}
    assert row["key"] == "chat-dailydialog-0000__L036__p23" and len(row["samples"]) == 3
    assert [v["class"] for v in row["verdict"]] == ["consistent", "hallucinated", "consistent"]
    assert [v["claims"] for v in row["verdict"]] == [
        {
            "false": 0,
            "true": 1,
            "unverifiable": 1,
            "disputed": 0,
            "off_topic": 0,
            "n_unverified": 1,
        },
        {
            "false": 1,
            "true": 0,
            "unverifiable": 0,
            "disputed": 0,
            "off_topic": 0,
            "n_unverified": 0,
        },
        {
            "false": 0,
            "true": 0,
            "unverifiable": 1,
            "disputed": 1,
            "off_topic": 1,
            "n_unverified": 0,
        },
    ]
    # resume: every cell cached, no calls, same numbers
    fake2 = fake_llm(_responder)
    again = hc.run(args)
    assert fake2.calls == [] and again.value == res.value
    # another judge model: nothing cached is usable, every cell re-judged, not pinned
    fake3 = fake_llm(_responder)
    other = hc.run(
        mk_args(
            TOY,
            items=TOY_ITEMS,
            allow_missing=True,
            judge=resolve(JudgeConfig(), flag="other/model", env={}),
        )
    )
    assert len(_judge_calls(fake3)) == 4 and other.pinned_instrument is False
    # switching back reuses the earlier verdicts
    fake4 = fake_llm(_responder)
    hc.run(args)
    assert fake4.calls == []


def test_partly_judged_cell_is_unjudged_and_retried(tmp_path, fake_llm, mk_args):
    def half(system, user):
        if user.startswith("Return"):
            return {"a": 1}
        return {"samples": [{"idx": 0, "kind": "specific", "spans": [], "rationale": ""}]}

    fake = fake_llm(half)
    args = mk_args(TOY, items=TOY_ITEMS, allow_missing=True)
    res = hc.run(args)
    assert res.counts["n_unjudged_cells"] == 4 and res.extras["n_unjudged_cells"] == 4
    assert res.value is None and res.ci95 is None
    assert all(r["verdict"] is None for r in res.rows)
    first = _judge_calls(fake)
    cached = [json.loads(line) for line in (args.out / "cells.jsonl").read_text().splitlines()]
    assert all(r["result"] is None and "raw" in r["meta"] for r in cached)
    # the second run re-issues exactly those keys
    fake2 = fake_llm(_responder)
    res2 = hc.run(args)
    assert sorted(_judge_calls(fake2)) == sorted(first)
    assert res2.counts["n_unjudged_cells"] == 0 and res2.value == pytest.approx(4 / 12)


def test_topk_rows_are_summarised_then_judged_k1(tmp_path, fake_llm, mk_args):
    fake = fake_llm(_responder)
    res = hc.run(mk_args(TOPK, items=["chat-dailydialog-0000"], allow_missing=True))
    users = [u for s, u in fake.calls if s != hp.VERIFY_SYSTEM and not u.startswith("Return")]
    assert len(users) == 2 and users[0].startswith("TOKEN READOUTS:\n AI (9.10) |  spouse (8.40)")
    assert len(_verify_calls(fake)) == 1  # the summary is verified as one readout
    assert res.config["kind"] == "tokens" and res.config["k"] == 1
    assert res.config["summary_prompt_version"] == "interp-v1"
    row = res.rows[0]
    assert row["samples"] == ["The model is about to say it is an AI with no spouse."]
    assert row["tokens"][:2] == [" AI", " spouse"]
    assert "ĠAI" not in users[1] and row["samples"][0] in users[1]
    assert res.extras["n_readouts"] == 1 and res.extras["n_short_cells"] == 0
    assert res.value == 0.0  # the toy responder flags idx 1 only, so a k=1 cell is consistent


def test_failed_resummary_never_reuses_the_old_summary(tmp_path, fake_llm, mk_args):
    src = tmp_path / "r.jsonl"
    row = {
        "id": "chat-dailydialog-0000",
        "layer": 36,
        "pos": 23,
        "tokens": ["Ġv1"],
        "scores": [1.0],
    }
    write_jsonl(src, [row])
    fake_llm(_responder)
    args = mk_args(src, items=["chat-dailydialog-0000"], allow_missing=True)
    hc.run(args)
    write_jsonl(src, [{**row, "tokens": ["Ġv2", "Ġnew"], "scores": [2.0, 1.0]}])

    def summaries_fail(system, user):
        if user.startswith("TOKEN READOUTS:"):
            return None
        return _responder(system, user)

    fake_llm(summaries_fail)
    res = hc.run(args)
    assert res.rows[0]["samples"] is None and res.rows[0]["verdict"] is None
    assert res.counts["n_unjudged_cells"] == 1 and res.extras["n_readouts"] == 0


def test_malformed_mixed_and_empty_rows(tmp_path, fake_llm, mk_args):
    rows = [
        [1, 2],
        {"id": "chat-dailydialog-0000", "layer": 36, "pos": 23, "samples": "not a list"},
        {"id": "chat-dailydialog-0000", "layer": "x", "pos": 23, "samples": ["a"]},
        {"id": "chat-dailydialog-0000", "layer": 36, "pos": 23, "samples": ["ok"], "tokens": []},
        {"id": "chat-dailydialog-0000", "layer": 36, "pos": 38, "samples": [" ", ""]},
        {"id": "chat-dailydialog-0000", "layer": 36, "pos": 23, "samples": ["The AI replies."]},
        {"id": "chat-dailydialog-0000", "layer": 36, "pos": 23, "samples": ["dup"]},
        {"id": "nope", "layer": 36, "pos": 23, "samples": ["a"]},
        {"id": "chat-dailydialog-0000", "layer": 36, "pos": 999, "samples": ["off-site"]},
    ]  # fmt: skip
    src = write_jsonl(tmp_path / "r.jsonl", rows)
    fake_llm(_responder)
    res = hc.run(mk_args(src, items=["chat-dailydialog-0000"], allow_missing=True))
    # 4 malformed + 1 unknown id + 1 duplicate + 1 off-site; the empty cell is a result
    assert res.counts["skipped_rows"] == 7 and res.counts["n_empty_cells"] == 1
    # 8 expected - 1 judged - 1 empty
    assert res.counts["n_missing_cells"] == 6 and res.extras["n_cells"] == 1


def test_all_empty_layer_stays_in_the_grid(tmp_path, fake_llm, mk_args):
    it = load_bank("hallucination")["items"][0]
    rows = [
        {"id": it["id"], "layer": 36, "pos": s["pos"], "samples": ["The assistant says hi."]}
        for s in it["sites"]
    ]
    rows += [{"id": it["id"], "layer": 60, "pos": s["pos"], "samples": [""]} for s in it["sites"]]
    fake_llm(_responder)
    res = hc.run(mk_args(write_jsonl(tmp_path / "r.jsonl", rows), items=[it["id"]]))
    assert res.config["judged_layers"] == [36, 60]
    assert res.counts["n_empty_cells"] == len(it["sites"]) == 8
    assert res.counts["n_missing_cells"] == 0 and res.complete is False  # --items subset


def _full_grid(tmp_path, layer: int = 36) -> Path:
    items = load_bank("hallucination")["items"]
    return write_jsonl(
        tmp_path / "full.jsonl",
        [
            {"id": it["id"], "layer": layer, "pos": s["pos"], "samples": ["The assistant replies."]}
            for it in items
            for s in it["sites"]
        ],
    )


def test_complete_requires_the_full_grid_and_the_pinned_judge(tmp_path, fake_llm, mk_args):
    src = _full_grid(tmp_path)
    fake = fake_llm(_responder)
    res = hc.run(mk_args(src))
    assert len(_judge_calls(fake)) == 1123
    assert res.complete is True and res.pinned_instrument is True and res.n_items == 149
    assert res.extras["n_cells"] == 1123 and res.counts["n_expected_cells"] == 1123
    # a single-layer lens judged at its one layer via --layers is still complete
    fake_llm(_responder)
    res = hc.run(mk_args(src, layers=[36]))
    assert res.complete is True and res.config["layers"] == [36]
    fake_llm(_responder)
    res = hc.run(mk_args(src, judge=resolve(JudgeConfig(), flag="other/model", env={})))
    assert res.complete is False and res.pinned_instrument is False
    fake_llm(_responder)
    assert hc.run(mk_args(src, limit=3)).complete is False


def test_cache_survives_torn_and_foreign_lines(tmp_path, fake_llm, mk_args):
    fake_llm(_responder)
    args = mk_args(TOY, items=TOY_ITEMS, allow_missing=True)
    hc.run(args)
    cache = args.out / "cells.jsonl"
    text = cache.read_text()
    cache.write_text(text + '{"note": "hi"}\n[1]\n' + text.splitlines()[0][:40])  # torn tail
    fake2 = fake_llm(_responder)
    hc.run(args)
    assert fake2.calls == []


def test_dry_run_prints_the_prompt_and_makes_no_calls(tmp_path, mk_args, capsys, monkeypatch):
    monkeypatch.setattr("wsbench.llm._make_client", lambda r, k: pytest.fail("no client"))
    res = hc.run(mk_args(TOY, dry_run=True))  # missing cells are reported, not fatal, on a dry run
    out = capsys.readouterr().out
    assert hp.READ_MARK in out and hp.JUDGE_SYSTEM in out and "missing cells" in out
    assert hp.VERIFY_SYSTEM not in out  # no verdict exists on a dry run, so no verify prompt
    assert res.value is None and res.ci95 is None and res.rows == []
    assert res.config["dry_run"] is True and res.complete is False and res.n_items == 149
    assert res.counts["n_missing_cells"] == 1123 - 4 and res.counts["spend_usd"] == 0.0
    # a tokens file prints only the summarizer prompt
    hc.run(mk_args(TOPK, dry_run=True))
    out = capsys.readouterr().out
    assert "TOKEN READOUTS:" in out and hp.JUDGE_SYSTEM not in out


# ------------------------------------------------------------------ stage 2: claim verification


def _vraw(*per_readout: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "samples": [
            {
                "idx": i,
                "kind": "specific",
                "claims": [{"quote": q, "status": st, "why": "x"} for q, st in claims],
                "rationale": "",
            }
            for i, claims in enumerate(per_readout)
        ]
    }


def test_tally_claims_rules() -> None:
    samples = ["The assistant names its wife, Sarah.", "A friendly exclamation."]
    wrong = [["its wife, Sarah"], []]
    t = hc.tally_claims(
        _vraw(
            [
                ("names its wife", "true"),  # overlaps the established-false span -> skipped
                ("The assistant", "true"),
                ("the assistant", "TRUE "),  # case / whitespace normalised, still verbatim
                ("Sarah is her wife", "true"),  # paraphrase: not verbatim -> dropped, counted
                ("names", "bogus"),  # unknown status never inflates true
            ],
            [
                ("friendly", "disputed"),
                ("exclamation", "off_topic"),
                ("A friendly", "unverifiable"),
            ],
        ),
        samples,
        wrong,
    )
    assert t == [
        {
            "false": 1,
            "true": 3,
            "unverifiable": 1,
            "disputed": 0,
            "off_topic": 0,
            "n_unverified": 1,
        },
        {
            "false": 0,
            "true": 0,
            "unverifiable": 2,
            "disputed": 1,
            "off_topic": 1,
            "n_unverified": 0,
        },
    ]


def test_tally_claims_rejects_partial_or_wrong_shaped_answers() -> None:
    samples = ["one readout", "another readout"]
    assert hc.tally_claims(None, samples, [[], []]) is None
    assert hc.tally_claims({"samples": "x"}, samples, [[], []]) is None
    assert hc.tally_claims(_vraw([]), samples, [[], []]) is None  # idx 1 missing
    span_shaped = _raw(("specific", []), ("specific", []))  # a stage-1 answer: no `claims`
    assert hc.tally_claims(span_shaped, samples, [[], []]) is None
    assert hc.tally_claims(_vraw([], []), samples, [["x"], []]) == [
        {
            "false": 1,
            "true": 0,
            "unverifiable": 0,
            "disputed": 0,
            "off_topic": 0,
            "n_unverified": 0,
        },
        {
            "false": 0,
            "true": 0,
            "unverifiable": 0,
            "disputed": 0,
            "off_topic": 0,
            "n_unverified": 0,
        },
    ]


def test_claims_numbers_arithmetic() -> None:
    def rv(cls, claims):
        return hc.ReadoutVerdict(0, "specific", [], [], 0, cls, False, claims)

    tallied = {
        "false": 2,
        "true": 3,
        "unverifiable": 4,
        "disputed": 1,
        "off_topic": 0,
        "n_unverified": 1,
    }
    cells = [
        hc.CellRecord(
            "a", 36, 1, "clause", 3, [rv("consistent", tallied), rv("hallucinated", None)]
        ),
        hc.CellRecord("a", 36, 2, "clause", 3, [rv("consistent", tallied)]),
        hc.CellRecord("b", 36, 1, "clause", 3, None),
    ]
    n = hs.numbers(cells)
    assert n["n_readouts_claims_judged"] == 2 and n["n_cells_claims_unjudged"] == 1
    assert n["n_claims_true"] == 6 and n["n_claims_false"] == 4 and n["n_claims_unverifiable"] == 8
    assert n["n_claims_disputed"] == 2 and n["n_unverified_claim_quotes"] == 2
    assert n["verifiable_share"] == pytest.approx(10 / 18)
    assert n["false_share_of_verifiable"] == pytest.approx(4 / 10)
    assert n["unverifiable_claims_per_readout"] == pytest.approx(8 / 2)
    empty = hs.numbers([hc.CellRecord("a", 36, 1, "clause", 3, [rv("consistent", None)])])
    assert empty["verifiable_share"] is None and empty["n_cells_claims_unjudged"] == 1


def test_verify_can_be_switched_off(tmp_path, fake_llm, mk_args):
    fake = fake_llm(_responder)
    res = hc.run(mk_args(TOY, items=TOY_ITEMS, allow_missing=True, extra={"verify": "0"}))
    assert len(_judge_calls(fake)) == 4 and _verify_calls(fake) == []
    assert res.config["verify"] is False and res.config["verify_prompt_version"] is None
    assert res.value == pytest.approx(4 / 12)  # the headline is the same instrument
    e = res.extras
    assert e["n_cells_claims_unjudged"] == 4 and e["n_readouts_claims_judged"] == 0
    assert e["n_claims_true"] == 0 and e["verifiable_share"] is None
    assert all(v["claims"] is None for r in res.rows for v in r["verdict"])


def test_failed_verify_never_scores_and_is_retried(tmp_path, fake_llm, mk_args):
    def no_verify(system, user):
        return None if system == hp.VERIFY_SYSTEM else _responder(system, user)

    fake = fake_llm(no_verify)
    args = mk_args(TOY, items=TOY_ITEMS, allow_missing=True)
    res = hc.run(args)
    assert len(_verify_calls(fake)) == 4 and res.value == pytest.approx(4 / 12)
    assert res.counts["n_unjudged_cells"] == 0  # the headline's coverage is untouched
    assert res.extras["n_cells_claims_unjudged"] == 4 and res.extras["n_claims_true"] == 0

    # a wrong-shaped verify answer is cached as a failure with the raw answer, then re-queued
    def span_shaped(system, user):
        if system == hp.VERIFY_SYSTEM:
            return _raw(*[("specific", [])] * len(_readout_texts(user)))
        return _responder(system, user)

    fake2 = fake_llm(span_shaped)
    hc.run(args)
    assert len(_verify_calls(fake2)) == 4 and _judge_calls(fake2) == []
    cached = [json.loads(line) for line in (args.out / "cells.jsonl").read_text().splitlines()]
    vrows = [r for r in cached if r["key"].endswith(":V")]  # 4 API failures + 4 rejects
    assert len(vrows) == 8 and all(r["result"] is None for r in vrows)
    assert all("raw" in r["meta"] for r in vrows[4:])
    assert not any("raw" in r["meta"] for r in vrows[:4])
    fake3 = fake_llm(_responder)
    res3 = hc.run(args)
    assert len(_verify_calls(fake3)) == 4 and _judge_calls(fake3) == []
    assert res3.extras["n_cells_claims_unjudged"] == 0 and res3.extras["n_claims_true"] == 4
    fake4 = fake_llm(_responder)
    hc.run(args)
    assert fake4.calls == []
