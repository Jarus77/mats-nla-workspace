"""The read plan: every family plans, positions rules resolve on real token strings, and the
CLI writes one JSONL per family."""

import json

from wsbench import readplan
from wsbench.cli import main

# a Qwen chat render of one user-modelling item, as byte-level BPE display strings
UM_TOKENS = [
    *["<|im_start|>", "system", "Ċ", "You", "Ġare", "Ġhelpful", ".", "<|im_end|>", "Ċ"],
    *["<|im_start|>", "user", "Ċ", "I", "Ġlove", "Ġsopa", "."],
    *["ĠWrite", "Ġa", "Ġbio", "Ġfor", "Ġme", "."],
    *["<|im_end|>", "Ċ", "<|im_start|>", "assistant", "Ċ", "<think>", "ĊĊ", "</think>", "ĊĊ"],
]
POEM = [
    "A",
    "Ġrhyming",
    "Ġcouplet",
    ":",
    "Ċ",
    "The",
    "Ġroute",
    "Ġahead",
    ",",
    "Ċ",
    "And",
    "Ġwhere",
    "Ġhe",
    "'d",
    "Ġ",
]


def test_every_family_plans_and_rows_serialize():
    fams = readplan.families()
    assert len(fams) == 27
    for fam in fams:
        specs = readplan.plan(fam)
        assert specs, fam
        ids = [s.id for s in specs]
        assert len(set(ids)) == len(ids), fam
        for s in specs:
            assert s.render in readplan.RENDERS, (fam, s.render)
            assert s.layers and all(isinstance(L, int) for L in s.layers), fam
            row = s.row()
            json.dumps(row)  # serializable
            assert row["positions"]["kind"] and "family" in row and "id" in row
            if s.render not in ("captured",):
                assert s.text or s.messages, (fam, s.id)


def test_plans_match_the_bank_specs():
    arith = {s.id: s for s in readplan.plan("arithmetic_intermediates")}
    s = arith["addmul-000"]
    assert s.positions == {"kind": "all_from_end"} and s.layers == [56, 60]
    assert s.extra["frozen_cell"] == {"layer": 60, "pos": -8}
    buggy = {s.id: s for s in readplan.plan("buggy_code")}
    assert buggy["py-quantity-refund-credit"].layers == [60]
    assert next(s for s in buggy.values() if s.id.startswith("rust")).layers == [56]
    brew = readplan.plan("brew_intermediates")[0]
    assert brew.positions["kind"] == "positions" and len(brew.positions["positions"]) == 24
    assert brew.prefill == "Answer:" and brew.render == "chat_prefill"
    mcdm = readplan.plan("multi_concept_directed_modulation")[0]
    assert mcdm.layers == [44, 52, 56, 60] and mcdm.positions == {"kind": "last_n", "n": 12}
    conj = readplan.plan("conjunctive_association")[0]
    assert conj.layers == readplan.SIX and conj.suffix == readplan.SUMMARIZE_SUFFIX
    hal = readplan.plan("hallucination")[0]
    assert hal.render == "captured" and hal.extra["input_ids"] and hal.positions["positions"]
    multihop = readplan.plan("multihop")
    assert {s.positions["kind"] for s in multihop} == {"final_token", "offset_from_end"}
    poetry = readplan.plan("poetry")[0]
    assert poetry.positions == {"kind": "line_one_newline"} and poetry.layers == readplan.GRID
    jb = readplan.plan("jailbreak_recognition")[0]
    span = jb.positions["positions"]
    assert jb.render == "chat_context" and jb.messages[-1]["role"] == "user"
    assert jb.layers == readplan.FIVE
    implicit = next(s for s in readplan.plan("basic_readout") if s.render == "chat_prefill")
    assert implicit.prefill == readplan.ANSWER_STEM
    assert span[0] == 3 and span[-1] == 745 and len(span) == 743  # every token of the last turn


def test_resolve_each_rule():
    r = readplan.resolve
    assert r({"kind": "final_token"}, POEM) == [14]
    assert r({"kind": "offset_from_end", "k": 2}, POEM) == [-2]
    assert r({"kind": "last_n", "n": 3}, POEM) == [12, 13, 14]
    assert r({"kind": "all"}, POEM) == list(range(15))
    assert r({"kind": "all_from_end"}, POEM) == list(range(-15, 0))
    assert r({"kind": "positions", "positions": [1, 3, 99]}, POEM) == [1, 3]
    # the newline ending line one is the LAST newline: position 9, not the header's at 4
    assert r({"kind": "line_one_newline"}, POEM) == [9]
    # user modelling: from the request sentence ("Write a bio for me.") through the end
    got = r({"kind": "from_last_sentence_start"}, UM_TOKENS)
    assert got[0] == UM_TOKENS.index("ĠWrite") and got[-1] == len(UM_TOKENS) - 1
    # chain: from the question token through the end
    chain = [
        "The",
        "Ġstart",
        "Ġis",
        "Ġ23",
        ".",
        "ĠWhat",
        "Ġis",
        "Ġthe",
        "Ġfinal",
        "?",
        "<|im_end|>",
    ]
    assert r({"kind": "from_token", "token": " What"}, chain) == list(range(5, 11))
    # directed modulation: the carrier sentence's own tokens
    dm = [
        "Think",
        "Ġhard",
        ".",
        "<|im_end|>",
        "Ċ",
        "<|im_start|>",
        "assistant",
        "Ċ",
        "The",
        "Ġcommittee",
        "Ġmet",
        ".",
    ]
    assert r({"kind": "suffix_text", "text": "The committee met."}, dm) == [8, 9, 10, 11]
    # association families: the summarize suffix and everything after it
    story = [
        "Nora",
        "Ġwept",
        ".",
        "ĊĊ",
        "Sum",
        "mar",
        "ize",
        "Ġthe",
        "Ġstory",
        ".",
        "<|im_end|>",
        "Ċ",
        "<|im_start|>",
        "assistant",
        "Ċ",
        "<think>",
        "ĊĊ",
        "</think>",
        "ĊĊ",
    ]
    assert r({"kind": "suffix_text_then_tail", "text": "Summarize the story."}, story) == list(
        range(4, 19)
    )
    assert r({"kind": "suffix_text", "text": "Summarize the story."}, story) == list(range(4, 10))
    assert r({"kind": "final_token"}, []) == []


def test_cli_writes_one_file_per_family(tmp_path, capsys):
    assert main(["plan", "families=poetry,buggy_code", f"out={tmp_path}", "limit=3"]) == 0
    out = capsys.readouterr().out
    assert "poetry: 3 items" in out and "buggy_code: 3 items" in out
    rows = [json.loads(x) for x in (tmp_path / "poetry.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["positions"] == {"kind": "line_one_newline"}
    assert rows[0]["layers"] == readplan.GRID and rows[0]["render"] == "plain" and rows[0]["text"]
    assert main(["plan", "families=nope", f"out={tmp_path}"]) == 2
