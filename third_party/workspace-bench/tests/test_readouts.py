import json

from wsbench.readouts import (
    Cell,
    convert_gen_dir,
    expected_cells,
    load_readouts,
    missing_cells,
)


def _write(path, rows):
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


def test_prose_file_loads(tmp_path):
    p = _write(
        tmp_path / "r.jsonl",
        [
            {"id": "a", "layer": 36, "pos": 3, "samples": ["x", "y"]},
            {"id": "a", "layer": 20, "pos": 3, "samples": []},
            {"id": "b", "layer": 36, "pos": 0, "samples": ["  "]},
        ],
    )
    cells, rep = load_readouts(p)
    assert rep.kind == "prose" and rep.n_rows == 3 and rep.n_empty == 2
    assert rep.layers == [20, 36]
    assert sum(rep.skipped.values()) == 0
    assert cells[0] == Cell(id="a", layer=36, pos=3, samples=("x", "y"), tokens=None, scores=None)
    assert cells[0].kind == "prose" and cells[0].key == "a__L036__p3"


def test_tokens_file_loads(tmp_path):
    p = _write(
        tmp_path / "r.jsonl",
        [
            {"id": "a", "layer": 36, "pos": 3, "tokens": ["Ġw", "x"], "scores": [1.5, 1]},
            {"id": "a", "layer": 36, "pos": 4, "tokens": ["q"]},
            {"id": "a", "layer": 36, "pos": 5, "tokens": []},
        ],
    )
    cells, rep = load_readouts(p)
    assert rep.kind == "tokens" and rep.n_empty == 1
    assert (
        cells[0].kind == "tokens"
        and cells[0].tokens == ("Ġw", "x")
        and cells[0].scores == (1.5, 1.0)
    )
    assert cells[1].scores is None


def test_every_skip_category(tmp_path):
    p = _write(
        tmp_path / "r.jsonl",
        [
            "{not json",
            {"id": "a", "layer": "36", "pos": 3, "samples": ["x"]},  # layer not int
            {"id": "a", "layer": 36, "pos": 3},  # neither samples nor tokens
            {"id": "a", "layer": 36, "pos": 3, "samples": ["x"], "tokens": ["y"]},  # both
            {"id": "a", "layer": 36, "pos": 3, "tokens": ["y"], "scores": [1, 2]},  # length
            {"id": "a", "layer": 36, "pos": 3, "samples": [1]},  # non-str sample
            [1, 2],
            {"id": "a", "layer": 36, "pos": 3, "samples": ["ok"]},
            {"id": "a", "layer": 36, "pos": 3, "samples": ["dup"]},
            {"id": "a", "layer": 36, "pos": 4, "tokens": ["mixed"]},
            {"id": "zzz", "layer": 36, "pos": 3, "samples": ["unknown"]},
            {"id": "a", "layer": 99, "pos": 3, "samples": ["layer"]},
            {"id": "a", "layer": 36, "pos": 77, "samples": ["pos"]},
        ],
    )
    cells, rep = load_readouts(p, ids=["a"], layers=[36], positions={"a": [3]})
    assert [c.samples for c in cells] == [("ok",)]
    assert rep.skipped == {
        "malformed": 7,
        "duplicate": 1,
        "mixed_kind": 1,
        "unknown_id": 1,
        "layer_not_selected": 1,
        "pos_not_selected": 1,
    }
    assert rep.n_rows == 13


def test_unknown_id_from_positions_when_ids_absent(tmp_path):
    p = _write(
        tmp_path / "r.jsonl",
        [
            {"id": "a", "layer": 1, "pos": 0, "samples": ["x"]},
            {"id": "b", "layer": 1, "pos": 0, "samples": ["x"]},
        ],
    )
    cells, rep = load_readouts(p, positions={"a": [0]})
    assert len(cells) == 1 and rep.skipped["unknown_id"] == 1


def test_mixed_kind_first_row_wins(tmp_path):
    p = _write(
        tmp_path / "r.jsonl",
        [
            {"id": "a", "layer": 1, "pos": 0, "tokens": ["x"]},
            {"id": "a", "layer": 1, "pos": 1, "samples": ["x"]},
        ],
    )
    cells, rep = load_readouts(p)
    assert rep.kind == "tokens" and len(cells) == 1 and rep.skipped["mixed_kind"] == 1


def test_empty_file(tmp_path):
    p = _write(tmp_path / "r.jsonl", [])
    cells, rep = load_readouts(p)
    assert cells == [] and rep.kind is None and rep.layers == [] and rep.n_rows == 0


def test_expected_and_missing(tmp_path):
    exp = expected_cells({"a": [1, 2], "b": [5]}, [10, 20])
    assert exp == {
        ("a", 10, 1),
        ("a", 10, 2),
        ("a", 20, 1),
        ("a", 20, 2),
        ("b", 10, 5),
        ("b", 20, 5),
    }
    cells = [
        Cell("a", 10, 1, ("x",), None, None),
        Cell("a", 10, 2, (), None, None),  # empty readout is present, not missing
        Cell("c", 10, 1, ("x",), None, None),  # unexpected cells are ignored
    ]
    assert missing_cells(cells, exp) == [("a", 20, 1), ("a", 20, 2), ("b", 10, 5), ("b", 20, 5)]


def _gen_dir(tmp_path):
    g = tmp_path / "gen"
    for label in ("item1", "item2"):
        d = g / label
        d.mkdir(parents=True)
        for layer in (20, 36):
            rows = [
                {
                    "label": "old",
                    "layer": 999,
                    "pos": 3,
                    "samples": ["s1", "s2"],
                    "scores": [2.0, 1.0],
                },
                {"layer": layer, "pos": 4, "samples": ["s3"]},
                {"layer": layer, "samples": ["nopos"]},
                {"layer": layer, "pos": 5},
            ]
            (d / f"L{layer:03d}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (g / "notes.txt").write_text("ignored")
    return g


def test_convert_prose_round_trip(tmp_path):
    g = _gen_dir(tmp_path)
    out = tmp_path / "out" / "prose.jsonl"
    rep = convert_gen_dir(g, out, kind="prose")
    assert rep.kind == "prose" and rep.layers == [20, 36]
    assert rep.skipped["malformed"] == 8  # 2 bad rows x 2 labels x 2 layers
    cells, _rep2 = load_readouts(out)
    assert len(cells) == 8
    assert {c.id for c in cells} == {"item1", "item2"}  # dir name, not row label
    assert {c.layer for c in cells} == {20, 36}  # filename, not row layer
    c = next(c for c in cells if c.id == "item1" and c.layer == 20 and c.pos == 3)
    assert c.samples == ("s1", "s2") and c.tokens is None and c.scores is None
    raw = [json.loads(line) for line in out.read_text().splitlines()]
    assert all("scores" not in r and "tokens" not in r for r in raw)


def test_convert_tokens_and_layer_restrict(tmp_path):
    g = _gen_dir(tmp_path)
    out = tmp_path / "tok.jsonl"
    rep = convert_gen_dir(g, out, kind="tokens", layers=[36])
    assert rep.kind == "tokens" and rep.layers == [36]
    assert rep.skipped["malformed"] == 4
    cells, _ = load_readouts(out)
    assert len(cells) == 4 and all(c.layer == 36 for c in cells)
    c = next(c for c in cells if c.id == "item2" and c.pos == 3)
    assert c.tokens == ("s1", "s2") and c.scores == (2.0, 1.0)
    c = next(c for c in cells if c.id == "item2" and c.pos == 4)
    assert c.tokens == ("s3",) and c.scores is None


def test_invalid_utf8_line_is_malformed_not_fatal(tmp_path):
    from wsbench.readouts import load_readouts

    f = tmp_path / "r.jsonl"
    good = b'{"id": "a", "layer": 1, "pos": 0, "samples": ["x"]}\n'
    f.write_bytes(good + b'{"id": "b", "layer": 1, "pos": 0, "samples": ["\xff\xfe"]}\n')
    cells, rep = load_readouts(f)
    assert len(cells) == 2 and rep.skipped["malformed"] == 0
    assert "\ufffd" in cells[1].samples[0]


def test_token_field_is_optional(tmp_path):
    p = _write(
        tmp_path / "r.jsonl",
        [
            {"id": "a", "layer": 1, "pos": 0, "samples": ["x"], "token": "'s"},
            {"id": "a", "layer": 1, "pos": 1, "samples": ["x"]},
            {"id": "a", "layer": 1, "pos": 2, "samples": ["x"], "token": 7},
        ],
    )
    cells, rep = load_readouts(p)
    assert [c.token for c in cells] == ["'s", None]
    assert rep.skipped["malformed"] == 1
    assert Cell("a", 1, 0, ("x",), None, None).token is None


def test_convert_passes_token_through(tmp_path):
    g = tmp_path / "gen" / "item1"
    g.mkdir(parents=True)
    (g / "L036.jsonl").write_text(
        json.dumps({"pos": 2, "samples": ["s"], "token": "'s"})
        + "\n"
        + json.dumps({"pos": 3, "samples": ["t"]})
        + "\n"
    )
    out = tmp_path / "c.jsonl"
    convert_gen_dir(tmp_path / "gen", out, kind="tokens")
    cells, _ = load_readouts(out)
    assert [c.token for c in cells] == ["'s", None]
