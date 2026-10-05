import json

import pytest

from wsbench.cache import Cache, fingerprint


def test_fingerprint_stable_and_order_sensitive():
    a = fingerprint("m", "v1", {"x": 1, "y": 2})
    assert a == fingerprint("m", "v1", {"y": 2, "x": 1})
    assert len(a) == 16
    assert a != fingerprint("m", "v2", {"x": 1, "y": 2})


def test_put_get_round_trip(tmp_path):
    c = Cache(tmp_path / "cache.jsonl")
    assert len(c) == 0
    c.put("k1", "fp", {"result": {"label": "ok"}, "extra": 1})
    assert len(c) == 1
    row = c.get("k1", "fp")
    assert row is not None and row["result"] == {"label": "ok"} and row["extra"] == 1
    assert row["key"] == "k1" and row["fp"] == "fp" and "ts" in row
    assert c.get("k1", "other") is None
    assert c.get("nope", "fp") is None
    c.close()


def test_reserved_keys_rejected(tmp_path):
    c = Cache(tmp_path / "c.jsonl")
    for bad in ({"key": 1}, {"fp": 1}, {"ts": 1}):
        with pytest.raises(ValueError):
            c.put("k", "fp", {"result": 1, **bad})
    c.close()


def test_failed_rows_skipped_unless_included(tmp_path):
    c = Cache(tmp_path / "c.jsonl")
    c.put("k", "fp", {"result": {"a": 1}})
    c.put("k", "fp", {"result": None})
    c.put("k", "fp", {"note": "no result key"})
    got = c.get("k", "fp")
    assert got is not None and got["result"] == {"a": 1}
    got = c.get("k", "fp", include_failed=True)
    assert got is not None and got.get("note") == "no result key"
    # newest good row wins
    c.put("k", "fp", {"result": {"a": 2}})
    assert c.get("k", "fp")["result"] == {"a": 2}
    c.close()


def test_torn_last_line_and_foreign_lines(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"key": "k", "fp": "fp", "ts": 0, "result": {"a": 1}})
        + "\n"
        + "not json\n"
        + json.dumps({"no_key": True})
        + "\n"
        + '{"key": "k", "fp": "fp", "ts": 1, "result": {"a": 2'  # torn
    )
    c = Cache(p)
    assert len(c) == 1
    assert c.get("k", "fp")["result"] == {"a": 1}
    c.put("k2", "fp", {"result": 3})
    c.close()
    lines = p.read_text().splitlines()
    assert json.loads(lines[-1])["key"] == "k2"
    # the torn line was terminated, so the new row is on its own line
    assert lines[-2].startswith('{"key": "k", "fp": "fp", "ts": 1')


def test_resume_across_instances(tmp_path):
    p = tmp_path / "c.jsonl"
    c = Cache(p)
    c.put("a", "fp1", {"result": 1})
    c.put("a", "fp2", {"result": 2})
    c.close()
    c2 = Cache(p)
    assert len(c2) == 2
    assert c2.get("a", "fp1")["result"] == 1
    assert c2.get("a", "fp2")["result"] == 2
    c2.put("b", "fp1", {"result": 3})
    assert len(c2) == 3
    c2.close()
    assert len(p.read_text().splitlines()) == 3


def test_creates_parent_dir(tmp_path):
    c = Cache(tmp_path / "deep" / "dir" / "c.jsonl")
    c.put("k", "fp", {"result": 1})
    c.close()
    assert (tmp_path / "deep" / "dir" / "c.jsonl").exists()
