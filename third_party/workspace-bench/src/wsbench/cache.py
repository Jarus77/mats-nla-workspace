"""Append-only JSONL verdict cache: rows are never rewritten, a changed fingerprint appends."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Self

_RESERVED = ("key", "fp", "ts")


def fingerprint(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class Cache:
    """Rows keyed by ``(key, fp)``; a row is *failed* iff ``row.get("result") is None``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torn = False
        if self.path.exists() and self.path.stat().st_size:
            data = self.path.read_bytes()
            torn = not data.endswith(b"\n")
            for line in data.decode("utf-8", "replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("key"), str):
                    self._rows.setdefault(row["key"], []).append(row)
                    self._n += 1
        self._fh = self.path.open("a", encoding="utf-8")
        if torn:  # terminate a line torn by an interrupted write
            self._fh.write("\n")
            self._fh.flush()

    def get(self, key: str, fp: str, *, include_failed: bool = False) -> dict[str, Any] | None:
        for row in reversed(self._rows.get(key, [])):
            if row.get("fp") != fp:
                continue
            if include_failed or row.get("result") is not None:
                return row
        return None

    def put(self, key: str, fp: str, payload: dict[str, Any]) -> None:
        clash = [k for k in _RESERVED if k in payload]
        if clash:
            raise ValueError(f"payload uses reserved cache keys: {clash}")
        row = {"key": key, "fp": fp, "ts": time.time(), **payload}
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._rows.setdefault(key, []).append(row)
        self._n += 1

    def __len__(self) -> int:
        return self._n

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._fh.close()
