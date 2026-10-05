"""The producer: one model, one method, readouts in the benchmark's contract. Read a single
cell, a prompt over positions and layers, a plan row, or a whole eval set."""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from wsbench import readplan
from wsbench.readplan import GRID, ReadSpec

from .backend import DEFAULT_MODEL, Backend
from .methods import Method, Readout, method
from .render import Rendered, render, render_text

Positions = int | list[int] | dict[str, Any] | str  # -1 | [3, 5] | {"kind": ...} | "all"


@dataclass(frozen=True)
class Row:
    """One readouts-contract row plus the token it was read at."""

    id: str
    layer: int
    pos: int
    token: str
    readout: Readout

    def contract(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "pos": self.pos,
            "token": self.token,
            **self.readout.row(),
        }


def _positions(rule: Positions, tokens: list[str]) -> list[int]:
    """Indices a positions rule selects. Plain ints count from the end when negative and are
    reported as absolute indices; a plan rule is reported as the plan resolves it (negative for
    ``offset_from_end`` and ``all_from_end``, the convention of the banks that use them)."""
    n = len(tokens)
    if isinstance(rule, int | list):
        want = [rule] if isinstance(rule, int) else list(rule)
        bad = [p for p in want if not -n <= p < n]
        if bad:
            raise ValueError(f"positions {bad} out of range for a {n}-token render")
        return [p % n for p in want]
    if isinstance(rule, str):
        rule = {"kind": rule}
    return readplan.resolve(rule, tokens)


@dataclass
class Producer:
    """``Producer.load(model, method)`` once, then ``read`` / ``read_prompt`` / ``read_spec`` /
    ``run_family``; ``use`` swaps the method without reloading the model. Every method returns
    rows in the readouts contract; ``write`` saves them."""

    backend: Backend
    method: Method

    @classmethod
    def load(
        cls,
        model: str = DEFAULT_MODEL,
        method_spec: str | Method = "logit_lens",
        *,
        device: str = "cuda",
        **method_kw: Any,
    ) -> Self:
        m = method(method_spec, **method_kw)  # before the model load, so a typo fails fast
        backend = Backend.load(model, device=device)
        m.bind(backend)
        return cls(backend=backend, method=m)

    def use(self, method_spec: str | Method, **method_kw: Any) -> Self:
        """Switch method on the loaded model (``p.use("jlens")``); returns self."""
        m = method(method_spec, **method_kw)
        m.bind(self.backend)
        self.method = m
        return self

    # ---- the four entry points

    def read(
        self, text: str, pos: int = -1, layer: int | None = None, *, chat: bool = False
    ) -> Row:
        """One cell of one prompt: the readout at ``pos`` (negative counts from the end) after
        block ``layer`` (default: the method's own layer if it has one, else 44)."""
        layers = [layer] if layer is not None else (self.method.layers or [44])
        return self.read_prompt(text, positions=pos, layers=layers, chat=chat)[0]

    def read_prompt(
        self,
        text: str,
        *,
        positions: Positions = -1,
        layers: Iterable[int] | None = None,
        chat: bool = False,
        system: str | None = None,
        item_id: str = "adhoc",
    ) -> list[Row]:
        """A prompt outside any bank, over a positions rule and a layer list (default: the
        method's own layers, else the benchmark grid)."""
        rendered = render_text(text, self.backend.tokenizer, chat=chat, system=system)
        return list(self._read_rendered(item_id, rendered, positions, self._layers(layers, GRID)))

    def read_spec(self, spec: ReadSpec, *, layers: Iterable[int] | None = None) -> list[Row]:
        """A read-plan row: its own render and positions rule; its layers unless the method has
        fixed ones or the caller overrides."""
        rendered = render(spec, self.backend.tokenizer)
        return list(
            self._read_rendered(
                spec.id, rendered, spec.positions, self._layers(layers, spec.layers)
            )
        )

    def run_family(
        self,
        family: str,
        out: Path | str,
        *,
        limit: int = 0,
        layers: Iterable[int] | None = None,
        items: Iterable[str] | None = None,
    ) -> Path:
        """Every item of a family into one readouts JSONL, resumable: an item whose rows are
        already in ``out`` is skipped; each item's rows are written in one call."""
        out = Path(out)
        specs = readplan.plan(family)
        if items is not None:
            want = set(items)
            specs = [s for s in specs if s.id in want]
        specs = specs[: limit or None]
        done = set()
        if out.exists():
            for line in out.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["id"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fh:
            for spec in specs:
                if spec.id in done:
                    continue
                rows = self.read_spec(spec, layers=layers)
                fh.write("".join(json.dumps(r.contract(), ensure_ascii=False) + "\n" for r in rows))
                fh.flush()
        return out

    # ---- internals

    def _layers(self, given: Iterable[int] | None, default: Iterable[int]) -> list[int]:
        if given is not None:
            return list(given)
        return list(self.method.layers or default)

    def _read_rendered(
        self, item_id: str, rendered: Rendered, positions: Positions, layers: list[int]
    ) -> Iterator[Row]:
        pos = _positions(positions, rendered.tokens)
        if not pos:
            return
        idx = [p % len(rendered) for p in pos]  # a plan may report negative offsets
        acts = self.backend.capture(rendered.ids, layers, idx)
        for layer in layers:
            h_all = acts[layer]
            for i, (p, j) in enumerate(zip(pos, idx, strict=True)):
                yield Row(item_id, layer, p, rendered.decoded[j], self.method.read(h_all[i], layer))


def write(rows: Iterable[Row], path: Path | str) -> Path:
    """Rows to a readouts JSONL (one contract row per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.contract(), ensure_ascii=False) + "\n")
    return path
