"""Residual capture and activation editing on the subject model (Qwen3.6-27B).

Layer convention (K0, experiments/00_nla_gate): "layer L" = OUTPUT of decoder block L =
``model.layers[L]`` forward-hook output = HF ``output_hidden_states[L+1]``. Every function here
hooks ``model.layers[L]`` directly, so callers pass L (40/42/44), never L+1.

Two editing modes (prereg §4.4):
- arm R (headline, read-layer): no forward pass after the edit. Capture h once, then build
  steered vectors arithmetically with :func:`steer_vectors` / :func:`clamp_vectors` /
  :func:`read_layer_steer` and hand them to the readers.
- arm U (validation, upstream): :func:`steer` / :func:`clamp` context managers add a vector at
  block-L output during the model's own forward pass, then capture downstream layers.

Batching: sequences are RIGHT-padded (the Gated DeltaNet layers carry state left to right, so
left padding would contaminate real tokens). Positions may be negative (relative to each
sequence's true length). Per-row edit vectors/positions stay aligned with rows across
mini-batches through a shared batch context set by :func:`capture` / :func:`logits_at`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch

Positions = Sequence[Sequence[int]]
_BATCH: dict[str, Any] = {"rows": None, "lengths": None}


class _StopForward(Exception):
    """Raised by the last capture hook to skip the rest of the forward pass."""


# ----------------------------------------------------------------------------- model helpers


def decoder_blocks(model: Any) -> torch.nn.ModuleList:
    """The decoder-block ModuleList (works through PeftModel / multimodal wrappers)."""
    n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    cands = [(name, m) for name, m in model.named_modules()
             if name.endswith("layers") and isinstance(m, torch.nn.ModuleList)]
    if n is not None:
        cands = [(name, m) for name, m in cands if len(m) == n] or cands
    if not cands:
        raise AttributeError("decoder blocks not found")
    return min(cands, key=lambda t: t[0].count("."))[1]


def _device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _hidden(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def _replace_hidden(out: Any, h: torch.Tensor) -> Any:
    return (h, *out[1:]) if isinstance(out, tuple) else h


def _resolve(positions: Sequence[int], length: int) -> list[int]:
    res = []
    for p in positions:
        q = p + length if p < 0 else p
        if not 0 <= q < length:
            raise IndexError(f"position {p} outside a sequence of length {length}")
        res.append(q)
    return res


def _pad(seqs: Sequence[Sequence[int]], pad_id: int, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.as_tensor(list(s), dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


@contextmanager
def _batch_context(rows: list[int], lengths: list[int]) -> Iterator[None]:
    prev = dict(_BATCH)
    _BATCH.update(rows=rows, lengths=lengths)
    try:
        yield
    finally:
        _BATCH.update(prev)


def _batches(n: int, size: int) -> Iterator[list[int]]:
    for s in range(0, n, size):
        yield list(range(s, min(n, s + size)))


# ----------------------------------------------------------------------------- capture


@torch.no_grad()
def capture(
    model: Any,
    seqs: Sequence[Sequence[int]],
    layers: Sequence[int],
    positions: Positions,
    batch_size: int = 8,
    stop_after: bool = True,
    pad_id: int = 0,
) -> dict[int, torch.Tensor]:
    """Block-output residuals {L: [N, P, d] fp32 CPU} at ``positions[i]`` of sequence i.

    Every row needs the same number of positions P. ``stop_after`` aborts the forward pass
    after the deepest requested block (no lm_head, no later blocks); it changes nothing
    numerically (causal model) and is what makes capture cheap.
    """
    if len(seqs) != len(positions):
        raise ValueError("one positions list per sequence")
    resolved = [_resolve(p, len(s)) for p, s in zip(positions, seqs)]
    if len({len(p) for p in resolved}) > 1:
        raise ValueError("every sequence needs the same number of positions")
    blocks = decoder_blocks(model)
    layers = sorted(set(layers))
    last = layers[-1]
    out: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    cur: dict[int, torch.Tensor] = {}

    def make_hook(L: int) -> Callable:
        def hook(_m: Any, _i: Any, o: Any) -> None:
            cur[L] = _hidden(o).detach()
            if stop_after and L == last:
                raise _StopForward

        return hook

    handles = [blocks[L].register_forward_hook(make_hook(L)) for L in layers]
    dev = _device(model)
    try:
        for rows in _batches(len(seqs), batch_size):
            ids, mask = _pad([seqs[r] for r in rows], pad_id, dev)
            with _batch_context(rows, [len(seqs[r]) for r in rows]):
                try:
                    model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
                except _StopForward:
                    pass
            pos = torch.tensor([resolved[r] for r in rows], device=dev)
            ar = torch.arange(len(rows), device=dev)[:, None]
            for L in layers:
                out[L].append(cur[L][ar, pos].float().cpu())
    finally:
        for h in handles:
            h.remove()
    return {L: torch.cat(v) for L, v in out.items()}


@torch.no_grad()
def logits_at(
    model: Any,
    seqs: Sequence[Sequence[int]],
    positions: Positions,
    batch_size: int = 8,
    pad_id: int = 0,
) -> torch.Tensor:
    """Next-token logits [N, P, V] (fp32 CPU) at the given positions (edits in scope apply)."""
    resolved = [_resolve(p, len(s)) for p, s in zip(positions, seqs)]
    dev = _device(model)
    res = []
    for rows in _batches(len(seqs), batch_size):
        ids, mask = _pad([seqs[r] for r in rows], pad_id, dev)
        with _batch_context(rows, [len(seqs[r]) for r in rows]):
            lg = model(input_ids=ids, attention_mask=mask).logits
        pos = torch.tensor([resolved[r] for r in rows], device=dev)
        ar = torch.arange(len(rows), device=dev)[:, None]
        res.append(lg[ar, pos].float().cpu())
    return torch.cat(res)


def next_token_kl(logits_clean: torch.Tensor, logits_edit: torch.Tensor) -> torch.Tensor:
    """KL(clean || edited) of the next-token distributions, per row (last dim = vocab)."""
    lp = torch.log_softmax(logits_clean.float(), -1)
    lq = torch.log_softmax(logits_edit.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


# ----------------------------------------------------------------------------- arm U hooks


def _row_vectors(vectors: torch.Tensor, rows: list[int]) -> torch.Tensor:
    return vectors[None].expand(len(rows), -1) if vectors.ndim == 1 else vectors[rows]


@contextmanager
def edit(
    model: Any,
    layer: int,
    fn: Callable[[torch.Tensor, int], torch.Tensor],
    positions: Positions | None = None,
) -> Iterator[None]:
    """Generic block-output edit: ``h[row, p] <- fn(h[row, p], global_row)`` at ``positions``.

    ``positions=None`` edits every position, including KV-cache decode steps; otherwise the edit
    applies only on passes that contain the whole prompt (prefill), as in the NLA hook.
    """
    blocks = decoder_blocks(model)

    def hook(_m: Any, _i: Any, o: Any) -> Any:
        h = _hidden(o)
        rows = _BATCH["rows"] or list(range(h.shape[0]))
        lengths = _BATCH["lengths"] or [h.shape[1]] * h.shape[0]
        h = h.clone()
        for b, r in enumerate(rows):
            if positions is None:
                ps = list(range(h.shape[1]))
            else:
                if h.shape[1] < lengths[b]:  # decode step with cache: prompt already edited
                    continue
                ps = _resolve(positions[r], lengths[b])
            h[b, ps] = fn(h[b, ps].float(), r).to(h.dtype)
        return _replace_hidden(o, h)

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def steer(
    model: Any,
    layer: int,
    vectors: torch.Tensor,
    positions: Positions | None,
    dose: float = 1.0,
) -> Iterator[None]:
    """Add ``dose * v`` at block-``layer`` output. ``vectors`` is [d] (shared) or [N, d] (row i
    of the full sequence list passed to capture/logits_at gets vectors[i])."""
    vec = vectors.float()

    def fn(h: torch.Tensor, r: int) -> torch.Tensor:
        v = vec if vec.ndim == 1 else vec[r]
        return h + dose * v.to(h.device)

    with edit(model, layer, fn, positions):
        yield


@contextmanager
def clamp(
    model: Any,
    layer: int,
    direction: torch.Tensor,
    value: float | torch.Tensor,
    positions: Positions | None,
) -> Iterator[None]:
    """Clamp the projection on ``direction`` ([d] unit, or orthonormal [d, k]) to ``value``
    (FollowText, prereg §4.4: h' = h + (c - u.h) u)."""
    def fn(h: torch.Tensor, _r: int) -> torch.Tensor:
        return clamp_vectors(h, direction.to(h.device), value)

    with edit(model, layer, fn, positions):
        yield


# ----------------------------------------------------------------------------- arm R (pure)


def steer_vectors(h: torch.Tensor, u: torch.Tensor, doses: Sequence[float]) -> torch.Tensor:
    """[n_doses, N, d]: h + dose * u for each signed dose (u is used as given; pass unit u)."""
    hf, uf = h.float(), u.float().to(h.device)
    return torch.stack([hf + float(a) * uf for a in doses])


def read_layer_steer(
    h: torch.Tensor, u: torch.Tensor, doses: Sequence[float], signs: Sequence[int] = (1, -1)
) -> dict[tuple[int, float], torch.Tensor]:
    """Arm R helper: {(sign, dose): steered activations [N, d]} plus {(0, 0.0): h}, ready to
    hand to a reader (no forward pass after the edit)."""
    out = {(0, 0.0): h.float()}
    for d in doses:
        for s in signs:
            out[(int(s), float(d))] = h.float() + s * float(d) * u.float().to(h.device)
    return out


def clamp_vectors(h: torch.Tensor, direction: torch.Tensor, value: float | torch.Tensor) -> torch.Tensor:
    """Set the projection of rows of h on ``direction`` to ``value``.

    direction [d] (unit) -> value scalar; direction [d, k] (orthonormal columns) -> value [k].
    """
    hf = h.float()
    U = direction.float().to(hf.device)
    if U.ndim == 1:
        return hf + (float(value) - hf @ U)[..., None] * U
    c = torch.as_tensor(value, dtype=torch.float32, device=hf.device)
    return hf + (c - hf @ U) @ U.T


def doses_from_projection(
    proj: torch.Tensor | Sequence[float],
    quantiles: Sequence[float] = (0.9, 0.99),
    extra_multiple: float | None = 2.0,
) -> list[float]:
    """Prereg §4.4 doses from the natural projection p = u.h over the item bank:
    [q90 - m, q99 - m, 2 (q99 - m)] with m the median (linear-interpolated quantiles)."""
    p = torch.as_tensor(proj, dtype=torch.float64).ravel()
    m = float(torch.quantile(p, 0.5))
    d = [float(torch.quantile(p, q)) - m for q in quantiles]
    if extra_multiple is not None:
        d.append(extra_multiple * d[-1])
    return d
