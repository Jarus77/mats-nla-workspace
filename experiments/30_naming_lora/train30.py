"""Exp 30 training loop (imported by modal30's train stage; exercised on CPU by test_train30_smoke.py).

One arm per call: AdamW on the AV's trainable parameters (the LoRA adapter, see nla.make_trainable), loss = mean token
NLL of the target explanation given the injected activation (nla.nll_of_targets, backward inside the injection so
gradient-checkpoint recomputes see the vectors), bf16 autocast on CUDA, gradient accumulation over ``micro``-sized
chunks of each ``batch``, linear warmup then constant lr, grad clip 1.0. Every ``save_every`` steps and at the end the
adapter, the optimizer state and the loss log are written, and a later call with the same ``out`` resumes from the log's
step (the caller reloads the saved adapter; this loop reloads the optimizer state). Batches are a deterministic function
of (seed, step), so a resumed run sees the same batches it would have seen."""
from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from mats.readers import nla as NL


def batch_indices(n: int, batch: int, step: int, seed: int) -> list[int]:
    """Items of global step ``step``: a seeded permutation per epoch, consumed ``batch`` at a time."""
    per_epoch = max(1, n // batch)
    epoch, pos = divmod(step, per_epoch)
    perm = list(range(n))
    random.Random(seed + epoch).shuffle(perm)
    return perm[pos * batch : (pos + 1) * batch]


def _state_path(out: str, arm: str) -> Path:
    return Path(out) / f"train30_{arm}.json"


def load_state(out: str, arm: str) -> dict | None:
    p = _state_path(out, arm)
    return json.loads(p.read_text()) if p.exists() else None


def _save(nla: NL.NLA, opt: torch.optim.Optimizer, out: str, arm: str, state: dict, on_save: Callable | None) -> dict:
    adir = Path(out) / f"adapter_{arm}"
    adir.mkdir(parents=True, exist_ok=True)
    path = NL.save_adapter(nla, adir)
    torch.save(opt.state_dict(), adir / "optim.pt")
    new = {**state, "adapter": path, "saved_at": time.time()}
    _state_path(out, arm).write_text(json.dumps(new, indent=1))
    if on_save:
        on_save()
    return new


def train(nla: NL.NLA, acts: torch.Tensor, targets: Sequence[str], out: str, arm: str, steps: int = 400, batch: int = 8,
          micro: int = 8, lr: float = 1e-5, warmup: int = 20, clip: float = 1.0, log_every: int = 10,
          save_every: int = 100, seed: int = 30, on_save: Callable | None = None) -> dict:
    """Train ``nla`` on (acts[i], targets[i]) pairs; returns the final state dict (see module docstring)."""
    if acts.shape[0] != len(targets):
        raise ValueError(f"{acts.shape[0]} activations for {len(targets)} targets")
    params = [p for p in nla.av.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters: call load_nla(trainable=True) / make_trainable first")
    state = load_state(out, arm) or {"arm": arm, "step": 0, "steps": steps, "losses": [], "done": False,
                                     "n_items": len(targets), "batch": batch, "lr": lr, "seed": seed, "wall_s": 0.0}
    if state["step"] >= steps:
        return {**state, "done": True}
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
    optim_path = Path(out) / f"adapter_{arm}" / "optim.pt"
    if state["step"] > 0 and optim_path.exists():
        opt.load_state_dict(torch.load(optim_path, map_location=nla.device, weights_only=False))
    cuda = str(nla.device).startswith("cuda")
    t0 = time.time()
    losses = list(state["losses"])
    for step in range(state["step"], steps):
        idx = batch_indices(len(targets), batch, step, seed)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / max(1, warmup))
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for s in range(0, len(idx), micro):
            chunk = idx[s : s + micro]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = NL.nll_of_targets(nla, acts[chunk], [targets[i] for i in chunk], backward=True,
                                         scale=len(chunk) / len(idx))
            total += float(loss) * len(chunk) / len(idx)
        if clip:
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        done = step + 1
        if done % log_every == 0 or done == steps:
            losses.append([done, total])
        state = {**state, "step": done, "losses": losses, "done": done == steps,
                 "wall_s": round(state["wall_s"] + time.time() - t0, 1)}
        if done % save_every == 0 or done == steps:
            state = _save(nla, opt, out, arm, state, on_save)
            t0 = time.time()
    return state
