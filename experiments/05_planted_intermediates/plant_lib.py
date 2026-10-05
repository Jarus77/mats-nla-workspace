"""Planting machinery for (C): batched forward with a per-row norm-preserving edit at one block,
greedy-consistency scoring of answer aliases, block-output capture, bridge directions, partners.

Layer convention (K0): "layer L" = output of decoder block L = hidden_states[L+1].
Edit (D2.5, D9.1): h' = (h + v) * ||h|| / ||h + v|| at the given positions of block-L output,
with v = alpha * (mu_Y - mu_X) (class means of the bridge, leave-own-clue-out).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from mats.capture import decoder_blocks


# ----------------------------------------------------------------------------- tokenisation


def alias_ids(tok, prompt: str, alias: str) -> list[int]:
    text = alias if prompt.endswith(" ") else " " + alias
    return tok.encode(text, add_special_tokens=False)


def first_tokens(tok, prompt: str, aliases: Sequence[str]) -> set[int]:
    return {alias_ids(tok, prompt, a)[0] for a in aliases}


# ----------------------------------------------------------------------------- forward


@dataclass
class Row:
    """One sequence = prompt (+ optional continuation). ``edit_pos`` absolute positions."""

    ids: list[int]
    plen: int  # prompt length; continuation tokens are ids[plen:]
    vec: int  # index into the edit-vector table (-1 = no edit)
    edit_pos: tuple[int, ...]
    cap_pos: int  # absolute position to capture at cap layers (-1 = none)
    ref: int = -1  # index of a reference next-token distribution for KL at plen-1


@torch.no_grad()
def run_rows(
    model, rows: Sequence[Row], vecs: torch.Tensor | None, edit_layer: int | None,
    cap_layers: Sequence[int] = (), watch_ids: torch.Tensor | None = None,
    ref_lp: torch.Tensor | None = None, batch: int = 64, pad_id: int = 0,
    targets: torch.Tensor | None = None, renorm: bool = True,
) -> dict[str, torch.Tensor]:
    """Forward every row once (right-padded). Returns per row:
    greedy (all continuation tokens are argmax), cont_lp (summed continuation log-prob),
    top1 / p_top1 at plen-1, watch_lp [R, W] (log-probs of ``watch_ids`` at plen-1),
    kl (KL(ref || edited) at plen-1 if row.ref >= 0, else nan), caps {L: [R, d]}.

    Edit modes: ``targets is None`` -> additive, norm-preserving h' = (h+v)||h||/||h+v||;
    else centred swap (clamp) with unit u = vecs[k]: h' = h + (t_k - u.h) u (the item's own
    u-projection is removed and set to t_k; D11 equal-dilution construction)."""
    blocks = decoder_blocks(model)
    dev = next(model.parameters()).device
    R = len(rows)
    out = {"greedy": torch.zeros(R, dtype=torch.bool), "cont_lp": torch.zeros(R),
           "top1": torch.zeros(R, dtype=torch.long), "p_top1": torch.zeros(R),
           "kl": torch.full((R,), float("nan"))}
    if watch_ids is not None:
        out["watch_lp"] = torch.zeros(R, len(watch_ids))
        watch_ids = watch_ids.to(dev)
    caps: dict[int, torch.Tensor] = {}
    state: dict = {}

    def edit_hook(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        mask, V = state["mask"], state["V"]
        if mask is None:
            return o
        hf = h.float()
        if state.get("T") is not None:
            u = V[:, None, :]
            new = hf + (state["T"][:, None, None] - (hf * u).sum(-1, keepdim=True)) * u
        else:
            new = hf + mask[..., None] * V[:, None, :]
            if renorm:
                new = new * (hf.norm(dim=-1, keepdim=True) / new.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        hf = torch.where(mask[..., None] > 0, new, hf)
        h2 = hf.to(h.dtype)
        return (h2, *o[1:]) if isinstance(o, tuple) else h2

    def cap_hook(L):
        def f(_m, _i, o):
            state[f"cap{L}"] = (o[0] if isinstance(o, tuple) else o).detach()
        return f

    handles = []
    if edit_layer is not None:
        handles.append(blocks[edit_layer].register_forward_hook(edit_hook))
    for L in cap_layers:
        handles.append(blocks[L].register_forward_hook(cap_hook(L)))
    try:
        for s in range(0, R, batch):
            br = rows[s : s + batch]
            B, T = len(br), max(len(r.ids) for r in br)
            ids = torch.full((B, T), pad_id, dtype=torch.long)
            att = torch.zeros((B, T), dtype=torch.long)
            for b, r in enumerate(br):
                ids[b, : len(r.ids)] = torch.tensor(r.ids)
                att[b, : len(r.ids)] = 1
            ids, att = ids.to(dev), att.to(dev)
            if edit_layer is not None and vecs is not None:
                mask = torch.zeros((B, T), device=dev)
                V = torch.zeros((B, vecs.shape[1]), device=dev)
                Tt = torch.zeros(B, device=dev)
                for b, r in enumerate(br):
                    if r.vec >= 0:
                        mask[b, list(r.edit_pos)] = 1.0
                        V[b] = vecs[r.vec].to(dev)
                        if targets is not None:
                            Tt[b] = float(targets[r.vec])
                state["mask"], state["V"] = mask, V
                state["T"] = Tt if targets is not None else None
            else:
                state["mask"], state["V"], state["T"] = None, None, None
            logits = model(input_ids=ids, attention_mask=att).logits
            for b, r in enumerate(br):
                i = s + b
                p = r.plen - 1
                lp_p = torch.log_softmax(logits[b, p].float(), -1)
                t1 = int(lp_p.argmax())
                out["top1"][i], out["p_top1"][i] = t1, float(lp_p[t1].exp())
                if watch_ids is not None:
                    out["watch_lp"][i] = lp_p[watch_ids].cpu()
                if r.ref >= 0 and ref_lp is not None:
                    ref = ref_lp[r.ref].to(dev)
                    out["kl"][i] = float((ref.exp() * (ref - lp_p)).sum())
                if len(r.ids) > r.plen:
                    lg = torch.log_softmax(logits[b, p : len(r.ids) - 1].float(), -1)
                    tgt = ids[b, r.plen : len(r.ids)]
                    out["cont_lp"][i] = float(lg.gather(-1, tgt[:, None]).sum())
                    out["greedy"][i] = bool((lg.argmax(-1) == tgt).all())
                for L in cap_layers:
                    if L not in caps:
                        caps[L] = torch.zeros(R, state[f"cap{L}"].shape[-1], dtype=torch.float16)
                    if r.cap_pos >= 0:
                        caps[L][i] = state[f"cap{L}"][b, r.cap_pos].half().cpu()
    finally:
        for h in handles:
            h.remove()
    out["caps"] = caps
    return out


# ----------------------------------------------------------------------------- directions


def relation_means(acts: torch.Tensor, items: list[dict], use: np.ndarray) -> dict[str, torch.Tensor]:
    rels = sorted({it["relation"] for it in items})
    return {r: acts[[i for i, it in enumerate(items) if it["relation"] == r and use[i]]].mean(0)
            for r in rels}


def centred(acts: torch.Tensor, items: list[dict], rmeans: dict[str, torch.Tensor]) -> torch.Tensor:
    return acts - torch.stack([rmeans[it["relation"]] for it in items])


def class_mean(cen: torch.Tensor, items: list[dict], use: np.ndarray, country: str,
               exclude_clue: int | None) -> torch.Tensor:
    idx = [i for i, it in enumerate(items) if use[i] and it["source"] == "cons"
           and it["country"] == country and (exclude_clue is None or it["clue"] != exclude_clue)]
    return cen[idx].mean(0)


def item_means(cen: torch.Tensor, items: list[dict], use: np.ndarray, i: int, partner: str
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """(mu_X leave-own-clue-out, mu_Y) for item i."""
    it = items[i]
    mx = class_mean(cen, items, use, it["country"], it["clue"] if it["source"] == "cons" else None)
    my = class_mean(cen, items, use, partner, None)
    return mx, my


# ----------------------------------------------------------------------------- partners


def _seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def assign_partners(items: list[dict], gate: np.ndarray, tok, table: dict,
                    min_known: int = 3) -> list[str | None]:
    """One pre-assigned partner country per gated item (seeded by item id; chosen before any
    planting, never by outcome). Partner must: differ from X; have the relation; have >=
    ``min_known`` gated constructed prompts in that relation (the model knows Y's answer) and
    >= 8 gated constructed prompts overall (a usable class mean); answer first-token sets
    disjoint from X's."""
    known: dict[tuple[str, str], int] = {}
    tot: dict[str, int] = {}
    for it, g in zip(items, gate):
        if g and it["source"] == "cons":
            known[(it["country"], it["relation"])] = known.get((it["country"], it["relation"]), 0) + 1
            tot[it["country"]] = tot.get(it["country"], 0) + 1
    out: list[str | None] = []
    for it, g in zip(items, gate):
        if not g:
            out.append(None)
            continue
        fx = first_tokens(tok, it["prompt"], it["answers"])
        cands = []
        for c, row in sorted(table.items()):
            from plant_items import answers_of
            ans = answers_of(c, it["relation"])
            if c == it["country"] or ans is None:
                continue
            if known.get((c, it["relation"]), 0) < min_known or tot.get(c, 0) < 8:
                continue
            if first_tokens(tok, it["prompt"], ans) & fx:
                continue
            cands.append(c)
        if not cands:
            out.append(None)
            continue
        rng = np.random.default_rng(_seed(it["id"]))
        out.append(cands[int(rng.integers(len(cands)))])
    return out
