"""CPU smoke test of the exp 30 training path on a tiny stand-in model: the same `nll_of_targets` + cloning injection
hook + backward that the H200 stage runs, on a 3-layer random Qwen2 (the injection sits at block 1, so a later block must exist) with a character tokenizer. Proves (1) autograd
flows through the injection (gradient reaches the injected activation and the trainable weights), (2) the cloning hook
is numerically the in-place hook, (3) the loss is the gradient twin of `logprob_of_strings`, (4) gradient checkpointing
recomputes WITH the injection, (5) the loop in train30.py logs, saves, resumes and lowers the loss. Runs in seconds."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from mats.readers import nla as NL  # noqa: E402
import train30 as T  # noqa: E402

VOCAB, D = 64, 32
MARKER = 3


class CharTok:
    """Character tokenizer with the three attributes the NLA code reads."""
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [2 + (ord(c) % (VOCAB - 2)) for c in text]


def _tiny(seed: int = 0):  # noqa: ANN202
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=D, intermediate_size=2 * D, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg)


def _nla(model, hook: str = "grad", checkpointing: bool = False) -> NL.NLA:  # noqa: ANN001
    nla = NL.NLA(av=model, tok=CharTok(), ar=None, value_head=None, meta={}, prompt_ids=[5, 6, 7, 8, 9, 10],
                 marker=MARKER, device="cpu")
    if hook == "grad":
        NL.make_trainable(nla, param_filter=lambda n: "q_proj" in n, checkpointing=checkpointing)
    else:
        model.model.layers[1].register_forward_hook(nla._hook)
    return nla


def _batch(n: int = 2):  # noqa: ANN202
    g = torch.Generator().manual_seed(1)
    acts = torch.randn(n, D, generator=g) * 4
    targets = ["ab cd.\n</explanation>", "inside a string.\n</explanation>"][:n]
    return acts, targets


# ---- 1. autograd flows through the injection ---------------------------------------------------------------------

def test_gradient_reaches_the_injected_activation_and_only_the_trainable_weights():
    nla = _nla(_tiny())
    acts, targets = _batch()
    acts.requires_grad_(True)
    loss = NL.nll_of_targets(nla, acts, targets)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert acts.grad is not None and float(acts.grad.norm()) > 0          # the hook is in the graph
    grads = {n: p.grad for n, p in nla.av.named_parameters()}
    assert all(g is not None and float(g.norm()) > 0 for n, g in grads.items() if "q_proj" in n)
    assert all(g is None for n, g in grads.items() if "q_proj" not in n)


def test_the_loss_depends_on_the_injected_vector():
    nla = _nla(_tiny())
    acts, targets = _batch()
    with torch.no_grad():
        a = float(NL.nll_of_targets(nla, acts, targets))
        b = float(NL.nll_of_targets(nla, -acts, targets))
    assert a != pytest.approx(b)


# ---- 2. the cloning hook is the in-place hook --------------------------------------------------------------------

def test_cloning_hook_matches_the_in_place_hook_numerically():
    m1 = _tiny()
    m2 = copy.deepcopy(m1)
    n1, n2 = _nla(m1, "grad"), _nla(m2, "inplace")
    acts, targets = _batch()
    with torch.no_grad():
        a = NL.nll_of_targets(n1, acts, targets)
        b = NL.nll_of_targets(n2, acts, targets)
    assert torch.allclose(a, b, atol=1e-6)


# ---- 3. gradient twin of logprob_of_strings ------------------------------------------------------------------------

def test_nll_is_the_mean_token_nll_of_logprob_of_strings():
    nla = _nla(_tiny())
    acts, targets = _batch()
    with torch.no_grad():
        loss = float(NL.nll_of_targets(nla, acts, targets, add_eos=False))
        lp = [float(nla.logprob_of_strings(acts[i], [t])[0]) for i, t in enumerate(targets)]
    n_tok = sum(len(nla.tok.encode(t)) for t in targets)
    assert loss == pytest.approx(-sum(lp) / n_tok, rel=1e-4)


def test_eos_is_scored_by_default():
    nla = _nla(_tiny())
    acts, targets = _batch()
    with torch.no_grad():
        a = float(NL.nll_of_targets(nla, acts, targets, add_eos=True))
        b = float(NL.nll_of_targets(nla, acts, targets, add_eos=False))
    assert a != pytest.approx(b)


# ---- 4. gradient checkpointing recomputes with the injection -------------------------------------------------------

def test_checkpointed_backward_inside_the_injection_matches_the_plain_backward():
    m1 = _tiny()
    m2 = copy.deepcopy(m1)
    plain, ckpt = _nla(m1, checkpointing=False), _nla(m2, checkpointing=True)
    acts, targets = _batch()
    NL.nll_of_targets(plain, acts, targets).backward()
    l2 = NL.nll_of_targets(ckpt, acts, targets, backward=True)
    assert not l2.requires_grad
    g1 = {n: p.grad for n, p in plain.av.named_parameters() if p.grad is not None}
    g2 = {n: p.grad for n, p in ckpt.av.named_parameters() if p.grad is not None}
    assert g1.keys() == g2.keys() and g1
    for n in g1:
        assert torch.allclose(g1[n], g2[n], atol=1e-5, rtol=1e-4), n


def test_checkpointing_refuses_a_backward_outside_the_injection():
    nla = _nla(_tiny(), checkpointing=True)
    acts, targets = _batch()
    with pytest.raises(RuntimeError, match="backward=True"):
        NL.nll_of_targets(nla, acts, targets)


# ---- 5. the training loop ----------------------------------------------------------------------------------------

def test_batch_indices_cycle_every_item_once_per_epoch_deterministically():
    seen = [i for s in range(5) for i in T.batch_indices(10, 2, s, seed=30)]
    assert sorted(seen) == list(range(10))
    assert T.batch_indices(10, 2, 7, seed=30) == T.batch_indices(10, 2, 7, seed=30)
    assert sorted(i for s in range(5, 10) for i in T.batch_indices(10, 2, s, seed=30)) == list(range(10))


def test_train_logs_saves_resumes_and_lowers_the_loss(tmp_path: Path):
    nla = _nla(_tiny())
    acts, targets = _batch()
    out = str(tmp_path)
    st = T.train(nla, acts, targets, out, "S", steps=6, batch=2, micro=1, lr=1e-2, warmup=1, log_every=1, save_every=3, seed=30)
    assert st["step"] == 6 and st["done"] and len(st["losses"]) == 6
    assert st["losses"][-1][1] < st["losses"][0][1]
    state = json.loads(Path(out, "train30_S.json").read_text())
    assert state["step"] == 6 and Path(out, "adapter_S", "optim.pt").exists()
    assert Path(state["adapter"]).exists()
    st2 = T.train(nla, acts, targets, out, "S", steps=8, batch=2, micro=1, lr=1e-2, warmup=1, log_every=1, save_every=3, seed=30)
    assert st2["step"] == 8 and len(st2["losses"]) == 8 and st2["losses"][:6] == st["losses"]
    assert T.train(nla, acts, targets, out, "S", steps=8, batch=2)["done"]      # already done: nothing happens


def test_train_with_checkpointing_and_accumulation_runs(tmp_path: Path):
    nla = _nla(_tiny(), checkpointing=True)
    acts, targets = _batch()
    st = T.train(nla, acts, targets, str(tmp_path), "C", steps=2, batch=2, micro=1, lr=1e-3, log_every=1, save_every=100)
    assert st["step"] == 2 and all(torch.isfinite(torch.tensor(v)) for _, v in st["losses"])
