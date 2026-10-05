"""Offline tests for the J-Lens reader math (toy W_U / J, fake tokenizer)."""

import pytest

torch = pytest.importorskip("torch")

from mats.readers.jlens import JLensReader, token_variants  # noqa: E402


class FakeTok:
    vocab = {" French": 0, "French": 1, " German": 2, "German": 3, " cat": 4}

    def encode(self, s, add_special_tokens=False):
        return [self.vocab[s]] if s in self.vocab else [9, 9]

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return inv.get(ids[0], "?")


def _reader():
    g = torch.Generator().manual_seed(0)
    W_U = torch.randn(5, 8, generator=g)
    J = {42: torch.randn(8, 8, generator=g)}
    return JLensReader(J=J, W_U=W_U, tok=FakeTok(), device="cpu"), W_U, J[42]


def test_token_variants():
    assert token_variants(FakeTok(), ["French"]) == [0, 1]


def test_cosine_scores_match_definition():
    r, W_U, J = _reader()
    h = torch.randn(3, 8)
    ref = (h @ J.T @ W_U.T) / (W_U @ J).norm(dim=1)
    assert torch.allclose(r.scores(h, 42), ref, atol=1e-5)
    assert torch.allclose(r.scores(h, 42, ids=[2, 4]), ref[:, [2, 4]], atol=1e-5)


def test_paired_logodds_and_linearity():
    r, W_U, J = _reader()
    h = torch.randn(4, 8)
    s = r.scores(h, 42)
    lo = r.paired_logodds(h, 42, ["French"], ["German"])
    assert torch.allclose(lo, s[:, [0, 1]].max(1).values - s[:, [2, 3]].max(1).values, atol=1e-5)
    assert r.paired_logodds(h, 42, [0], [2]).shape == (4,)
    with pytest.raises(KeyError):
        r.scores(h, 40)


def test_readout_topk():
    r, _, _ = _reader()
    out = r.readout(torch.randn(2, 8), 42, k=3)
    assert len(out) == 2 and len(out[0]) == 3
