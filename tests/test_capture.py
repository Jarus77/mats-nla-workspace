"""Offline tests for src/mats/capture.py on a tiny fake decoder (no HF download, CPU only)."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from mats import capture as C  # noqa: E402


class Block(nn.Module):
    """Causal toy block: h_t <- h_t + tanh(W h_t) + 0.5 * mean_{s<=t} h_s. Returns a tuple, as HF."""

    def __init__(self, d, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter(torch.randn(d, d, generator=g) / d**0.5)

    def forward(self, h, attention_mask=None):
        cum = torch.cumsum(h, dim=1) / torch.arange(1, h.shape[1] + 1).view(1, -1, 1)
        return (h + torch.tanh(h @ self.w) + 0.5 * cum,)


class Inner(nn.Module):
    def __init__(self, d, n_layers, vocab):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([Block(d, i) for i in range(n_layers)])
        self.norm = nn.Identity()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **_):
        h = self.embed_tokens(input_ids)
        hs = [h]
        for blk in self.layers:
            h = blk(h)[0]
            hs.append(h)
        return type("O", (), {"last_hidden_state": h, "hidden_states": tuple(hs)})()


class Toy(nn.Module):
    def __init__(self, d=16, n_layers=6, vocab=50):
        super().__init__()
        torch.manual_seed(0)
        self.model = Inner(d, n_layers, vocab)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.config = type("Cfg", (), {"num_hidden_layers": n_layers, "hidden_size": d})()

    def forward(self, input_ids, attention_mask=None, **kw):
        out = self.model(input_ids, attention_mask=attention_mask, **kw)
        return type("O", (), {"logits": self.lm_head(out.last_hidden_state)})()


SEQS = [[1, 2, 3, 4, 5], [7, 8, 9], [10, 11, 12, 13, 14, 15, 16]]


def _hs(model, seq):
    with torch.no_grad():
        return model.model(torch.tensor([seq]), output_hidden_states=True).hidden_states


def test_decoder_blocks_found():
    m = Toy()
    assert len(C.decoder_blocks(m)) == 6


def test_capture_matches_hidden_states_plus_one_with_padding():
    m = Toy()
    out = C.capture(m, SEQS, layers=[1, 4], positions=[[-1, -2], [-1, 0], [2, -1]], batch_size=2)
    assert set(out) == {1, 4}
    assert out[1].shape == (3, 2, 16)
    for i, (seq, pos) in enumerate(zip(SEQS, [[-1, -2], [-1, 0], [2, -1]])):
        hs = _hs(m, seq)
        for L in (1, 4):
            for j, p in enumerate(pos):
                ref = hs[L + 1][0, p % len(seq)]
                assert torch.allclose(out[L][i, j], ref, atol=1e-5), (i, L, p)


def test_capture_early_exit_does_not_change_values():
    m = Toy()
    a = C.capture(m, SEQS, layers=[2], positions=[[-1]] * 3, stop_after=True)
    b = C.capture(m, SEQS, layers=[2], positions=[[-1]] * 3, stop_after=False)
    assert torch.allclose(a[2], b[2])


def test_capture_rejects_out_of_range_positions():
    m = Toy()
    with pytest.raises(IndexError):
        C.capture(m, SEQS, layers=[1], positions=[[-1], [5], [-1]])


def test_steer_hook_adds_vector_at_positions_only():
    m = Toy()
    v = torch.randn(16)
    seq = SEQS[0]
    base = _hs(m, seq)
    with C.steer(m, layer=2, vectors=v, positions=[[3]], dose=2.0, preserve_norm=False):
        st = _hs(m, seq)
    assert torch.allclose(st[3][0, 3], base[3][0, 3] + 2.0 * v, atol=1e-5)
    assert torch.allclose(st[3][0, :3], base[3][0, :3], atol=1e-6)  # earlier positions intact
    assert not torch.allclose(st[5][0, 4], base[5][0, 4])  # propagates downstream (causal)
    after = _hs(m, seq)  # hook removed on exit
    assert torch.allclose(after[3], base[3])


def test_steer_per_row_vectors_batched_capture():
    m = Toy()
    vs = torch.randn(3, 16)
    with C.steer(m, layer=1, vectors=vs, positions=[[-1], [-1], [-1]], dose=1.0,
                 preserve_norm=False):
        got = C.capture(m, SEQS, layers=[1], positions=[[-1]] * 3, batch_size=3)
    clean = C.capture(m, SEQS, layers=[1], positions=[[-1]] * 3)
    assert torch.allclose(got[1][:, 0], clean[1][:, 0] + vs, atol=1e-5)


def test_clamp_hook_sets_projection():
    m = Toy()
    u = torch.randn(16)
    u = u / u.norm()
    with C.clamp(m, layer=3, direction=u, value=0.7, positions=[[-1]] * 3):
        got = C.capture(m, SEQS, layers=[3], positions=[[-1]] * 3)
    assert torch.allclose(got[3][:, 0] @ u, torch.full((3,), 0.7), atol=1e-5)


def test_pure_steer_and_clamp_vectors():
    h = torch.randn(4, 8)
    u = torch.zeros(8)
    u[0] = 1.0
    out = C.steer_vectors(h, u, [-1.0, 0.0, 2.0], preserve_norm=False)
    assert out.shape == (3, 4, 8)
    assert torch.allclose(out[2, :, 0], h[:, 0] + 2.0)
    assert torch.allclose(out[1], h)
    cl = C.clamp_vectors(h, u, 0.3)
    assert torch.allclose(cl[:, 0], torch.full((4,), 0.3))
    assert torch.allclose(cl[:, 1:], h[:, 1:])
    # subspace clamp
    U = torch.eye(8)[:, :2]
    cl2 = C.clamp_vectors(h, U, torch.tensor([0.1, 0.2]))
    assert torch.allclose(cl2[:, :2], torch.tensor([[0.1, 0.2]]).expand(4, 2))


def test_doses_from_projection():
    p = torch.arange(101.0)
    d = C.doses_from_projection(p, quantiles=(0.9, 0.99), extra_multiple=2.0)
    assert d == pytest.approx([40.0, 49.0, 98.0])


def test_next_token_kl_zero_for_identical():
    lg = torch.randn(3, 10)
    assert torch.allclose(C.next_token_kl(lg, lg), torch.zeros(3), atol=1e-6)
    assert (C.next_token_kl(lg, lg + torch.randn(3, 10)) > 0).all()


def test_logits_at_positions():
    m = Toy()
    lg = C.logits_at(m, SEQS, positions=[[-1]] * 3)
    assert lg.shape == (3, 1, 50)
    with torch.no_grad():
        ref = m(torch.tensor([SEQS[1]])).logits[0, -1]
    assert torch.allclose(lg[1, 0], ref, atol=1e-5)


def test_steer_vectors_preserve_norm_default():
    h = torch.randn(5, 8)
    u = torch.randn(8)
    u = u / u.norm()
    out = C.steer_vectors(h, u, [-3.0, 3.0])  # default: norm-preserving
    assert torch.allclose(out.norm(dim=-1), h.norm(dim=-1).expand(2, 5), atol=1e-5)
    raw = C.steer_vectors(h, u, [-3.0, 3.0], preserve_norm=False)
    cos = torch.nn.functional.cosine_similarity(out, raw, dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-6)  # same direction


def test_read_layer_steer_preserve_flag():
    h = torch.randn(4, 8)
    u = torch.eye(8)[0]
    d = C.read_layer_steer(h, u, [2.0])
    assert torch.allclose(d[(1, 2.0)].norm(dim=-1), h.norm(dim=-1), atol=1e-5)
    d2 = C.read_layer_steer(h, u, [2.0], preserve_norm=False)
    assert torch.allclose(d2[(1, 2.0)], h + 2.0 * u)


def test_centred_steer_equal_norms_on_both_sides():
    h = torch.randn(6, 8) + 3.0 * torch.eye(8)[0]  # strong existing projection on u
    u = torch.eye(8)[0]
    plus, minus = C.centred_steer_vectors(h, u, [2.0, -2.0])
    assert torch.allclose(plus.norm(dim=-1), minus.norm(dim=-1), atol=1e-5)
    assert torch.allclose(plus[:, 0], torch.full((6,), 2.0))
    assert torch.allclose(plus[:, 1:], h[:, 1:])


def test_steer_hook_preserve_norm_default():
    m = Toy()
    v = torch.randn(16) * 5
    seq = SEQS[0]
    base = _hs(m, seq)
    with C.steer(m, layer=2, vectors=v, positions=[[3]], dose=1.0):
        st = _hs(m, seq)
    assert torch.allclose(st[3][0, 3].norm(), base[3][0, 3].norm(), atol=1e-4)
    target = base[3][0, 3] + v
    assert float(torch.nn.functional.cosine_similarity(st[3][0, 3], target, dim=0)) > 0.9999
