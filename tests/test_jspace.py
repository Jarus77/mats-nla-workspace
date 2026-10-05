"""Offline unit tests for src/mats/jspace.py (J-space basis, projection, enrichment, controls)."""

import pytest

torch = pytest.importorskip("torch")

from mats import jspace as J  # noqa: E402


def _toy(d=32, V=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_U = torch.randn(V, d, generator=g)
    Jm = torch.randn(d, d, generator=g)
    gain = torch.rand(d, generator=g) + 0.5
    return W_U, Jm, gain


def test_basis_equals_svd_right_singular_vectors():
    W_U, Jm, gain = _toy()
    basis = J.jspace_basis(Jm, W_U, gain, r_max=10, chunk=37)
    M = W_U @ torch.diag(gain) @ Jm
    _, S, Vh = torch.linalg.svd(M.double(), full_matrices=False)
    for i in range(10):  # same subspace, vector by vector up to sign
        assert abs(float(basis.vecs[:, i].double() @ Vh[i])) == pytest.approx(1.0, abs=1e-4)
    assert torch.allclose(basis.evals[:10].double(), (S[:10] ** 2), rtol=1e-3)
    assert torch.allclose(basis.vecs.T @ basis.vecs, torch.eye(10), atol=1e-5)


def test_project_out_and_enrichment():
    W_U, Jm, gain = _toy()
    b = J.jspace_basis(Jm, W_U, gain, r_max=16)
    v = b.vecs[:, 0] + b.vecs[:, 20 % 16] * 0  # pure J direction
    assert J.enrichment(b, v, r=4) == pytest.approx(32 / 4, rel=1e-4)  # all energy in S_4
    out = J.project_out(b, v, r=4)
    assert float(out.norm()) < 1e-5
    w = torch.randn(32, generator=torch.Generator().manual_seed(1))
    wo = J.project_out(b, w, r=8)
    assert float((b.vecs[:, :8].T @ wo).abs().max()) < 1e-4
    # batched [N, d]
    W = torch.randn(5, 32)
    assert J.project_out(b, W, r=8).shape == (5, 32)
    assert J.enrichment(b, W, r=8).shape == (5,)


def test_j_removed_direction_is_unit_and_orthogonal():
    W_U, Jm, gain = _toy()
    b = J.jspace_basis(Jm, W_U, gain, r_max=16)
    w = torch.randn(32)
    u = J.j_removed_direction(b, w, r=6)
    assert float(u.norm()) == pytest.approx(1.0, abs=1e-5)
    assert float((b.vecs[:, :6].T @ u).abs().max()) < 1e-5


def test_random_subspace_control_enrichment_near_one():
    d, r = 512, 64
    Q = J.random_subspace(d, r, seed=0)
    assert torch.allclose(Q.T @ Q, torch.eye(r), atol=1e-4)
    vs = torch.randn(400, d, generator=torch.Generator().manual_seed(2))
    e = J.enrichment_in(Q, vs).mean()
    assert float(e) == pytest.approx(1.0, abs=0.05)
    out = J.project_out_of(Q, vs)
    assert float((out @ Q).abs().max()) < 1e-3


def test_random_direction_controls():
    W_U, Jm, gain = _toy()
    b = J.jspace_basis(Jm, W_U, gain, r_max=16)
    nonj = J.random_direction(b, r=8, kind="nonj", seed=0)
    inj = J.random_direction(b, r=8, kind="j", seed=0)
    iso = J.random_direction(b, r=8, kind="iso", seed=0)
    for v in (nonj, inj, iso):
        assert float(v.norm()) == pytest.approx(1.0, abs=1e-5)
    assert float((b.vecs[:, :8].T @ nonj).abs().max()) < 1e-5
    assert float(J.project_out(b, inj, r=8).norm()) < 1e-5


def test_rank_exceeding_basis_raises():
    W_U, Jm, gain = _toy()
    b = J.jspace_basis(Jm, W_U, gain, r_max=8)
    with pytest.raises(ValueError):
        J.project_out(b, torch.randn(32), r=9)


def test_nn_pursuit_recovers_sparse_nonnegative_combo():
    g = torch.Generator().manual_seed(0)
    atoms = torch.randn(100, 64, generator=g)
    atoms = atoms / atoms.norm(dim=1, keepdim=True)
    H = 2.0 * atoms[3] + 1.0 * atoms[17]
    idx, coef, recon = J.nn_pursuit(H[None], atoms.to(torch.bfloat16), k=4, refine_steps=20)
    assert {3, 17} <= set(idx[0].tolist())
    assert float((recon - H).norm() / H.norm()) < 0.05
    assert (coef >= 0).all()
    res = J.project_out_span(H[None], atoms, idx)
    assert float(res.norm()) < 1e-3 * float(H.norm()) + 1e-3
