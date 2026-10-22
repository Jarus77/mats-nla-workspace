import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o10g4_lib as L  # noqa: E402


def test_bank_and_aprime():
    it = L.onedigit_items()
    assert len(it) == 117 and all(x["answer"] >= 1 for x in it) and len({x["text"] for x in it}) == 117
    assert sorted(L.aprime(a) for a in range(1, 10)) == list(range(1, 10)) and all(L.aprime(a) != a for a in range(1, 10))
    t = L.fewshot_text_onedigit(it[0])
    assert t.endswith("\n1 + 1 =") and t.count("\n") == 3


def test_swap_sets_profile_and_jcomplement_subspaces():
    d, k = 32, 9
    V = torch.randn(k, d)
    U = L.orthonormal_span(V)
    assert torch.allclose(U.T @ U, torch.eye(k), atol=1e-5)
    H = torch.randn(5, d)
    tgt = torch.randn(5, k)
    Hs = L.swap_vectors(H, U, tgt)
    assert torch.allclose(Hs @ U, tgt, atol=1e-4)
    assert torch.allclose(Hs - Hs @ U @ U.T, H - H @ U @ U.T, atol=1e-4)  # complement untouched
    Q = torch.linalg.qr(torch.randn(d, 6))[0]
    S = L.random_subspaces(d, k, 3, Q, 0)
    assert S.shape == (3, d, k) and (S[0].T @ Q).abs().max() < 1e-5
    disp = L.random_displacements(S[0], torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]), 1)
    assert torch.allclose(disp.norm(dim=1), torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]), atol=1e-4)
    prof = L.class_profiles(H, np.array([1, 1, 2, 2, 2]), U, (1, 2))
    assert torch.allclose(prof[1], (H[:2] @ U).mean(0), atol=1e-5)


def test_swap_stats():
    ans = {"none": [5, 5, 5, None], "s": [8, 6, 5, 7]}
    truth, pred = np.array([5, 5, 5, 5]), np.array([8, 8, 5, 8])
    st = L.swap_stats(ans, "s", truth, pred)
    assert st["n_valid"] == 3 and abs(st["follow_exact"] - 1 / 3) < 1e-9 and st["moved_toward"] == 1.0
    assert abs(L.flip_rate(ans, "s") - 0.75) < 1e-9
