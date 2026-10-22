"""CPU checks for the stage-2a helpers (run: python -m pytest experiments/10_operand_gate/test_o10s2_lib.py)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o10s2_lib as L  # noqa: E402


def test_exemplars_disjoint_positive_deterministic():
    ex1, ex2 = L.exemplars(51, 49), L.exemplars(51, 49)
    assert ex1 == ex2 and len(ex1) == 3
    for x, op, y, z in ex1:
        assert not ({x, y} & {51, 49}) and z >= 1 and z == (x + y if op == "+" else x - y)
    assert L.exemplars(51, 49) != L.exemplars(51, 48)


def test_fewshot_text_ends_with_item():
    t = L.fewshot_text({"a": 51, "b": 49, "text": "51 - 49 ="})
    assert t.endswith("\n51 - 49 =") and t.count("\n") == 3 and t.count("=") == 4


def test_parse_and_first_digit():
    assert L.parse_answer(" 102\n") == 102 and L.parse_answer("2") == 2 and L.parse_answer(" ?") is None
    assert L.first_digit(105) == 1 and L.first_digit(7) == 7


def test_folds_balanced_and_centred_geometry():
    f = L.folds(601)
    assert abs((f == 0).sum() - (f == 1).sum()) <= 1
    h = torch.randn(5, 8)
    v = L.unit(torch.randn(8))
    out = L.centred(h, v, 2.0)
    assert torch.allclose(out @ v, torch.full((5,), 2.0), atol=1e-5)
    Q = torch.linalg.qr(torch.randn(8, 3))[0]
    R = L.random_dirs(8, 4, 2, Q, 0)
    assert torch.allclose(R.norm(dim=1), torch.ones(6), atol=1e-5)
    assert (R[:4] @ Q).abs().max() < 1e-5 and (R[4:] @ Q).abs().max() > 1e-3


def test_signed_value_shift_and_flip():
    ans = {"none": [5, 5, None], "d+": [7, 5, None], "d-": [3, 5, 6], "n0+": [5, 5, None], "n0-": [5, 5, None]}
    s = L.signed_value_shift(ans, "d", np.array([1, -1, 1]))
    assert s[0] == 2.0 and s[1] == 0.0 and np.isnan(s[2])
    fl = L.flip_summary(ans, ["d"], {"iso": ["n0"]})
    assert abs(fl["d"]["flip_rate"] - 0.5) < 1e-9
