import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import lib24 as L  # noqa: E402


def _planted(n: int = 200, d: int = 24, seed: int = 0):
    """Two clusters of points separated along one direction, grouped in blocks of four."""
    g = torch.Generator().manual_seed(seed)
    y = np.arange(n) % 2
    X = torch.randn(n, d, generator=g)
    X[:, 0] += 4.0 * torch.as_tensor(y, dtype=torch.float32)
    groups = np.arange(n) // 4
    return X, y, groups


def test_shuffle_keeps_the_multiset_and_changes_the_order():
    ids = list(range(32))
    s = L.shuffle_tokens(ids, random.Random(24))
    assert sorted(s) == ids and s != ids
    assert L.shuffle_tokens([7], random.Random(0)) == [7]


def test_frozen_probe_is_high_when_r_keeps_the_direction_and_at_chance_on_noise():
    X, y, groups = _planted()
    kept = X + 0.1 * torch.randn(X.shape, generator=torch.Generator().manual_seed(1))
    noise = torch.randn(X.shape, generator=torch.Generator().manual_seed(2))
    assert L.frozen_probe_auroc(X, kept, y, groups) > 0.95
    assert abs(L.frozen_probe_auroc(X, noise, y, groups) - 0.5) < 0.12


def test_class_direction_cos_is_one_for_the_reference_and_near_zero_for_noise():
    X, y, groups = _planted()
    noise = torch.randn(X.shape, generator=torch.Generator().manual_seed(3))
    assert abs(L.class_direction_cos(X, X, y) - 1.0) < 1e-6
    assert abs(L.class_direction_cos(X, noise, y)) < 0.3


def test_conditions_returns_the_seven_keys_in_order_with_empty_fallback_and_caps():
    enc = {}

    def tok_encode(text: str) -> list[int]:
        enc[text] = [ord(c) for c in text]
        return enc[text]

    span = list(range(100, 132))
    long = list(range(4, 132))
    rng = random.Random(24)
    c = L.conditions(span, long, "", "x" * 300, tok_encode, rng, cap=256)
    assert list(c) == ["span_32", "span_128", "bag_32", "ol_readout", "ol_readout_shuffled",
                       "nla_readout", "nla_readout_shuffled"]
    assert list(c) == list(L.CONDITIONS)
    assert c["span_32"] == span and c["span_128"] == long
    assert sorted(c["bag_32"]) == span and c["bag_32"] != span
    assert "(empty)" in enc and c["ol_readout"] == [ord(ch) for ch in "(empty)"]
    assert sorted(c["ol_readout_shuffled"]) == sorted(c["ol_readout"])
    assert len(c["nla_readout"]) == 256 and len(c["nla_readout_shuffled"]) == 256
    assert sorted(c["nla_readout_shuffled"]) == sorted(c["nla_readout"])
    assert L.conditions(span, long, "  ", "", tok_encode, rng, cap=8)["nla_readout"] == [ord(ch) for ch in "(empty)"][:8]


def test_table_lists_conditions_in_order_and_formats_two_decimals():
    vals = {c: 0.5 + 0.05 * i for i, c in enumerate(L.CONDITIONS)}
    row = {"feature": "py_string", "n": 160, "variant": "li/ln",
           "layers": {"44": {"frozen_auroc": vals, "cv_auroc": vals, "class_direction_cos": {c: 0.123456 for c in L.CONDITIONS}},
                      "40": {"frozen_auroc": vals, "cv_auroc": vals, "class_direction_cos": vals}}}
    md = L.table([row], layer="44")
    lines = md.splitlines()
    headers = [ln for ln in lines if ln.startswith("| row |")]
    assert len(headers) == 3
    for h in headers:
        assert h == "| row | " + " | ".join(L.CONDITIONS) + " |"
    assert "| py_string | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 |" in lines
    assert "0.12 |" in md and "0.123" not in md
    assert md.count("py_string") == 3 and "44" in md


def test_olens_ar_helpers_are_importable_from_src():
    from mats.readers.olens_ar import AR_REPO, AR_SUBDIR, BASE, rename_lora_keys, right_pad

    assert (AR_REPO, AR_SUBDIR, BASE) == ("agu18dec/olens_and_ar", "ar_ptag_pooled", "Qwen/Qwen3.6-27B")
    sd = rename_lora_keys({"base_model.model.layers.3._orig_mod.mlp.up_proj.lora_A.weight": torch.zeros(1)})
    assert list(sd) == ["base_model.model.layers.3.mlp.up_proj.lora_A.weight"]
    ids, mask = right_pad([[7, 8, 9], [5]], pad_id=0)
    assert ids.tolist() == [[7, 8, 9], [5, 0, 0]] and mask.tolist() == [[1, 1, 1], [1, 0, 0]]
