"""Shared loaders for g1 WorkspaceBench figures (exps 01 and 06). Reads judged regex results only."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E01 = ROOT / "experiments" / "01_wsbench_replication"
E06 = ROOT / "experiments" / "06_resolving_decomposition"
FAMILIES = ["typo_mt", "multilingual_mt", "multihop_mt", "basic_readout_mt"]
FAMILY_LABEL = {
    "typo_mt": "typo correction",
    "multilingual_mt": "multilingual concept",
    "multihop_mt": "multi-hop bridge",
    "basic_readout_mt": "basic readout",
}
# the eight prompt-only texts of exp 06 (2 models x 2 instructions, Qwen k=3)
PROMPT_ONLY_ARMS = [f"qwen_prompt_nla_s{i}" for i in range(3)] + [f"qwen_prompt_wsb_s{i}" for i in range(3)] + [
    "gem_prompt_nla", "gem_prompt_wsb"]


def judged_path(exp_dir: Path, arm: str, fam: str) -> Path:
    return exp_dir / "judged" / arm / fam / "results.json"


def load_rows(exp_dir: Path, arm: str) -> dict:
    """{(family, id): row} for one judged arm."""
    out = {}
    for fam in FAMILIES:
        d = json.loads(judged_path(exp_dir, arm, fam).read_text())
        for r in d["rows"]:
            out[(fam, r["id"])] = r
    return out


def load_pass(exp_dir: Path, arm: str) -> dict:
    return {k: bool(r["pass"]) for k, r in load_rows(exp_dir, arm).items()}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))
