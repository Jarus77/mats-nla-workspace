"""Every prompt string a family sends lives verbatim in its README (and the summarizer's in
docs/summarizer.md) inside a fenced block."""

import importlib
import re
from pathlib import Path

import pytest

from wsbench import registry, summarizer

ROOT = Path(__file__).resolve().parents[1]
# fences of any length (```` when a prompt itself contains ```), closed by the same run
_FENCE = re.compile(r"(`{3,})[^\n]*\n(.*?)\1", re.DOTALL)


def fenced_blocks(md: str) -> list[str]:
    return [body for _fence, body in _FENCE.findall(md)]


def _families() -> list[str]:
    registry.load_all()
    return sorted(registry.FAMILIES)


@pytest.mark.parametrize("family", _families())
def test_family_prompts_verbatim_in_readme(family: str):
    mod = importlib.import_module(f"wsbench.evals.{family}.prompts")
    prompts: dict[str, str] = mod.PROMPTS
    readme = (ROOT / "evals" / family / "README.md").read_text(encoding="utf-8")
    assert prompts, f"{family}: every family is LLM-judged and must export PROMPTS"
    blocks = fenced_blocks(readme)
    assert blocks
    for name, text in prompts.items():
        assert isinstance(text, str) and text
        assert any(text in b for b in blocks), f"{family}: {name} is not verbatim in the README"
    assert isinstance(mod.PROMPT_VERSION, str) and mod.PROMPT_VERSION
    assert mod.PROMPT_VERSION in (ROOT / "evals" / family / "README.md").read_text()


def test_summarizer_prompt_verbatim_in_docs():
    blocks = fenced_blocks((ROOT / "docs/summarizer.md").read_text(encoding="utf-8"))
    for name, text in summarizer.PROMPTS.items():
        assert any(text in b for b in blocks), name
    doc = (ROOT / "docs/summarizer.md").read_text()
    assert summarizer.SUMMARIZER_PROMPT_VERSION in doc and "interpretation" in doc


def test_fence_parser():
    assert fenced_blocks("a\n```text\nX\n```\nb\n```\nY\n```") == ["X\n", "Y\n"]
