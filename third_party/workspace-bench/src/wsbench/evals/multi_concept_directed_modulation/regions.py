"""Region labels for the write cells (only IN_SENTENCE cells are judged) and the write-compliance
gate; ported unchanged from the source repo's ``regions.py``. Rationale in the family README."""

from collections.abc import Sequence
from enum import StrEnum
from typing import NamedTuple

TARGET_SENTENCE = "The committee will meet on Thursday to review the annual budget."
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class Region(StrEnum):
    IN_SENTENCE = "in_sentence"
    IN_THINK = "in_think"
    POST_THINK = "post_think"
    SELF_ADDED_META = "self_added_meta"
    OFF_TASK = "off_task"


def _is_blank_line(token: str) -> bool:
    """A paragraph break: whitespace only, containing a newline. This is the boundary token."""
    return token.strip() == "" and "\n" in token


def classify_regions(tokens: Sequence[str]) -> list[Region]:
    """Label each cell of the read window, given the cells' tokens in FORWARD order: everything
    before the first paragraph break is IN_SENTENCE; a ``<think>`` block through ``</think>`` is
    IN_THINK and what follows is POST_THINK; a break with no think block starts SELF_ADDED_META;
    a window with no break is entirely IN_SENTENCE."""
    open_at = next((i for i, t in enumerate(tokens) if THINK_OPEN in t), None)
    close_at = next((i for i, t in enumerate(tokens) if THINK_CLOSE in t), None)
    break_at = next((i for i, t in enumerate(tokens) if _is_blank_line(t)), None)
    if open_at is not None and (break_at is None or break_at > open_at):
        break_at = open_at
    if open_at is None and close_at is not None:
        break_at = 0  # the window opens inside the think block
    if break_at is None:
        return [Region.IN_SENTENCE] * len(tokens)
    tail = (
        Region.IN_THINK if open_at is not None or close_at is not None else Region.SELF_ADDED_META
    )
    out = [Region.IN_SENTENCE if i < break_at else tail for i in range(len(tokens))]
    if close_at is not None:
        for i in range(close_at + 1, len(tokens)):
            out[i] = Region.POST_THINK
    return out


class WriteCompliance(NamedTuple):
    complied: bool
    matched: str  # the longest contiguous fragment of the target found in the window
    run: int  # its length in words
    min_run: int


def _words(text: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in text.lower()).split()


def write_compliance(
    cell_tokens: Sequence[str], target: str = TARGET_SENTENCE, *, min_run: int = 3
) -> WriteCompliance:
    """``min_run`` consecutive words of the target must appear in the window (a contiguous run,
    not bag overlap: off-task prose shares "the" and "a" with any target)."""
    want, got = _words(target), _words(" ".join(cell_tokens))
    best, at = 0, 0
    for i in range(len(want)):
        for j in range(len(got)):
            n = 0
            while i + n < len(want) and j + n < len(got) and want[i + n] == got[j + n]:
                n += 1
            if n > best:
                best, at = n, i
    return WriteCompliance(best >= min_run, " ".join(want[at : at + best]), best, min_run)


def label_cells(
    cell_tokens: Sequence[str], *, target: str = TARGET_SENTENCE, min_run: int = 3
) -> tuple[list[Region], WriteCompliance]:
    """Region per cell (forward order) plus the write-compliance verdict; a non-compliant window
    is OFF_TASK throughout."""
    compliance = write_compliance(cell_tokens, target, min_run=min_run)
    if not compliance.complied:
        return [Region.OFF_TASK] * len(cell_tokens), compliance
    return classify_regions(cell_tokens), compliance
