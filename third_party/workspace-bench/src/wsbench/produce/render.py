"""Turn a read-plan row into token ids: the renders of ``wsbench.readplan.RENDERS``."""

from dataclasses import dataclass
from typing import Any

from wsbench.readplan import ReadSpec


@dataclass(frozen=True)
class Rendered:
    """One rendered prompt: its token ids, their display strings, and how it was built."""

    ids: list[int]
    tokens: list[str]  # display spelling (``" the"``, ``Ċ``), what the positions rules match
    decoded: list[str]  # ``tokenizer.decode([id])`` per token, what the banks store as `token`
    render: str

    def __len__(self) -> int:
        return len(self.ids)


def display_tokens(tokenizer: Any, ids: list[int]) -> list[str]:
    """Token strings as the benchmark's readouts spell them: byte-level BPE with ``Ġ``/``▁``
    shown as the space they encode (``" the"``), so a readout can be read and matched as text."""
    return [
        t.replace("Ġ", " ").replace("▁", " ") if isinstance(t, str) else ""
        for t in tokenizer.convert_ids_to_tokens(ids)
    ]


def _chat(tokenizer: Any, messages: list[dict[str, str]], prefill: str = "") -> list[int]:
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    ids: list[int] = list(out["input_ids"] if hasattr(out, "keys") else out)
    if prefill:
        ids += tokenizer(prefill, add_special_tokens=False)["input_ids"]
    return ids


def render(spec: ReadSpec, tokenizer: Any) -> Rendered:
    """Token ids for a plan row under its ``render``. The chat renders use the tokenizer's own
    template with thinking off, which is how every bank was captured."""
    r = spec.render
    if r == "captured":
        ids = [int(x) for x in spec.extra["input_ids"]]
    elif r in ("plain", "bare"):
        ids = tokenizer(spec.text or "", add_special_tokens=False)["input_ids"]
    elif r == "chat":
        messages = ([{"role": "system", "content": spec.system}] if spec.system else []) + [
            {"role": "user", "content": spec.text or ""}
        ]
        ids = _chat(tokenizer, messages)
    elif r == "chat_context":
        ids = _chat(tokenizer, list(spec.messages or []))
    elif r == "chat_prefill":
        ids = _chat(tokenizer, [{"role": "user", "content": spec.text or ""}], spec.prefill or "")
    elif r == "chat_dm":
        ids = _chat(tokenizer, [{"role": "user", "content": spec.text or ""}], spec.assistant or "")
    elif r == "chat_summarize":
        ids = _chat(
            tokenizer, [{"role": "user", "content": (spec.text or "") + (spec.suffix or "")}]
        )
    else:
        raise ValueError(f"unknown render {r!r}")
    ids = list(ids)
    return Rendered(
        ids=ids,
        tokens=display_tokens(tokenizer, ids),
        decoded=[tokenizer.decode([i]) for i in ids],
        render=r,
    )


def render_text(
    text: str, tokenizer: Any, *, chat: bool = False, system: str | None = None
) -> Rendered:
    """A one-off prompt outside any bank: plain, or as a single user turn."""
    spec = ReadSpec(
        "adhoc", "adhoc", "chat" if chat else "plain", {"kind": "all"}, [], text=text, system=system
    )
    return render(spec, tokenizer)
