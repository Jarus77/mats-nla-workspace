"""The one token-bag -> prose step for J-lens (``kind == "tokens"``) readouts: the families that
judge token bags hand the summarizer one bundle text per cell and judge the returned
interpretation in place of the readout. Prompt and provenance: ``docs/summarizer.md``."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence

from wsbench import llm
from wsbench.cache import Cache, fingerprint
from wsbench.judge_config import ResolvedJudge
from wsbench.llm import Spend, schema_block

SUMMARIZER_PROMPT_VERSION = "interp-v1"

INTERP_SYSTEM = """You are shown the top-10 token readouts from an interpretability lens at one \
position inside a language model that was reading a passage you cannot see. Tokens may include \
noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, \
state what these outputs are collectively trying to say — the situation or mental content they \
point to. Commit to the most specific reading the tokens support; do not just say they are noisy."""
INTERP_SCHEMA = schema_block("interp", {"interpretation": {"type": "string"}}, ["interpretation"])
INTERP_USER = "TOKEN READOUTS:\n{txt}"

PROMPTS: dict[str, str] = {"INTERP_SYSTEM": INTERP_SYSTEM, "INTERP_USER": INTERP_USER}


def render_user(txt: str) -> str:
    return INTERP_USER.replace("{txt}", txt)


def render_bag(tokens: Sequence[str], scores: Sequence[float] | None) -> str:
    """``tok (score)`` list, best first, scores to 2 dp; without scores the tokens joined by
    ``" | "``."""
    if scores is None:
        return " | ".join(tokens)
    pairs = sorted(zip(tokens, scores, strict=True), key=lambda p: -p[1])
    return " | ".join(f"{t} ({s:.2f})" for t, s in pairs)


def aux_judge(judge: ResolvedJudge, aux_models: Mapping[str, str], role: str) -> ResolvedJudge:
    """The judge used for an auxiliary stage: ``aux_models[role]`` if set, else the judge."""
    model = aux_models.get(role, judge.model)
    if model == judge.model:
        return judge
    return dataclasses.replace(judge, model=model)


def summarize(
    bundles: Mapping[str, str],
    *,
    judge: ResolvedJudge,
    cache: Cache,
    spend: Spend,
    concurrency: int,
    rpm: float,
    dry_run: bool = False,
    preflight: Callable[[], None] | None = None,
) -> dict[str, str | None]:
    """key -> interpretation text (``None`` on failure). Cache key ``summ:<key>``; the fingerprint
    is ``(SUMMARIZER_PROMPT_VERSION, model, bundle_text)``. ``spend`` is accumulated in place.
    ``dry_run`` prints the first bundle's rendered prompt and returns ``{}`` (no client)."""
    if dry_run:
        for key, txt in bundles.items():
            print(f"--- summarizer prompt for {key} (model={judge.model}) ---")
            print("[system]")
            print(INTERP_SYSTEM)
            print("[user]")
            print(render_user(txt))
            break
        return {}
    out: dict[str, str | None] = {}
    pending: list[tuple[str, str, str]] = []  # (key, fp, text)
    for key, txt in bundles.items():
        fp = fingerprint(SUMMARIZER_PROMPT_VERSION, judge.model, txt)
        row = cache.get(f"summ:{key}", fp)
        if row is not None:
            out[key] = _text(row.get("result"))
        else:
            pending.append((key, fp, txt))
    print(f"summarizer: {len(pending)} calls ({len(out)} cached)")
    if not pending:
        return out

    def on_result(i: int, r: dict | None) -> None:
        key, fp, _txt = pending[i]
        cache.put(f"summ:{key}", fp, {"result": r})
        out[key] = _text(r)

    if preflight is not None:
        preflight()
    llm.stream_json(
        [(INTERP_SYSTEM, render_user(txt)) for _k, _fp, txt in pending],
        schema=INTERP_SCHEMA,
        model=judge.model,
        reasoning=judge.reasoning,
        on_result=on_result,
        concurrency=concurrency,
        rpm=rpm,
        spend=spend,
    )
    for key, _fp, _txt in pending:
        out.setdefault(key, None)
    return out


def _text(r: object) -> str | None:
    if not isinstance(r, dict):
        return None
    t = r.get("interpretation")
    if not isinstance(t, str) or not t.strip():
        return None
    return t
