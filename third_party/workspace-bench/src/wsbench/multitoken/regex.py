"""The regex contract of the six multi-token families — a deterministic port of the source repo's
conjunctive bank scorer (``olens_suite/bank/{matching,contract,conjunctive}.py`` and
``olens_sglang/{score_targets,common}.py`` at ``global-workspace-clean``), the scorer of record
again since 2026-09-23 (Camila: "regex on the multitoken please").

The rules, in the order a cell is scored:

1. **Samples.** A prose cell's samples, or a token lens's top-k strings (each token is one
   sample, exactly the string the producer wrote — ``Ġ``/``▁`` already rendered as a space,
   byte-level pieces of non-Latin tokens left as they are, as the source scorer saw them).
   Chat scaffolding (``<|im_start|>``, ``<|im_end|>``, ``<think>``, ``</think>``,
   ``<explanation>`` tags) is stripped first (``extract_phrase``).
2. **Units.** ``scored_units``: every ``units[]`` entry of the item; when the bank contract is
   ``multi_token`` and the unit does not set ``multi_token: false`` (the language units do),
   only forms the bank stamped as multi-token stay (``probe_token_lens.units[role][lang][i] >
   1`` — the source's strict count: min over case variants x bare/leading-space, so a form any
   single Qwen3.6-27B token could equal is never creditable). ``include_target`` banks
   (multihop_mt) add an OPTIONAL ``target`` unit from ``target`` + ``target_alts``, likewise
   filtered (``probe_token_lens.target`` / ``.target_alts``), minus forms a unit already covers.
   A REQUIRED unit left without a creditable form is a bank error.
3. **A form hits a sample** (``unicode_word_matcher``): both sides folded (NFKD, combining marks
   dropped, the right single quotation mark -> ``'``, casefold). A form containing CJK or
   Hangul, or not made of word characters, is a plain substring; a wordy form is a
   word-boundary match with ``[\\s-]+`` between its words (``Bohr magneton`` ==
   ``Bohr-magneton``); a purely numeric form must sit in
   ANSWER position (after ``=``, ``->``, ``equals``, ``answer/result/value/total/sum/product/
   quotient [is][:]``, their Chinese counterparts, or alone at the start of the sample, never
   as a prefix of a longer number; a CJK numeral in those positions counts when it parses to
   the value). A form is matched against ONE sample at a time — a phrase is never assembled
   across two samples or two top-k tokens.
4. **A unit hits at a layer** when any of its forms hits any single sample of any row at that
   layer (``layer_unit_hits``: role -> the languages whose forms hit, in authored order).
5. **A layer passes an item** when EVERY required unit hits there; **the item passes** at any
   layer (``item_result``). ``any_hit`` (some unit hit somewhere) is the parent-comparable extra.

What this means for a top-k token lens: a strictly multi-token form can never be one token, so
on the raw bag required multi-token units are unreachable by construction — the source's stated
asymmetry ("multi-token targets can only be hit by the oracle lens"). The bench therefore keeps
its bag -> prose step for token readouts (``wsbench.summarizer``, ``docs/summarizer.md``): each
cell's bag is interpreted once by the shared summarizer (cached, same fingerprint as the MC judge
used) and the contract is applied to the interpretation as one more sample beside the raw tokens
— a unit hits when a form is found in the summary OR in a single token (``run_regex`` in
``family.py``; ``SUMMARIZED_SCORER_VERSION``).
"""

import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "SCORER_VERSION",
    "SUMMARIZED_SCORER_VERSION",
    "BankContract",
    "ScoredUnit",
    "contract_for",
    "extract_phrase",
    "fold",
    "hit_forms",
    "item_result",
    "layer_passes",
    "layer_unit_hits",
    "parse_cjk_numeral",
    "scored_units",
    "unicode_word_matcher",
]

SCORER_VERSION = "mt-regex-2026-09-23"  # prose arms: the contract on the samples
# token arms: the contract on each top-k token AND on the shared summarizer's interpretation of
# the bag (docs/summarizer.md), as separate samples — Camila 2026-09-23 16:52 ("for the token
# lenses on mt i thought we were doing an llm summarizer step?")
SUMMARIZED_SCORER_VERSION = "mt-regex-summarized-2026-09-23"

# ---------------------------------------------------------------------------------------------
# olens_sglang/common.py — scaffolding stripped from every sample before scoring
# ---------------------------------------------------------------------------------------------
_STRIP_PATTERNS = re.compile(
    r"</?explanation>|<\|im_start\|>(system|user|assistant)?|<\|im_end\|>|<think>|</think>"
)


def extract_phrase(text: str) -> str:
    """Strip chat scaffolding + ``<explanation>`` tags from one lens sample."""
    return _STRIP_PATTERNS.sub("", text).strip()


# ---------------------------------------------------------------------------------------------
# global_workspace/glossary.py — the two numeric helpers the matcher needs
# ---------------------------------------------------------------------------------------------
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９．", "0123456789.")  # noqa: RUF001
CJK_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6}
CJK_DIGIT |= {"七": 7, "八": 8, "九": 9}
CJK_UNIT = {"十": 10, "百": 100, "千": 1000}


def parse_cjk_numeral(s: str) -> int | None:
    """Standard CJK numerals up to 9999 (十五=15, 四十九=49, 三百=300, 两=2), else ``None``."""
    if not s or any(c not in CJK_DIGIT and c not in CJK_UNIT for c in s):
        return None
    total, current = 0, 0
    for c in s:
        if c in CJK_DIGIT:
            current = CJK_DIGIT[c]
        else:
            total += (current or 1) * CJK_UNIT[c]
            current = 0
    return total + current


# ---------------------------------------------------------------------------------------------
# olens_suite/bank/matching.py — hit semantics for conjunctive units
# ---------------------------------------------------------------------------------------------
_WRAP = r"[\s$*{(\\`：（＊]*"  # noqa: RUF001 — markdown/mathjax/fullwidth wrapping before a number
_ANSWER_CONTEXT = (
    r"(?:=|->|→|\bequals\b|\b(?:answer|result|value|total|sum|product|quotient)"
    r"\b(?:\s+is)?\s*:?"
    r"|(?:答案|结果|总和|总数)(?:是|为|：|:)?"  # noqa: RUF001
    r"|(?:值|积|商|差|和)(?:是|为|：|:)"  # noqa: RUF001
    r"|等于|得出|即|共计|共)"
)
_HAN = r"[一-鿿]"


def _number_pattern(t: str) -> str:
    """The digits of a purely numeric target with full-number identity: 84 ∉ 84.5 / 184."""
    return re.escape(t) + r"(?![\d.,]?\d)(?!\.\d)"


def _numeric_matcher(target: str) -> Callable[[str], bool]:
    """The source ``word_matcher`` numeric branch: the number in ANSWER position — after an
    answer marker (English or Chinese), or alone (possibly wrapped) at the start of the sample
    and not opening an expression — or a CJK numeral there that parses to the value. Fullwidth
    digits are folded first. The sample is NOT lowercased on this path (as in the source's
    conjunctive scorer), so the English markers match in lowercase only."""
    t = target.lower()
    num = _number_pattern(t)
    after_eq = re.compile(_ANSWER_CONTEXT + _WRAP + num)
    at_start = re.compile(r"\A[\s$*#>\-]*" + num + r"(?!\s*[+\-*/×÷^=]\s*\d)")  # noqa: RUF001
    cjk_numeral = re.compile(
        r"(?:" + _ANSWER_CONTEXT + _WRAP + r"|\A[\s$*#>\-]*)"
        r"([零一二两三四五六七八九十百千]{1,6})(?!" + _HAN + ")"
    )
    value = int(t)

    def numeric(text: str) -> bool:
        folded = text.translate(FULLWIDTH_DIGITS)
        if after_eq.search(folded) or at_start.search(folded):
            return True
        return any(parse_cjk_numeral(m) == value for m in cjk_numeral.findall(folded))

    return numeric


def fold(s: str) -> str:
    """NFKD, strip combining marks, straighten the right single quote, casefold — applied to
    BOTH the form and the sample (Mexico with/without the accent, pointed and unpointed Hebrew,
    curly and straight apostrophes all coincide)."""
    s = unicodedata.normalize("NFKD", s).replace("\u2019", "'")
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


_CJK_HANGUL = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_WORDY = re.compile(r"[^\W_]+(?:[\s\-'][^\W_]+)*")


def unicode_word_matcher(form: str) -> Callable[[str], bool]:
    """Boundary match on folded text for a unit form (any script); CJK/Hangul and non-wordy
    forms are substrings of the folded sample; purely numeric forms use the answer-context
    rule. The script decision is made on the RAW form (NFKD decomposes Hangul syllables)."""
    f = fold(form)
    if re.fullmatch(r"\d+", f):
        return _numeric_matcher(f)
    if _CJK_HANGUL.search(form) or not _WORDY.fullmatch(f):
        return lambda text: f in fold(text)
    parts = [re.escape(w) for w in re.split(r"[\s\-]+", f) if w]
    pat = re.compile(r"(?<![^\W_])" + r"[\s\-]+".join(parts) + r"(?![^\W_])")
    return lambda text: bool(pat.search(fold(text)))


def _standalone_number_matcher(form: str) -> Callable[[str], bool]:
    """``numeric_match: "standalone"``: the integer anywhere as a standalone number (no digit,
    decimal, thousands or sign continuation on either side, no letters). None of the six banks
    sets it; ported so a bank that does scores as the source would."""
    pat = re.compile(r"(?<![\w.,-])" + re.escape(form.strip()) + r"(?![\w.,]?\d)(?!\w)")
    return lambda text: bool(pat.search(fold(text).replace(",", "")))


NumericMatch = Literal["context", "standalone"]


def hit_forms(
    samples: list[str],
    forms: Mapping[str, list[str]],
    *,
    numeric_match: NumericMatch = "context",
) -> list[str]:
    """The language keys of ``forms`` whose strings hit any SINGLE sample, in dict order."""

    def matcher(f: str) -> Callable[[str], bool]:
        if numeric_match == "standalone" and re.fullmatch(r"-?\d+", f.strip()):
            return _standalone_number_matcher(f)
        return unicode_word_matcher(f)

    out: list[str] = []
    for lang, fs in forms.items():
        matchers = [matcher(f) for f in fs]
        if matchers and any(m(s) for m in matchers for s in samples):
            out.append(lang)
    return out


# ---------------------------------------------------------------------------------------------
# olens_suite/bank/contract.py — the bank contract and the per-unit form builder
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BankContract:
    multi_token: bool  # only multi-token forms may award credit
    include_target: bool  # the item's ``target`` is itself a scored (optional) unit
    conjunctive_units: bool  # a layer passes only when EVERY required unit hits there


def contract_for(header: Mapping[str, Any], family: str) -> BankContract:
    """The header's ``contract`` block, else the source's legacy name-suffix rules."""
    block = header.get("contract")
    if isinstance(block, Mapping):
        return BankContract(
            multi_token=bool(block.get("multi_token", family.endswith("-mt"))),
            include_target=bool(block.get("include_target", True)),
            conjunctive_units=bool(block.get("conjunctive_units", False)),
        )
    return BankContract(
        multi_token=family.endswith("-mt"),
        include_target=family != "multilingual-mt",
        conjunctive_units=False,
    )


@dataclass(frozen=True)
class ScoredUnit:
    role: str
    required: bool
    forms: dict[str, list[str]]  # language -> the strings that may award credit
    numeric_match: NumericMatch = "context"

    def all_forms(self) -> list[str]:
        out: list[str] = []
        for fs in self.forms.values():
            for f in fs:
                if f not in out:
                    out.append(f)
        return out

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "role": self.role,
            "required": self.required,
            "forms": dict(self.forms),
        }
        if self.numeric_match != "context":
            out["numeric_match"] = self.numeric_match
        return out


def _numeric_match_of(raw: Mapping[str, Any]) -> NumericMatch:
    mode = str(raw.get("numeric_match", "context"))
    if mode not in ("context", "standalone"):
        raise ValueError(f"unit {raw.get('role')!r}: unknown numeric_match {mode!r}")
    return "standalone" if mode == "standalone" else "context"


def _unit_forms(raw: Mapping[str, Any]) -> dict[str, list[str]]:
    forms = raw.get("forms")
    if isinstance(forms, Mapping) and forms:
        return {str(k): [str(x) for x in v] for k, v in forms.items()}
    return {"": [str(m) for m in raw.get("match", [])]}


def _counts(item: Mapping[str, Any], role: str, lang: str, n: int) -> list[int]:
    """``probe_token_lens.units[role][lang]`` — the bank's Qwen3.6-27B token count per form,
    positionally aligned with ``forms[lang]``. A missing stamp is a bank error, never a guess."""
    try:
        counts = [int(x) for x in item["probe_token_lens"]["units"][role][lang]]
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"item {item.get('name')!r}: no probe_token_lens count for unit {role!r} {lang!r}"
        ) from e
    if len(counts) != n:
        raise ValueError(
            f"item {item.get('name')!r}: probe_token_lens.units[{role!r}][{lang!r}] has "
            f"{len(counts)} counts for {n} forms"
        )
    return counts


def scored_units(item: Mapping[str, Any], contract: BankContract) -> list[ScoredUnit]:
    """Units of one item under ``contract`` — the source's ``scored_units`` with the token
    counts read from the bank's ``probe_token_lens`` stamp instead of a live tokenizer. No alias
    pruning: forms are an authored OR-list (the Russian nominative beside its genitive)."""
    units: list[ScoredUnit] = []
    for raw in item.get("units") or []:
        role = str(raw.get("role", "unit"))
        forms = _unit_forms(raw)
        if contract.multi_token and bool(raw.get("multi_token", True)):
            kept: dict[str, list[str]] = {}
            for lang, fs in forms.items():
                counts = _counts(item, role, lang, len(fs))
                fs2 = [f for f, n in zip(fs, counts, strict=True) if n > 1]
                if fs2:
                    kept[lang] = fs2
            forms = kept
        required = bool(raw.get("required", True))
        if required and not forms:
            raise ValueError(
                f"item {item.get('name')!r}: required unit {role!r} has no creditable "
                "(multi-token) form"
            )
        if forms:
            units.append(ScoredUnit(role, required, forms, _numeric_match_of(raw)))
    roles = {u.role for u in units}
    target = item.get("target")
    if contract.include_target and target and "target" not in roles:
        alts = [str(a) for a in item.get("target_alts", [])]
        tforms = [str(target), *alts]
        if contract.multi_token:
            ptl = item.get("probe_token_lens") or {}
            try:
                counts = [int(ptl["target"]), *(int(x) for x in ptl["target_alts"])]
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"item {item.get('name')!r}: include_target needs probe_token_lens.target "
                    "and .target_alts"
                ) from e
            if len(counts) != len(tforms):
                raise ValueError(
                    f"item {item.get('name')!r}: probe_token_lens.target_alts has "
                    f"{len(counts) - 1} counts for {len(alts)} target_alts"
                )
            tforms = [f for f, n in zip(tforms, counts, strict=True) if n > 1]
        covered = {f.lower() for u in units for f in u.all_forms()}
        tforms = [f for f in tforms if f.lower() not in covered]
        if tforms:
            units.append(ScoredUnit("target", False, {"en": tforms}))
    return units


# ---------------------------------------------------------------------------------------------
# olens_suite/bank/conjunctive.py — per-layer unit hits and the item verdict
# ---------------------------------------------------------------------------------------------
def layer_unit_hits(samples: list[str], units: list[ScoredUnit]) -> dict[str, list[str]]:
    """role -> the languages in which the unit hit any single sample at this layer."""
    return {u.role: hit_forms(samples, u.forms, numeric_match=u.numeric_match) for u in units}


def layer_passes(hits: Mapping[str, list[str]], units: list[ScoredUnit]) -> bool:
    return all(bool(hits.get(u.role)) for u in units if u.required)


def item_result(
    by_layer: Mapping[int, Mapping[str, list[str]]], units: list[ScoredUnit]
) -> dict[str, Any]:
    """Fold per-layer unit hits into the item row: pass = some layer where every required unit
    hits; ``unit_langs`` lists the languages in the unit's authored order (L2 -> en -> zh)."""
    passing = sorted(layer for layer, hits in by_layer.items() if layer_passes(hits, units))
    unit_hit = {u.role: any(bool(h.get(u.role)) for h in by_layer.values()) for u in units}
    unit_langs = {
        u.role: [
            lang for lang in u.forms if any(lang in h.get(u.role, []) for h in by_layer.values())
        ]
        for u in units
    }
    return {
        "pass": bool(passing),
        "earliest_layer": passing[0] if passing else None,
        "n_passing_layers": len(passing),
        "passing_layers": passing,
        "unit_hit": unit_hit,
        "unit_langs": unit_langs,
        "first_lang": {role: (langs[0] if langs else None) for role, langs in unit_langs.items()},
        "any_hit": any(unit_hit.values()),
    }
