"""Step 2 (local): LLM-assisted extraction of candidate template families from the MINING readouts.

Same procedure for every reader: Gemini 3.8 Flash (OpenRouter) sees the reader's mining readouts
grouped by the true feature value and proposes template families with one {v} slot + per-value
fillers, in the reader's native format. Candidates are validated mechanically (feature_spec.validate)
and capped at MAX_PER_READER. J-Lens word sets are mined statistically from the J-Lens top-50 lists
(tokens over-represented in one value's mining items). Also measures the base rate at which each
reader spontaneously mentions the feature (regex for every feature + an LLM yes/no pass for col).

Key from OPENROUTER_API_KEY / repo .env (never printed). Hard spend cap SPEND_CAP_USD.

    python extract_templates.py          -> results/candidates.json, results/extraction_log.json
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from feature_spec import (FEATURES, FIXED_JLENS, QUALITATIVE_PRIMARY, fixed_extra, fixed_template,
                          is_numeric, validate)  # noqa: E402
from mats.textnla import _api_key  # noqa: E402

RES = HERE / "results"
MODEL = "google/gemini-3.8-flash"
SPEND_CAP_USD = 15.0
MAX_PER_READER = 10
MAX_TOKENS_EXTRACT = 8000
SHOW_PER_VALUE = 70
TRUNC = 350
SEED = 0

READER_DESC = {
    "nla": ("a Natural Language Autoencoder verbalizer. Its output is a free-text explanation in a "
            "terse analytic register. Candidate strings are scored as the VERY BEGINNING of the "
            "explanation (the text right after '<explanation>\\n')."),
    "ol": ("the Oracle Lens. Its output is a list of '- ' bullet lines, often content continuations "
           "or paraphrases of the source text (sometimes drifting into other languages) rather than "
           "abstract labels. Candidate strings are scored as the FIRST bullet line of its response, "
           "so every template MUST start with '- '. Continuation-style templates are welcome "
           "(e.g. for language identity '- {v}' with fillers 'Le'/'Die') as long as the filler for "
           "each value is GENERIC (would fit any item with that value), not item-specific."),
}

MENTION_RE = {
    "nj1_col": r"\b(characters?|chars|line[- ]length|line[- ]width|column|wrap\w*|line[- ]?breaks?|newline|width|long line|short line)\b",
    "ij2_lang": r"\b(french|german|français|deutsch)\b",
    "nj2_tense": r"\b(past|present|tense)\b",
    "bk_indent": r"\b(indent\w*|nest\w*|level|depth|block)\b",
    "bk_wordpos": r"\b(word count|words? (in|into|so far)|sentence length|mid-sentence|start of (the|a) sentence|early in the sentence)\b",
    "bk_listidx": r"\b(item|items|list|bullet|entry|entries|number|count|ninth|second|third)\b",
    "bk_caps": r"\b(caps|capital\w*|uppercase|upper-case|lowercase|shout\w*)\b",
    "bk_code": r"\b(python|javascript|js|node)\b",
    "nj3_predpos": r"\b(adjective|noun|modifier|descriptor)\b",
    "ij_listnum": r"\b(item|items|list|entry|entries|number|count|ninth|second|third|enumerat\w*)\b",
    "bk_quote": r"\b(quot\w*|dialogue|speech|said|cited|citation)\b",
    "agr_gender": r"\b(masculine|feminine|gender|neuter)\b",
    "agr_number": r"\b(singular|plural)\b",
}
MENTION_RE["nj1_col_break"] = MENTION_RE["nj1_col"]
LENGTH_FEATURES = ("nj1_col", "nj1_col_break")

FAMILY_HINT = {
    "nj1_col": (
        "\nThis call mines the 'LONG LINE / LINE LENGTH' family (the COUNT: how far into the line "
        "the text is). PRIMARY templates must be QUALITATIVE, with no numbers (e.g. fillers "
        "'near the end of a long line' vs 'at the start of a line'); give at least 5 of those. "
        "Then give up to 3 SECONDARY templates whose fillers are numerals or number words. Add "
        "\"kind\": \"qualitative\"|\"numeric\" to every family."),
    "nj1_col_break": (
        "\nThis call mines the 'A LINE BREAK IS COMING' family (the DECISION to wrap soon, with "
        "the column matched across values). PRIMARY templates must be QUALITATIVE (e.g. fillers "
        "'a line break is coming' vs 'mid-line, no break yet'); give at least 5. Then up to 3 "
        "SECONDARY numeric ones (e.g. characters left before the wrap). Add \"kind\": "
        "\"qualitative\"|\"numeric\" to every family."),
}

PROMPT = """You are helping build scoring templates for an interpretability study.

A reader tool, {reader_desc}

The tool read internal activations of a language model taken at a token where a feature has a known value.
Feature: {feature} = {desc}.
Possible values: {values}.

Below are the tool's free-form readouts, grouped by the TRUE feature value of the item they came from.

{blocks}

Task: find recurring phrasings in these readouts that express (or reliably co-vary with) the feature value, and turn them into candidate TEMPLATE FAMILIES for teacher-forced scoring. Each template:
- is a string in the tool's native format (see above) containing exactly one slot "{{v}}";
- has "fillers" mapping EACH value ({values}) to the text that goes in the slot;
- the filled strings for the different values must differ ONLY in the slot text;
- must be generic (no item-specific content words), short (under ~15 words), and plausible as the start of this tool's output;
- prefer phrasings that actually recur in the readouts; you may include at most 2 templates that are natural paraphrases in the tool's style if the readouts never mention the feature explicitly.
For count-like features, the slot may hold a number word, a bin word, or a qualitative phrase.

Also estimate: what fraction of readouts (0-1) explicitly mention the feature at all ("mention_rate"), and whether that rate differs by value.

Return JSON: {{"families": [{{"family": short name, "template": "...{{v}}...", "fillers": {{value: text, ...}}, "evidence": "brief quote(s) from the readouts", "approx_count": integer}}], "mention_rate": number, "notes": "one sentence"}}. Give 4-8 families, best first."""

COL_CLASSIFY = """Each numbered line below is one readout of an interpretability tool describing a language model's internal state. For EACH line, answer whether it mentions line length, character count, column position, line width/wrapping, or how far the text is into the current line (yes/no).

{lines}

Return JSON: {{"answers": [{{"i": number, "mentions": true|false}}]}} covering every line."""


class Gemini:
    def __init__(self) -> None:
        self.spend = 0.0
        self.calls = 0
        self.key = _api_key()

    def __call__(self, prompt: str, schema: dict | None = None, max_tokens: int = 6000) -> dict | None:
        if self.spend >= SPEND_CAP_USD:
            raise RuntimeError(f"spend cap reached (${self.spend:.2f})")
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2, "max_tokens": max_tokens, "reasoning": {"effort": "low"},
                "response_format": {"type": "json_object"}}
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        for attempt in range(4):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=180))
                self.calls += 1
                self.spend += float(r.get("usage", {}).get("cost") or 0.0)
                txt = r["choices"][0]["message"]["content"] or ""
                m = re.search(r"\{.*\}", txt, re.DOTALL)
                return json.loads(m.group(0)) if m else None
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
                print(f"  retry {attempt}: {type(e).__name__}", flush=True)
                time.sleep(2 ** attempt)
        return None


def readouts(feature: str) -> dict[str, list[tuple[str, str, str]]]:
    """{reader: [(item_id, value, text)]} for the mining split (OL pooled over its 3 layers)."""
    g_ol = json.loads((RES / f"gen_ol_{feature}.json").read_text())
    val = {it["id"]: it["value"] for it in g_ol["items"]}
    out = {"ol": [], "nla": []}
    for L, d in g_ol["ol_gens"].items():
        for iid, texts in d.items():
            out["ol"] += [(iid, val[iid], f"[L{L}] {t}") for t in texts]
    p = RES / f"gen_nla_{feature}.json"
    if p.exists():
        for iid, texts in json.loads(p.read_text())["nla_gens"].items():
            out["nla"] += [(iid, val[iid], t) for t in texts if t]
    return out


def mention_rates(feature: str, rows: list[tuple[str, str, str]]) -> dict:
    rx = re.compile(MENTION_RE[feature], re.IGNORECASE)
    by = Counter()
    tot = Counter()
    for _, v, t in rows:
        tot[v] += 1
        by[v] += bool(rx.search(re.sub(r"^\[L\d+\] ", "", t)))
    return {"regex": MENTION_RE[feature], "overall": sum(by.values()) / max(1, sum(tot.values())),
            "by_value": {v: by[v] / tot[v] for v in tot}, "n": sum(tot.values())}


def col_llm_mentions(gem: Gemini, rows: list[tuple[str, str, str]]) -> dict:
    """LLM yes/no: does each readout mention line length / column / wrapping? (col only)."""
    chunks = [rows[s: s + 40] for s in range(0, len(rows), 40)]

    def run(chunk: list) -> list[tuple[str, bool]]:
        lines = "\n".join(f"{i}. {t[:TRUNC].replace(chr(10), ' ')}" for i, (_, _, t) in enumerate(chunk))
        r = gem(COL_CLASSIFY.format(lines=lines), max_tokens=4000) or {}
        ans = {int(a["i"]): bool(a["mentions"]) for a in r.get("answers", []) if "i" in a}
        return [(chunk[i][1], ans[i]) for i in range(len(chunk)) if i in ans]

    with ThreadPoolExecutor(6) as ex:
        res = [x for part in ex.map(run, chunks) for x in part]
    by, tot = Counter(), Counter()
    for v, m in res:
        tot[v] += 1
        by[v] += m
    return {"overall": sum(by.values()) / max(1, len(res)), "by_value": {v: by[v] / tot[v] for v in tot},
            "n_classified": len(res), "n": len(rows)}


def clean_template(t: str) -> str:
    """Strip display artefacts the LLM can copy (our ' / ' newline marker) and trailing whitespace
    (a trailing space changes the scored tokens). Applied identically to every reader."""
    t = re.sub(r"(\s*/\s*)+$", "", str(t))
    return t.rstrip()


def extract(gem: Gemini, reader: str, feature: str, rows: list) -> tuple[list[dict], dict]:
    rng = random.Random(SEED)
    vals = FEATURES[feature]["values"]
    blocks = []
    for v in vals:
        rv = [t for _, val, t in rows if val == v]
        rng.shuffle(rv)
        body = "\n".join(f"* {t[:TRUNC].replace(chr(10), ' / ')}" for t in rv[:SHOW_PER_VALUE])
        blocks.append(f"=== TRUE VALUE: {v} ({len(rv)} readouts, showing {min(len(rv), SHOW_PER_VALUE)}) ===\n{body}")
    prompt = PROMPT.format(reader_desc=READER_DESC[reader], feature=feature,
                           desc=FEATURES[feature]["desc"], values=vals, blocks="\n\n".join(blocks))
    prompt += FAMILY_HINT.get(feature, "")
    r = gem(prompt, max_tokens=MAX_TOKENS_EXTRACT) or {}
    fams, rejected, seen = [], [], set()
    for k, f in enumerate(r.get("families", [])):
        t = {"id": f"{reader}:{feature}:m{k}", "family": str(f.get("family", f"m{k}"))[:60],
             "template": clean_template(f.get("template", "")), "fillers": {str(a): str(b) for a, b in (f.get("fillers") or {}).items()}}
        err = validate(t, reader, feature)
        key = (t["template"], tuple(sorted(t["fillers"].items())))
        if err or key in seen:
            rejected.append({**t, "error": err or "duplicate"})
            continue
        seen.add(key)
        fams.append({**t, "evidence": str(f.get("evidence", ""))[:300],
                     "approx_count": f.get("approx_count"), "kind_llm": f.get("kind"),
                     "numeric": is_numeric(t)})
    if feature in QUALITATIVE_PRIMARY:  # keep up to 8 qualitative + 4 numeric
        q = [x for x in fams if not x["numeric"]][:8]
        n = [x for x in fams if x["numeric"]][:4]
        return q + n, {"mention_rate_llm": r.get("mention_rate"), "notes": r.get("notes"),
                       "rejected": rejected, "n_proposed": len(r.get("families", []))}
    return fams[:MAX_PER_READER], {"mention_rate_llm": r.get("mention_rate"), "notes": r.get("notes"),
                                   "rejected": rejected, "n_proposed": len(r.get("families", []))}


def jlens_sets(feature: str, top_k: int = 5) -> list[dict]:
    """Word sets from tokens over-represented in each value's J-Lens top-50 lists (mining items)."""
    g = json.loads((RES / f"gen_ol_{feature}.json").read_text())
    val = {it["id"]: it["value"] for it in g["items"]}
    vals = FEATURES[feature]["values"]
    df = {v: Counter() for v in vals}
    n = Counter()
    for iid, top in g["jlens_top50_L42"].items():
        n[val[iid]] += 1
        df[val[iid]].update({w.strip() for w, _ in top if re.fullmatch(r"\s?[A-Za-zÀ-ÿ]{2,}", w)})
    diff = {}
    for v, o in ((vals[0], vals[1]), (vals[1], vals[0])):
        sc = {w: df[v][w] / n[v] - df[o][w] / n[o] for w in df[v]}
        diff[v] = [w for w, s in sorted(sc.items(), key=lambda x: -x[1]) if s > 0.15][:top_k]
    out = [{"id": f"jlens:{feature}:fixed", "family": "fixed_label", "words": FIXED_JLENS[feature]}]
    for k in range(min(len(diff[vals[0]]), len(diff[vals[1]]), 3)):
        out.append({"id": f"jlens:{feature}:d{k}", "family": f"diff_rank{k}",
                    "words": {vals[0]: [diff[vals[0]][k]], vals[1]: [diff[vals[1]][k]]}})
    if diff[vals[0]] and diff[vals[1]]:
        out.append({"id": f"jlens:{feature}:dtop3", "family": "diff_top3",
                    "words": {vals[0]: diff[vals[0]][:3], vals[1]: diff[vals[1]][:3]}})
    return out


def main() -> None:
    pc, pl = RES / "candidates.json", RES / "extraction_log.json"
    cand = json.loads(pc.read_text()) if pc.exists() else {"nla": {}, "ol": {}, "jlens": {}}
    log: dict = json.loads(pl.read_text()) if pl.exists() else {}
    prior_spend = float(log.get("_spend_usd", 0.0))
    feats = [f for f in FEATURES if (RES / f"gen_ol_{f}.json").exists()
             and (RES / f"gen_nla_{f}.json").exists() and f not in cand["jlens"]]
    print("extracting:", feats)
    gem = Gemini()
    gem.spend = prior_spend  # the cap applies to the cumulative spend
    jobs = []
    for f in feats:
        rows = readouts(f)
        log[f] = {r: {"mention_regex": mention_rates(f, rows[r]), "n_readouts": len(rows[r])} for r in rows}
        if f in LENGTH_FEATURES:
            for r in rows:
                if rows[r]:
                    log[f][r]["mention_llm_classify"] = col_llm_mentions(gem, rows[r])
        for r in ("nla", "ol"):
            if rows[r]:
                jobs.append((r, f, rows[r]))
        cand["jlens"][f] = jlens_sets(f)
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(lambda j: (j[0], j[1], extract(gem, *j)), jobs))
    for r, f, (fams, info) in results:
        cand[r][f] = fams + [fixed_template(r, f)] + fixed_extra(r, f)
        log[f][r].update(info)
        print(f"{f:14s} {r:4s}: {len(fams)} mined (+fixed); llm mention {info['mention_rate_llm']}; "
              f"regex {log[f][r]['mention_regex']['overall']:.2f}")
    for r in ("nla", "ol"):
        for f in feats:
            cand[r].setdefault(f, [fixed_template(r, f)] + fixed_extra(r, f))
    log["_calls"] = log.get("_calls", 0)
    log["_spend_usd"] = round(gem.spend, 4)
    log["_calls"] += gem.calls
    (RES / "candidates.json").write_text(json.dumps(cand, indent=1, ensure_ascii=False))
    (RES / "extraction_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
    print(f"spend ${gem.spend:.3f} over {gem.calls} calls")


if __name__ == "__main__":
    main()
