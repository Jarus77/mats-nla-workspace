"""Claim-level text-verifiable precision (prereg §6.2, adapted to the 5-way codebook).

Sample: 100 readouts per reader at L42 (25 per family, seed 0, same items for every reader):
NLA@42, OL@42, and TextNLA(prompt, base Qwen, NLA instruction, sample 0) as the text-only
reference. Judge = Gemini 3.8 Flash (T=0, reasoning low): splits each readout into atomic
claims and labels each. Validation = 40 claims (stratified by reader x Gemini label) re-labelled
by Fireworks kimi-k3 given the same context and claim; Cohen's kappa (5-way and collapsed 3-way).

    python claims.py judge      # -> data/claims_gemini.jsonl
    python claims.py validate   # -> data/claims_kimi.jsonl
"""
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from common import FAMILIES, HERE, load_items, load_readouts
from llm import GEMINI, KIMI, Client

LABELS = ["contradicted_by_text", "prompt_restatement", "target_latent_consistent",
          "prompt_inferable", "unsupported_by_text"]
N_PER_FAM = 25
CODEBOOK = """Labels (mutually exclusive; if several apply, use the FIRST that applies in this order):
1. contradicted_by_text: the claim conflicts with the text or with the gold answer/intermediate (e.g. names a different entity than the gold one as the answer, wrong language, wrong spelling presented as correct, misquotes the text, wrong facts about the text).
2. prompt_restatement: the claim restates or paraphrases content that is explicitly present in the text (including describing the literal final word/token or quoting the text).
3. target_latent_consistent: the claim names or directly asserts the gold answer or a gold intermediate (or an equivalent form / translation), which is NOT literally present in the text.
4. prompt_inferable: not stated in the text, but clearly follows from the text by ordinary reading, grammar or general knowledge (genre, tone, register, syntactic role of the last token, what kind of word comes next, a correct spelling of a misspelled word, the language of the text, plausible next words consistent with the text).
5. unsupported_by_text: a specific assertion that the text and the gold answer neither support nor contradict (e.g. invented details, names, quotes, numbers or context not implied by the text). Unsupported does not mean false."""

SPLIT_PROMPT = """You audit readouts produced by interpretability tools. A tool was given a language model's internal activation at the FINAL token of the text below and wrote a readout describing what the model was representing (some readouts are lists of likely continuations: treat each continuation as asserting its content).

<text>{text}</text>
Gold answer the model is expected to produce next / name: {target}
Gold latent intermediate(s) (may equal the answer): {inter}

<readout>
{readout}
</readout>

Split the readout into its atomic claims (one checkable assertion each; merge duplicates; skip pure filler; at most 12 claims, keep the most specific ones) and label each claim.

{codebook}

Return JSON: {{"claims": [{{"claim": "...", "label": "<one of the labels>"}}]}}"""

CLASSIFY_PROMPT = """You audit readouts produced by interpretability tools. A tool was given a language model's internal activation at the FINAL token of the text below and wrote a readout; one atomic claim from that readout is shown.

<text>{text}</text>
Gold answer the model is expected to produce next / name: {target}
Gold latent intermediate(s) (may equal the answer): {inter}

Claim: "{claim}"

Label the claim.

{codebook}

Return JSON: {{"label": "<one of the labels>"}}"""

SPLIT_SCHEMA = {"type": "object", "properties": {"claims": {"type": "array", "items": {
    "type": "object", "properties": {"claim": {"type": "string"},
                                     "label": {"type": "string", "enum": LABELS}},
    "required": ["claim", "label"], "additionalProperties": False}}},
    "required": ["claims"], "additionalProperties": False}
LABEL_SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": LABELS}},
                "required": ["label"], "additionalProperties": False}


def sample_items() -> list[dict]:
    rng = random.Random(0)
    items = load_items()
    out = []
    for fam in FAMILIES:
        fi = [x for x in items if x["family"] == fam]
        out += rng.sample(fi, N_PER_FAM)
    return out


def readers() -> dict[str, dict]:
    out = {"NLA@42": load_readouts("nla"), "OL@42": load_readouts("olens")}
    qp = HERE / "data" / "qwen_gen.json"
    if qp.exists():
        q = json.loads(qp.read_text(encoding="utf-8"))["outs"]
        out["TextNLA(prompt,Qwen)"] = {tuple(k.split("|")[1:]): v["samples"][0]
                                       for k, v in q.items() if k.startswith("prompt_nla|")}
    return out


def ctx(it: dict) -> dict:
    return {"text": it["prompt"], "target": " / ".join([it["target"], *it["target_alts"]]),
            "inter": ", ".join(it["intermediates"]) or it["target"]}


def parse_json(txt: str | None) -> dict | None:
    if not txt:
        return None
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def judge() -> None:
    c = Client(model=GEMINI, temperature=0.0, max_tokens=4000, reasoning="low")
    rd = readers()
    path = HERE / "data" / "claims_gemini.jsonl"
    have = set()
    if path.exists():
        have = {json.loads(l)["reader"] for l in path.read_text(encoding="utf-8").splitlines()}
    jobs = [(r, it) for r in rd if r not in have for it in sample_items()]

    def run(job):
        r, it = job
        out = parse_json(c.chat(SPLIT_PROMPT.format(readout=rd[r][(it["family"], it["id"])],
                                                    codebook=CODEBOOK, **ctx(it)), SPLIT_SCHEMA))
        return {"reader": r, "family": it["family"], "id": it["id"],
                "claims": (out or {}).get("claims"), "ok": out is not None}

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(run, jobs))
    with path.open("a", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(n=len(res), failed=sum(not r["ok"] for r in res), usd=round(c.spend_usd, 4)))
    _log("claims_judge_gemini", c)


def validate(n: int = 40) -> None:
    rows = [json.loads(l) for l in (HERE / "data" / "claims_gemini.jsonl").read_text().splitlines()]
    flat = [(r, i, cl) for r in rows for i, cl in enumerate(r["claims"] or [])]
    rng = random.Random(1)
    # stratify: round-robin over (reader, gemini label) cells
    cells: dict = {}
    for x in flat:
        cells.setdefault((x[0]["reader"], x[2]["label"]), []).append(x)
    for v in cells.values():
        rng.shuffle(v)
    pick, keys = [], sorted(cells)
    while len(pick) < n and any(cells.values()):
        for k in keys:
            if cells[k] and len(pick) < n:
                pick.append(cells[k].pop())
    items = {(x["family"], x["id"]): x for x in load_items()}
    c = Client(model=KIMI, temperature=0.0, max_tokens=8000, reasoning=None)

    def run(x):
        r, i, cl = x
        it = items[(r["family"], r["id"])]
        out = parse_json(c.chat(CLASSIFY_PROMPT.format(claim=cl["claim"], codebook=CODEBOOK,
                                                       **ctx(it)), LABEL_SCHEMA))
        return {"reader": r["reader"], "family": r["family"], "id": r["id"], "idx": i,
                "claim": cl["claim"], "gemini": cl["label"], "kimi": (out or {}).get("label")}

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(run, pick))
    with (HERE / "data" / "claims_kimi.jsonl").open("w", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(n=len(res), unparsed=sum(r["kimi"] is None for r in res), usd_est=round(c.spend_usd, 4)))
    _log("claims_validate_kimi", c)


def _log(step: str, c: Client) -> None:
    with (HERE / "data" / "spend_log.jsonl").open("a") as fh:
        fh.write(json.dumps({"step": step, "model": c.model, "n_calls": c.n_calls,
                             "n_failed": c.n_failed, "usd": c.spend_usd}) + "\n")


if __name__ == "__main__":
    {"judge": judge, "validate": validate}[sys.argv[1]]()
