"""Re-solver consistency check (4): Gemini 3.8 Flash re-solver given the PROMPT (+ optionally a
reader's readout), same instructions as RESOLVE_TEMPLATE. k=1, T=1, max_tokens 256, reasoning
minimal. Arms: gem_rsP (prompt only), gem_rsP_nla (prompt + NLA@42 readout),
gem_rsP_ol42 (prompt + OL@42 readout). Cache: data/resolve_plus.jsonl; then regex-scored.
"""
import json
from concurrent.futures import ThreadPoolExecutor

from common import HERE, load_items, load_readouts
from llm import Client
from score import write_and_judge

HEAD = ("Below is a text that a language model was reading, cut off at its final token{extra}.\n\n"
        "<text>\n{text}\n</text>\n\n")
DESC = ("\n\nBelow is also a description, produced by an interpretability tool, of what the model "
        "was internally representing at that final token.\n\n<description>\n{expl}\n</description>")
TAIL = ("Work out what the model was thinking about at that token. Write a short summary (at most "
        "~150 words) that (1) states what the text is about, (2) names the specific entity, term, "
        "word or answer the model was most likely representing or about to produce, written out in "
        "full and spelled correctly, and (3) states the language of the text.")


def user(text: str, expl: str | None) -> str:
    extra = "" if expl is None else DESC.format(expl=expl)
    # description placed before the instructions, after the text
    return HEAD.format(extra="", text=text) + (extra.strip() + "\n\n" if expl else "") + TAIL


def main() -> None:
    nla, ol = load_readouts("nla"), load_readouts("olens")
    jobs = []
    for it in load_items():
        k = (it["family"], it["id"])
        for arm, ex in (("gem_rsP", None), ("gem_rsP_nla", nla[k]), ("gem_rsP_ol42", ol[k])):
            jobs.append((arm, k, user(it["prompt"], ex)))
    path = HERE / "data" / "resolve_plus.jsonl"
    done = {}
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done[(r["arm"], r["family"], r["id"])] = r["text"]
    todo = [j for j in jobs if (j[0], *j[1]) not in done]
    c = Client(max_tokens=256)
    with ThreadPoolExecutor(16) as ex, path.open("a", encoding="utf-8") as fh:
        for (arm, k, _), out in zip(todo, ex.map(lambda j: c.chat(j[2]), todo)):
            if out is not None:
                done[(arm, *k)] = out
                fh.write(json.dumps({"arm": arm, "family": k[0], "id": k[1], "text": out},
                                    ensure_ascii=False) + "\n")
    print(dict(n_calls=c.n_calls, failed=c.n_failed, usd=round(c.spend_usd, 4)))
    with (HERE / "data" / "spend_log.jsonl").open("a") as fh:
        fh.write(json.dumps({"step": "resolve_plus", "n_calls": c.n_calls, "usd": c.spend_usd}) + "\n")
    arms: dict = {}
    for (arm, f, i), t in done.items():
        arms.setdefault(arm, {})[(f, i)] = t
    write_and_judge(arms)


if __name__ == "__main__":
    main()
