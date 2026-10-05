"""Build the TextNLA job list (user-turn texts) shared by the Qwen (Modal) and Gemini backends.

Writes data/jobs.json: [{key, arm_suffix, family, id, user}] with arm_suffix in
{prompt_nla, prompt_wsb, resolve_nla, resolve_ol42}. key = f"{arm_suffix}|{family}|{id}".
"""
import json

from common import HERE, RESOLVE_TEMPLATE, WSB_PROMPT_ONLY, READOUTS_01, FAMILIES, load_items, load_readouts


def final_tokens() -> dict[tuple[str, str], str]:
    out = {}
    for fam in FAMILIES:
        for line in (READOUTS_01 / "nla" / f"{fam}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[(fam, r["id"])] = r["token"]
    return out


def main() -> None:
    instr = (HERE / "data" / "nla_instruction.txt").read_text(encoding="utf-8")
    nla, ol = load_readouts("nla"), load_readouts("olens")
    toks = final_tokens()
    jobs = []
    for it in load_items():
        k = (it["family"], it["id"])
        users = {
            "prompt_nla": instr.replace("{text}", it["prompt"]),
            "prompt_wsb": WSB_PROMPT_ONLY.format(token=toks[k], prefix=it["prompt"]),
            "resolve_nla": RESOLVE_TEMPLATE.format(expl=nla[k]),
            "resolve_ol42": RESOLVE_TEMPLATE.format(expl=ol[k]),
        }
        for arm, user in users.items():
            jobs.append({"key": f"{arm}|{it['family']}|{it['id']}", "arm": arm,
                         "family": it["family"], "id": it["id"], "user": user})
    (HERE / "data" / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    print(len(jobs), "jobs")


if __name__ == "__main__":
    main()
