"""Score every TextNLA arm with the fork's regex instrument (mt-regex-2026-09-23) and collect
per-item pass/fail for all arms (TextNLA arms + the 01 reader arms) into data/item_pass.json.

    python score.py      # needs data/qwen_gen.json and data/gemini_gen.jsonl
"""
import json
import shutil
import subprocess

from common import FAMILIES, FORK, HERE, JUDGED_01, READOUTS_01

JIN, JOUT = HERE / "judge_in", HERE / "judged"
READER_ARMS = {"nla_L42": "nla_L42", "olens_L42": "olens_L42", "olens_L40": "olens_L40",
               "olens_L44": "olens_L44", "jlens_L42": "jlens_L42", "olens_grid": "olens_grid"}


def cell_meta() -> dict:
    out = {}
    for fam in FAMILIES:
        for line in (READOUTS_01 / "nla" / f"{fam}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[(fam, r["id"])] = (r["pos"], r["token"])
    return out


def textnla_arms() -> dict[str, dict]:
    """arm -> {key-without-arm: text}"""
    arms: dict[str, dict] = {}
    q = json.loads((HERE / "data" / "qwen_gen.json").read_text(encoding="utf-8"))["outs"]
    for key, v in q.items():
        arm, fam, iid = key.split("|")
        for s, txt in enumerate(v["samples"]):
            arms.setdefault(f"qwen_{arm}_s{s}", {})[(fam, iid)] = txt
    for line in (HERE / "data" / "gemini_gen.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        arm, fam, iid = r["key"].split("|")
        arms.setdefault(f"gem_{arm}", {})[(fam, iid)] = r["text"]
    return arms


def write_and_judge(arms: dict[str, dict]) -> None:
    meta = cell_meta()
    for arm, cells in arms.items():
        d = JIN / arm
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for fam in FAMILIES:
            with (d / f"{fam}.jsonl").open("w", encoding="utf-8") as fh:
                for (f, iid), txt in sorted(cells.items()):
                    if f != fam:
                        continue
                    pos, tok = meta[(f, iid)]
                    fh.write(json.dumps({"id": iid, "layer": 42, "pos": pos, "token": tok,
                                         "samples": [txt]}, ensure_ascii=False) + "\n")
        r = subprocess.run(["uv", "run", "wsbench", "run", f"families={','.join(FAMILIES)}",
                            f"readouts_root={d}", f"out={JOUT / arm}"],
                           cwd=FORK, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{arm}: {r.stderr[-2000:]}")
        print(arm, "judged", flush=True)


def collect(arms: list[str]) -> dict:
    out = {}
    for arm in arms:
        root = JUDGED_01 / READER_ARMS[arm] if arm in READER_ARMS else JOUT / arm
        for fam in FAMILIES:
            res = json.loads((root / fam / "results.json").read_text(encoding="utf-8"))
            assert res["config"]["prompt_version"].startswith("mt-regex"), res["config"]
            out.setdefault(arm, {})[fam] = {row["id"]: bool(row["pass"]) for row in res["rows"]}
    return out


def main() -> None:
    arms = textnla_arms()
    write_and_judge(arms)
    item_pass = collect(list(arms) + list(READER_ARMS))
    (HERE / "data" / "item_pass.json").write_text(json.dumps(item_pass, indent=0))
    for arm, fams in sorted(item_pass.items()):
        print(f"{arm:26s}", " ".join(f"{sum(v.values()) / len(v):.2f}" for v in fams.values()))


if __name__ == "__main__":
    main()
