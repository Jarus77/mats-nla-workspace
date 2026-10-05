"""Item-level re-solving decomposition (tasks 1-2) from data/item_pass.json -> data/tables.json
and data/tables.md.

Text-only arms (no activation):
  T_qwen     qwen_prompt_nla_s0   base Qwen, NLA warm-start instruction, k=1 (primary, k-matched)
  T_qwen_k3  any of qwen_prompt_nla_s{0,1,2}
  T_wsb      qwen_prompt_wsb_s0   base Qwen, wsbench prompt-only template
  T_gem      gem_prompt_nla       Gemini 3.8 Flash, NLA instruction
  T_any      union of every prompt-only arm above + gem_prompt_wsb (most generous text baseline)
Re-solving ceilings: R_<reader>_<model> = text-only LLM given only that reader's readout.
"""
import json
import math
import random

from common import FAMILIES, HERE

IP = json.loads((HERE / "data" / "item_pass.json").read_text())
IDS = {f: sorted(IP["nla_L42"][f]) for f in FAMILIES}
B = 2000


def vec(arm: str, fams=FAMILIES) -> list[bool]:
    return [IP[arm][f][i] for f in fams for i in IDS[f]]


def union(arms: list[str], fams=FAMILIES) -> list[bool]:
    vs = [vec(a, fams) for a in arms]
    return [any(t) for t in zip(*vs)]


DERIVED = {
    "T_qwen_k3": ["qwen_prompt_nla_s0", "qwen_prompt_nla_s1", "qwen_prompt_nla_s2"],
    "T_any": ["qwen_prompt_nla_s0", "qwen_prompt_nla_s1", "qwen_prompt_nla_s2",
              "qwen_prompt_wsb_s0", "qwen_prompt_wsb_s1", "qwen_prompt_wsb_s2",
              "gem_prompt_nla", "gem_prompt_wsb"],
}
ALIAS = {"NLA@42": "nla_L42", "OL@42": "olens_L42", "OL@40": "olens_L40", "OL@44": "olens_L44",
         "JLens@42": "jlens_L42", "OL grid (any-of-11)": "olens_grid",
         "T_qwen": "qwen_prompt_nla_s0", "T_wsb": "qwen_prompt_wsb_s0", "T_gem": "gem_prompt_nla",
         "T_gem_wsb": "gem_prompt_wsb",
         "Resolve(NLA) Qwen": "qwen_resolve_nla_s0", "Resolve(NLA) Gemini": "gem_resolve_nla",
         "Resolve(OL@42) Qwen": "qwen_resolve_ol42_s0", "Resolve(OL@42) Gemini": "gem_resolve_ol42"}


def get(name: str, fams=FAMILIES) -> list[bool]:
    if name in DERIVED:
        return union(DERIVED[name], fams)
    return vec(ALIAS.get(name, name), fams)


def ci(v: list[bool], seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(v) / n, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """b10 = a passes & b fails; b01 = a fails & b passes; exact two-sided binomial p."""
    b10 = sum(x and not y for x, y in zip(a, b))
    b01 = sum(y and not x for x, y in zip(a, b))
    n, k = b10 + b01, min(b10, b01)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b10, b01, p


def fmt_ci(v):
    m, lo, hi = ci(v)
    return f"{m:.2f} [{lo:.2f},{hi:.2f}]"


def pass_table() -> tuple[list[str], dict]:
    names = ["NLA@42", "OL@42", "OL@40", "OL@44", "JLens@42", "OL grid (any-of-11)",
             "T_qwen", "T_qwen_k3", "T_wsb", "T_gem", "T_gem_wsb", "T_any",
             "Resolve(NLA) Qwen", "Resolve(NLA) Gemini", "Resolve(OL@42) Qwen", "Resolve(OL@42) Gemini"]
    rows, data = ["| arm | " + " | ".join(FAMILIES) + " | pooled (n=400) |",
                  "|---|" + "---|" * (len(FAMILIES) + 1)], {}
    for nm in names:
        cells = [fmt_ci(get(nm, [f])) for f in FAMILIES] + [fmt_ci(get(nm))]
        data[nm] = {f: sum(get(nm, [f])) / 100 for f in FAMILIES} | {"pooled": sum(get(nm)) / 400}
        rows.append(f"| {nm} | " + " | ".join(cells) + " |")
    # sample-to-sample stability of the k=3 Qwen arm
    s = [[sum(vec(f"qwen_prompt_nla_s{k}", [f])) / 100 for k in range(3)] for f in FAMILIES]
    rows.append("| T_qwen per-sample rates s0/s1/s2 | " + " | ".join(
        "/".join(f"{x:.2f}" for x in r) for r in s) + " | |")
    return rows, data


def decomposition(reader: str, texts: list[str]) -> tuple[list[str], dict]:
    rows = [f"| text arm | family | {reader} pass | text pass | P(text pass ∣ reader pass) | "
            "reader-only hits | text-only hits | McNemar p |", "|---|---|---|---|---|---|---|---|"]
    data = {}
    for t in texts:
        for fam in [*FAMILIES, "pooled"]:
            fs = FAMILIES if fam == "pooled" else [fam]
            r, x = get(reader, fs), get(t, fs)
            both = sum(a and b for a, b in zip(r, x))
            b10, b01, p = mcnemar(r, x)
            frac = both / sum(r) if sum(r) else float("nan")
            data[(t, fam)] = dict(reader=sum(r), text=sum(x), both=both, frac=frac,
                                  reader_only=b10, text_only=b01, p=p, n=len(r))
            rows.append(f"| {t} | {fam} | {sum(r)}/{len(r)} | {sum(x)}/{len(r)} | {frac:.2f} | "
                        f"{b10} | {b01} | {p:.2g} |")
    return rows, data


def resolve_table() -> list[str]:
    rows = ["| reader | re-solver | family | reader pass | re-solve pass | uplift (reader ✗, re-solve ✓) "
            "| loss (reader ✓, re-solve ✗) | McNemar p |", "|---|---|---|---|---|---|---|---|"]
    for reader, rs in (("NLA@42", ["Resolve(NLA) Qwen", "Resolve(NLA) Gemini"]),
                       ("OL@42", ["Resolve(OL@42) Qwen", "Resolve(OL@42) Gemini"])):
        for rv in rs:
            for fam in [*FAMILIES, "pooled"]:
                fs = FAMILIES if fam == "pooled" else [fam]
                r, x = get(reader, fs), get(rv, fs)
                b10, b01, p = mcnemar(r, x)
                rows.append(f"| {reader} | {rv.split()[-1]} | {fam} | {sum(r)} | {sum(x)} | {b01} | {b10} | {p:.2g} |")
    return rows


def matched_layer() -> list[str]:
    rows = ["| comparison | family | A pass | B pass | A-only | B-only | McNemar p |",
            "|---|---|---|---|---|---|---|"]
    for a, b in (("NLA@42", "OL@42"), ("NLA@42", "T_qwen"), ("OL@42", "T_qwen"),
                 ("NLA@42", "T_gem"), ("OL@42", "T_gem")):
        for fam in [*FAMILIES, "pooled"]:
            fs = FAMILIES if fam == "pooled" else [fam]
            x, y = get(a, fs), get(b, fs)
            b10, b01, p = mcnemar(x, y)
            rows.append(f"| {a} vs {b} | {fam} | {sum(x)} | {sum(y)} | {b10} | {b01} | {p:.2g} |")
    return rows


def hard_reader_only() -> tuple[list[str], dict]:
    """Items a reader passes that NO text-only arm (T_any, 8 samples over 2 models x 2 templates)
    passes, and that the reader's own re-solvers also do not need (just counted)."""
    rows = ["| reader | family | reader pass | hits also found by T_any | hard reader-only hits (T_any ✗) |",
            "|---|---|---|---|---|"]
    out = {}
    for reader in ("NLA@42", "OL@42", "OL grid (any-of-11)"):
        for fam in [*FAMILIES, "pooled"]:
            fs = FAMILIES if fam == "pooled" else [fam]
            r, t = get(reader, fs), get("T_any", fs)
            ids = [(f, i) for f in fs for i in IDS[f]]
            hard = [ids[j] for j in range(len(r)) if r[j] and not t[j]]
            out[(reader, fam)] = hard
            rows.append(f"| {reader} | {fam} | {sum(r)} | {sum(a and b for a, b in zip(r, t))} | {len(hard)} |")
    return rows, out


def main() -> None:
    md, js = [], {}
    r, js["pass"] = pass_table()
    md += ["## Pass rates (regex mt-regex-2026-09-23, L42, k=1 unless noted; 95% item bootstrap CI)", *r, ""]
    for reader in ("NLA@42", "OL@42"):
        r, d = decomposition(reader, ["T_qwen", "T_qwen_k3", "T_gem", "T_any"])
        js[f"decomp_{reader}"] = {f"{k[0]}|{k[1]}": v for k, v in d.items()}
        md += [f"## Decomposition: {reader} vs text-only arms", *r, ""]
    md += ["## Re-solving ceiling (text-only LLM given only the reader's readout)", *resolve_table(), ""]
    md += ["## Matched-layer (L42) comparisons", *matched_layer(), ""]
    r, hard = hard_reader_only()
    js["hard_reader_only"] = {f"{k[0]}|{k[1]}": v for k, v in hard.items()}
    md += ["## Hard reader-only hits (no text-only arm, 8 samples, finds the target)", *r, ""]
    (HERE / "data" / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (HERE / "data" / "tables.json").write_text(json.dumps(js, indent=1, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
