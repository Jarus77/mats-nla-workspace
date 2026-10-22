"""Build the concept-type x reader recall table from results/judged/*.json."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("RECALL_SPEC", str(HERE.parents[1] / "experiments" / "07_template_mining")))
from feature_spec import FEATURES  # noqa: E402
from recall_lib import READERS, aggregate  # noqa: E402

JUDGED = Path(os.environ.get("RECALL_OUT", HERE / "results")) / "judged"
NAMES = {
    "nj1_col": "line position (chars since newline)", "nj1_col_break": "line break coming",
    "ij2_lang": "language (FR/DE)", "nj2_tense": "tense", "bk_indent": "indentation level",
    "bk_wordpos": "words since sentence start", "bk_listidx": "bullet index (no numerals)",
    "bk_caps": "all-caps vs sentence case", "bk_code": "code language", "ij_listnum": "'Item N' index",
    "bk_quote": "inside a quote", "agr_gender": "upcoming noun gender", "agr_number": "upcoming noun number",
    "nj3_predpos": "next-word POS",
    "tb_sentcount": "sentences into paragraph", "tb_tablecol": "table column", "tb_codeline": "line in function body",
    "tb_paren": "inside a parenthesis", "tb_wordsleft": "words left under limit", "fmt_qa": "Q&A vs prose layout",
    "fmt_listdepth": "list nesting depth", "nj1_col_fresh": "line position (fresh)",
}


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}"


def main() -> None:
    table: dict[str, dict[str, dict]] = {}
    for feature in FEATURES:
        for reader in READERS:
            p = JUDGED / f"{feature}_{reader}.json"
            if p.exists():
                d = json.loads(p.read_text())
                table.setdefault(feature, {})[reader] = aggregate(d["judged"], d["values"])
    readers = [r for r in READERS if any(r in v for v in table.values())]
    head = "| concept type | " + " | ".join(f"{r} stated / correct / any-of-8" for r in readers) + " | " \
        + " | ".join(f"{r} exhibited correct" for r in readers) + " |"
    lines = [head, "|" + "---|" * (1 + 2 * len(readers))]
    for feature, cells in table.items():
        row = [NAMES.get(feature, feature)]
        for r in readers:
            c = cells.get(r)
            row.append("–" if not c else f"{_pct(c['stated_rate'])} / {_pct(c['stated_correct_rate'])} / "
                                         f"{_pct(c['any_correct_per_item'])}")
        for r in readers:
            c = cells.get(r)
            row.append("–" if not c else _pct(c["exhibited_correct_rate"]))
        lines.append("| " + " | ".join(row) + " |")
    md = ("# Spontaneous recall per concept type (exp 07 free readouts, L42, judged)\n\n"
          "stated = % of readouts that explicitly state the property; correct = % stating the true value; "
          "any-of-8 = % of items where at least one of 8 samples states the true value; "
          "exhibited correct = % of readouts whose own text exhibits the true value.\n\n" + "\n".join(lines) + "\n")
    (JUDGED.parent / "recall_table.md").write_text(md)
    (JUDGED.parent / "recall_table.json").write_text(json.dumps(table, indent=1))
    print(md)


if __name__ == "__main__":
    main()
