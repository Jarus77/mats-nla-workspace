"""Rescore exp 22's scorecard with the reconstructor-based "exhibits" measure: the `ol_exhibits` cell of every row
is re-observed as the frozen-probe AUROC of the critic's rebuild from Oracle Lens's free readout
(`results/recon25_{row}.json`, `auroc.ol_readout`), at exp 20's 0.70 line; every other cell, every prediction and
every threshold is exp 22's (`CELL_THRESH`), read from experiments/22_public_bank/results/score22.json.

    .venv/bin/python rescore25.py      -> results/score25.json, results/scorecard25.md

Fails before writing anything if any of the eight reconstruction files is missing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SCORE22 = HERE.parent / "22_public_bank" / "results" / "score22.json"
sys.path.insert(0, str(HERE))
from lib25 import (EXCLUDE, EXHIBITS_LINE, PRIMARY_CELLS, ROWS, SECONDARY_CELLS, accuracy_split,  # noqa: E402
                   per_row_lines, score22, swap_exhibits)


def load_recon(res: Path = RES) -> dict[str, dict]:
    paths = {row: res / f"recon25_{row}.json" for row in ROWS}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"rescore25: {len(missing)} of {len(ROWS)} reconstruction files missing in {res}: "
                         f"{', '.join(missing)}. Run run25.sh (both Modal stages) and download them first; "
                         "no scorecard written.")
    return {row: json.loads(p.read_text()) for row, p in paths.items()}


def _fmt(a: dict) -> str:
    return f"{a['correct']} of {a['n']}, {a['point']:.2f} [{a['lo']:.2f}, {a['hi']:.2f}]"


def rescore(old_rows: list[dict], recon: dict[str, dict], line: float = EXHIBITS_LINE) -> tuple[dict, str]:
    S = score22()
    new_rows = swap_exhibits(old_rows, recon, line=line)
    cells = [r["cells"] for r in new_rows]
    acc, conf = S.accuracy(cells), S.confusion(cells)
    splits = accuracy_split(new_rows, PRIMARY_CELLS, SECONDARY_CELLS, EXCLUDE)
    splits22 = accuracy_split(old_rows, PRIMARY_CELLS, SECONDARY_CELLS, EXCLUDE)
    no_cl = accuracy_split([r for r in new_rows if r["feature"] != "code_lang"], PRIMARY_CELLS, SECONDARY_CELLS, EXCLUDE)
    lines = per_row_lines(old_rows, new_rows)
    by_feat = {r["feature"]: r for r in new_rows}
    per_row = [{"feature": r["feature"], "judge": r["cells"]["ol_exhibits"]["observed"],
                "recon": by_feat[r["feature"]]["cells"]["ol_exhibits"]["observed"],
                "predicted_high": r["cells"]["ol_exhibits"]["predicted_high"],
                "correct": by_feat[r["feature"]]["cells"]["ol_exhibits"]["correct"]} for r in old_rows]
    out = {"rows": new_rows, "accuracy": acc, "confusion": conf,
           "splits": splits, "splits_exp22_judge": splits22, "without_code_lang": no_cl,
           "exhibits": {"line": line, "source": "recon25_{row}.json auroc.ol_readout (frozen-probe AUROC of the "
                        "critic's rebuild from Oracle Lens's k=1 readout, exp 20's measure)", "per_row": per_row}}
    md = [f"# Exp 25 scorecard: exp 22 with `ol_exhibits` observed by the reconstructor measure (exp 20's, line {line:.2f})", "",
          "Every other cell, prediction and threshold is exp 22's (`score22.json`); only the `ol_exhibits` column changes.", "",
          S.markdown(new_rows, acc, conf).rstrip(), "",
          f"**Primary (free-readout cells, J-lens on the six uncued rows): {_fmt(splits['primary'])}** "
          f"(exp 22 with the judge-scored cell: {_fmt(splits22['primary'])}).", "",
          f"**Secondary (prompted forced-choice cells): {_fmt(splits['secondary'])}** (unchanged from exp 22).", "",
          f"All {acc['n']} cells, the exp 18 convention: {_fmt(acc)} (exp 22: {_fmt(splits22['all'])}).", "",
          f"With vs without code_lang: primary {splits['primary']['correct']} of {splits['primary']['n']} vs "
          f"{no_cl['primary']['correct']} of {no_cl['primary']['n']}; secondary {splits['secondary']['correct']} of "
          f"{splits['secondary']['n']} vs {no_cl['secondary']['correct']} of {no_cl['secondary']['n']}.", "",
          "Per row, the `ol_exhibits` cell (judge exhibited-correct rate -> reconstructor AUROC of the Oracle Lens readout):", ""]
    md += [f"- {ln}" for ln in lines]
    return out, "\n".join(md) + "\n"


def main() -> None:
    old_rows = json.loads(SCORE22.read_text())["rows"]
    recon = load_recon(RES)
    out, md = rescore(old_rows, recon)
    (RES / "score25.json").write_text(json.dumps(out, indent=1, default=float))
    (RES / "scorecard25.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
