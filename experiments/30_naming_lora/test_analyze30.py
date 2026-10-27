"""analyze30 on synthetic eval / train dicts: the five prediction rows, the per-row table and the loss figure."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze30 as A  # noqa: E402
from lib30 import ALL_ROWS, TRAIN_ROWS  # noqa: E402


def _eval(acc: float, naming: float, fve: float, rp_true: float, rp_false: float) -> dict:
    rows = {r: {"trained": r in TRAIN_ROWS, "n": 32, "accuracy": acc, "naming_rate": naming, "fve": fve} for r in ALL_ROWS}
    rows["md_list"] = {**rows["md_list"], "accuracy": 0.5}
    return {"rows": rows, "gate": {"fve": fve}, "random_points": {"strict_regex": {"named_when_true": rp_true, "named_when_false": rp_false},
                                                                  "statement_check": {"accuracy": rp_true}}}


def test_prediction_rows_pass_and_fail_at_the_lines():
    ev = {"base": _eval(0.0, 0.0, 0.76, 0.03, 0.01), "C": _eval(0.0, 0.0, 0.75, 0.03, 0.01), "S": _eval(0.9, 0.95, 0.74, 0.6, 0.1)}
    rows = A.prediction_rows(ev, {"skipped": True})
    assert [r[0][:2] for r in rows] == ["P1", "P2", "P3", "P4", "P5"]
    assert [r[3] for r in rows] == ["PASS", "PASS", "FAIL", "PASS", "n/a"]      # P3: latex_math naming 0.95 >= 0.30
    ev2 = {**ev, "S": _eval(0.8, 0.2, 0.70, 0.4, 0.2)}
    assert [r[3] for r in A.prediction_rows(ev2, None)] == ["FAIL", "FAIL", "PASS", "FAIL", "n/a"]
    assert [r[3] for r in A.prediction_rows({"base": ev["base"]}, None)] == ["n/a"] * 5
    p5 = {"skipped": False, "pooled": {"mean_diff": 0.01, "n_items": 100}}
    assert A.prediction_rows(ev, p5)[4][3] == "PASS"


def test_per_row_table_and_figure(tmp_path: Path):
    ev = {"base": _eval(0.0, 0.0, 0.76, 0.03, 0.01), "S": _eval(0.9, 0.95, 0.74, 0.6, 0.1)}
    lines = A.per_row_table(ev)
    assert lines[0].startswith("| row | trained | n | base acc") and len([ln for ln in lines if ln.startswith("| ")]) == 1 + len(ALL_ROWS)
    assert any("no (transfer)" in ln for ln in lines) and "Exp 23" in lines[-1]
    train = {"S": {"losses": [[10, 2.0], [20, 1.5]]}, "C": {"losses": [[10, 2.0], [20, 1.8]]}}
    assert A.figure(train, tmp_path / "fig.png") and (tmp_path / "fig.png").stat().st_size > 1000
    assert A.figure({}, tmp_path / "none.png") is False
