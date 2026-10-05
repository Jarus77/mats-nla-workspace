"""multilingual_mt: option sets and judged roles of the optional MC judge (``opts=judge=mc``);
the regex scorer of record is tested in ``test_mt_regex.py``."""

import json
from pathlib import Path

from wsbench import registry
from wsbench.banks import load_bank
from wsbench.multitoken.options import judged_roles, option_sets
from wsbench.multitoken.prompts import CANNOT

REPO = Path(__file__).resolve().parents[1]


def test_options_match_the_golden_and_the_bank():
    header, items = load_bank(REPO / "evals/multilingual_mt/items.json")
    assert header["family"] == "multilingual-hard" and len(items) == 100
    opts = option_sets(items)
    golden = json.loads(
        (REPO / "tests/golden/multilingual_mt_options.json").read_text(encoding="utf-8")
    )
    want = {
        i: {r: (v["options"], v["gold_index"]) for r, v in per.items()}
        for i, per in golden["options"].items()
    }
    assert want == opts
    n_lang = 0
    for it in items:
        roles = judged_roles(it)
        assert set(roles) == set(opts[it["id"]])
        n_lang += "language" in roles
        for role, (o, g) in opts[it["id"]].items():
            assert len(o) == 6 and o[-1] == CANNOT and 0 <= g < 5 and len(set(o)) == 6
            if role != "language" and role in (it.get("mc") or {}):
                assert o[g] == it["mc"][role]["gold"]
    assert n_lang == 100


def test_registered():
    from wsbench.evals.multilingual_mt import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)
    assert SPEC.group == "basic_mt" and SPEC.scorer == "regex"
    assert (
        SPEC.judge.prompt_version == "mt-regex-2026-09-23"
    )  # the regex contract is the instrument
    assert "0 (regex)" in SPEC.calls_per_arm and "opts=judge=mc" in SPEC.calls_per_arm
