import random
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import items_build19 as IB  # noqa: E402

PAD = ''.join(f'# preamble line {i} with some ordinary words in it\n' for i in range(6))
PY = PAD + '''import os
def load(path, mode, retries):
    msg = "loading the configuration file from disk now"
    other = compute(alpha, beta, gamma)
    result = process(path, mode, retries)
    label = 'a second string literal with several words'
    return result, label
'''
JS = PAD.replace('#', '//') + '''const a = "the first string literal in this file";
const b = render(alpha, beta, gamma);
let c = 'another string with enough words inside';
export default b;
'''
MD = ''.join(f'Intro paragraph line {i} with ordinary words here.\n' for i in range(6)) + '''# Title of the document here
Some intro text goes here.
## Second level heading with words
- item one is the first entry here
  - nested item two
    - nested item three deep
### Third level heading with several words
#### Fourth level heading with more words here
'''
JSON = '{"preamble": "' + 'x' * 160 + '", "name": "alpha", "config": {"db": {"host": "localhost value", "port": "5432"}}, "note": "top level string"}'
CSV = ''.join(f'row{i},Lyon,staff,north,active\n' for i in range(8)).replace('row0', 'name') + '''name,city,role,team,status
alice,Lyon,staff,north,active
bruno,Prague,lead,central,retired
'''


def test_py_string_inside_and_outside_points():
    items = IB.build_py_string([("f.py", PY)], random.Random(0))
    assert items and {it["value"] for it in items} == {"inside", "outside"}
    for it in items:
        quote = it["meta"]["quote"]
        assert (it["text"].count(quote) % 2 == 1) == (it["value"] == "inside")
        assert not it["text"].endswith((" ", "\n"))


def test_py_arg_first_and_third():
    items = IB.build_py_arg([("f.py", PY)], random.Random(0))
    assert items and {it["value"] for it in items} == {"first", "third"}
    for it in items:
        tail = it["text"][it["text"].rfind("(") + 1:]
        assert tail.count(",") == {"first": 0, "third": 2}[it["value"]] and ")" not in tail


def test_js_string():
    items = IB.build_js_string([("f.js", JS)], random.Random(0))
    assert items and {it["value"] for it in items} == {"inside", "outside"}


def test_md_heading_levels():
    items = IB.build_md_heading([("r.md", MD)], random.Random(0))
    assert {it["value"] for it in items} == {"shallow", "deep"}
    for it in items:
        last = it["text"].splitlines()[-1]
        hashes = len(last) - len(last.lstrip("#"))
        assert (hashes <= 2) == (it["value"] == "shallow")


def test_md_list_depth():
    items = IB.build_md_list([("r.md", MD)], random.Random(0))
    assert {it["value"] for it in items} == {"top", "nested"}
    for it in items:
        last = it["text"].splitlines()[-1]
        assert (last.startswith(("- ", "* "))) == (it["value"] == "top")


def test_json_depth_scanner():
    items = IB.build_json_depth([("c.json", JSON)], random.Random(0))
    assert {it["value"] for it in items} == {"deep", "shallow"}
    for it in items:
        assert it["text"].endswith('"')
        depth = it["text"].count("{") - it["text"].count("}")
        assert (depth >= 3) == (it["value"] == "deep")


def test_csv_field():
    items = IB.build_csv_field([("d.csv", CSV)], random.Random(0))
    assert {it["value"] for it in items} == {"first", "fourth"}
    for it in items:
        last = it["text"].splitlines()[-1]
        assert last.count(",") == {"first": 0, "fourth": 3}[it["value"]]


def test_context_and_dedup():
    items = IB.build_py_string([("f.py", PY), ("g.py", PY)], random.Random(0))
    assert len({it["text"] for it in items}) == len(items)
    assert all(it["cluster"] in ("f.py", "g.py") for it in items)


HTML = '<html><body>\n' + ''.join(f'<p>intro paragraph number {i} with ordinary words here</p>\n' for i in range(5)) + '''<div><section><ul><li>first nested item with several words inside it</li>
<li>second nested item with more ordinary words</li></ul></section></div>
<p>a plain paragraph with ordinary words after the list</p>
</body></html>'''
TEX = ''.join(f'Intro sentence number {i} with ordinary words here.\n' for i in range(5)) + r'''We write the energy as $E = m c^2 + \frac{p^2}{2m}$ for every particle in the box.
Then the mass term $m \cdot v$ enters the sum over all the states we consider here.
'''


def test_html_depth_shallow_vs_deep():
    items = IB.build_html_depth([("a.html", HTML)], random.Random(0))
    assert {it["value"] for it in items} == {"shallow", "deep"}
    for it in items:
        opened = len(re.findall(r"<([a-z]+)[^>]*>", it["text"])) - len(re.findall(r"</([a-z]+)>", it["text"]))
        assert (opened >= IB.HTML_DEEP) == (it["value"] == "deep")
        assert opened >= IB.HTML_DEEP or opened <= IB.HTML_SHALLOW


def test_latex_math_inside_vs_outside():
    items = IB.build_latex_math([("a.tex", TEX)], random.Random(0))
    assert {it["value"] for it in items} == {"inside", "outside"}
    for it in items:
        assert (it["text"].count("$") % 2 == 1) == (it["value"] == "inside")
