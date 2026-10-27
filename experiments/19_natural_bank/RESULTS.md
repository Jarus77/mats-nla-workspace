# Exp 19: the prediction test on a natural-text bank, with the revised NLA rules

Pre-registration: `notes/prereg_natural_bank.md` (rules written 27 Oct before any instrument ran; predictions filled
and hashed before any reader ran). Pipeline: exp 18's, cloned (`modal19.py`, app `k3rw`); the bank is built on the
laptop by `build19.py` from public package sources on this machine and fresh arXiv e-prints, and uploaded.

## Bank (27 Oct)

Eight rows, 200 read points each (20 + 20 mining, 80 + 80 selection), both values of a row from the same source
files, clustered by file. sha256 `7ddfbceac1a5510156b1037239606536e50484a61f93c057629c26fa41525460`. A CSV row was
dropped at build time (16 items: the CSV files here are numeric fixtures).

| row | source files | lexical cue (bag of word tokens, last 48) | J-lens cell scored |
|---|---|---|---|
| py_string | 196 .py | 0.47 | yes |
| js_string | 193 .js | 0.56 | yes |
| latex_math | 157 arXiv .tex | 0.56 | yes |
| py_arg | 192 .py | 0.49 | yes |
| json_depth | 184 .json | 0.94 | no (deep and top-level strings use different words) |
| html_depth | 194 .html | 1.00 | no (deep text is navigation and list items, shallow is paragraphs) |
| md_heading | 189 .md | 0.45 | yes |
| md_list | 179 .md | 0.52 | yes |

## Instruments

(pending)

## Scorecard

(pending)
