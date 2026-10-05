#!/usr/bin/env bash
# Pull readouts from the mats-results volume, split into judge views, judge locally with the fork.
#   bash judge.sh "typo_mt,multilingual_mt"
# jlens needs OPENROUTER_API_KEY (Gemini summarizer); prose arms are regex-only (0 calls).
set -euo pipefail
FAMS="${1:-typo_mt,multilingual_mt}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
FORK="$ROOT/third_party/workspace-bench"
cd "$HERE"

for arm in olens jlens nla; do
  for fam in ${FAMS//,/ }; do
    mkdir -p "readouts/$arm"
    modal volume get --force mats-results "01_wsbench_replication/readouts/$arm/$fam.jsonl" "readouts/$arm/$fam.jsonl" >/dev/null 2>&1 \
      || echo "missing on volume: $arm/$fam"
  done
done
python3 split_readouts.py >/dev/null

set -a; source "$ROOT/.env"; set +a
cd "$FORK"
judge() {  # $1 = judge_in dir name
  uv run wsbench run families="$FAMS" readouts_root="$HERE/judge_in/$1" out="$HERE/judged/$1" 2>&1 | tail -3
}
# jlens: summarize every cell once (12 layers) then reuse the summary cache for each view
mkdir -p "$HERE/judge_in/jlens_all"
for fam in ${FAMS//,/ }; do
  [ -f "$HERE/readouts/jlens/$fam.jsonl" ] && cp "$HERE/readouts/jlens/$fam.jsonl" "$HERE/judge_in/jlens_all/"
done
judge jlens_all
for v in grid L40 L42 L44; do
  for fam in ${FAMS//,/ }; do
    mkdir -p "$HERE/judged/jlens_$v/$fam"
    cp "$HERE/judged/jlens_all/$fam/cells.jsonl" "$HERE/judged/jlens_$v/$fam/cells.jsonl" 2>/dev/null || true
  done
  judge "jlens_$v"
done
mkdir -p "$HERE/judge_in/olens_all"; cp "$HERE"/readouts/olens/*.jsonl "$HERE/judge_in/olens_all/"
for d in olens_all olens_grid olens_L40 olens_L42 olens_L44 nla_L42; do judge "$d"; done
python3 "$HERE/summarize.py"
