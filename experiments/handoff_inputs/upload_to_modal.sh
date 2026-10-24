#!/usr/bin/env bash
# Put the exp 15/16 input files onto the Modal results volume (rfm-results) in the layout the code expects.
# Usage: bash experiments/handoff_inputs/upload_to_modal.sh        (optionally: MODAL_PROFILE=<profile>)
set -euo pipefail
cd "$(dirname "$0")"
modal volume create rfm-results >/dev/null 2>&1 || true
modal volume create rfm-weights >/dev/null 2>&1 || true
find . -type f \( -name '*.pt' -o -name '*.json' \) | sed 's|^\./||' | while read -r f; do
  echo "put $f"; modal volume put --force rfm-results "$f" "$f" >/dev/null
done
echo "done: $(find . -type f \( -name '*.pt' -o -name '*.json' \) | wc -l) files on rfm-results"
