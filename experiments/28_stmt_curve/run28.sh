#!/bin/bash
# Exp 28 chain: paraphrases (laptop, Gemini; skipped if results/paraphrases28.json exists) -> upload rows28.json and
# paraphrases28.json to the volume -> stage stmt (H200, detached) -> poll for every stmt28_{key}.json -> fetch ->
# analyze28.py. Idempotent: the Modal function skips rows whose file exists; files already in results/ are not
# re-fetched. Logs to results/pipeline.log.
#   cd experiments/28_stmt_curve && nohup ./run28.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
mkdir -p results
LOG=results/pipeline.log
VOL=rfm-results
DIR=28_stmt_curve
PY=../../.venv/bin/python
MAX_POLLS=250                                    # 60 s each: a little over the 4 h function timeout

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }
listing() { modal volume ls "$VOL" "$DIR" 2>/dev/null || true; }
keys() { "$PY" -c 'import json; print(" ".join(r["key"] for r in json.load(open("results/rows28.json"))))'; }

log "run28 start"
if [ -s results/paraphrases28.json ]; then
  log "paraphrases28.json present (sha256 $(cut -d' ' -f1 results/paraphrases28.sha256 2>/dev/null)); not regenerating"
else
  log "generating paraphrases"
  "$PY" paraphrase28.py >> results/paraphrase.log 2>&1 || { log "PARAPHRASE FAILED (see results/paraphrase.log)"; exit 1; }
  [ -s results/paraphrases28.json ] || { log "paraphrases28.json not written (rows left?); see results/paraphrase.log"; exit 1; }
fi
[ -s results/rows28.json ] || { log "rows28.json missing"; exit 1; }
KEYS=$(keys)

missing_remote() { local l; l=$(listing); for k in $KEYS; do echo "$l" | grep -q "stmt28_${k}.json" || echo "$k"; done; }
missing_local() { for k in $KEYS; do [ -s "results/stmt28_${k}.json" ] || echo "$k"; done; }

if [ -z "$(missing_local)" ]; then
  log "all stmt28 files already in results/; skipping upload, launch and fetch"
else
  if [ -z "$(missing_remote)" ]; then
    log "all stmt28 files already on the volume; skipping launch"
  else
    for f in rows28.json paraphrases28.json; do
      modal volume put --force "$VOL" "results/$f" "$DIR/$f" >> "$LOG" 2>&1 && log "uploaded $f" || { log "UPLOAD FAILED $f"; exit 1; }
    done
    log "launching: modal run --detach modal28.py --stage stmt (missing: $(missing_remote | wc -l | tr -d ' ') rows)"
    modal run --detach modal28.py --stage stmt >> results/stmt.log 2>&1 &
  fi
  polls=0
  while [ -n "$(missing_remote)" ]; do
    polls=$((polls + 1))
    [ "$polls" -ge "$MAX_POLLS" ] && { log "GAVE UP after $polls polls; still missing: $(missing_remote | tr '\n' ' ')"; exit 1; }
    [ $((polls % 5)) -eq 0 ] && log "poll $polls: $(missing_remote | wc -l | tr -d ' ') rows still missing"
    sleep 60
  done
  for k in $(missing_local); do
    modal volume get --force "$VOL" "$DIR/stmt28_${k}.json" "results/stmt28_${k}.json" >> "$LOG" 2>&1 \
      && log "fetched stmt28_${k}.json" || log "FETCH FAILED stmt28_${k}.json"
  done
fi
if "$PY" analyze28.py >> results/analyze.log 2>&1; then log "analyzed (table28.md, fig28.png, stats28.json)"; else log "ANALYZE FAILED (see results/analyze.log)"; exit 1; fi
log "run28 done"
