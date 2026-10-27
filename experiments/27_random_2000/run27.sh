#!/bin/bash
# Exp 27 chain: build the bank if missing (needs ARXIV=<dir with src_*/ e-print folders>), upload it if not on the
# volume, stage ol (H100) -> stage nla (H200) strictly one after the other, fetch, analyse, judge (laptop, no GPU),
# analyse again to merge the judge table. Idempotent: a stage whose output is on the volume is not relaunched (and
# modal27.py skips itself too), files already fetched are not re-fetched, the judge resumes from its cache.
# Logs to results/pipeline.log.
#   cd experiments/27_random_2000 && ARXIV=<dir> nohup ./run27.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
mkdir -p results results/vol
LOG=results/pipeline.log
PY=../../.venv/bin/python
VOL=rfm-results
DIR=27_random_2000
MAX_POLLS=190                                    # 60 s each: a little over the 3 h function timeout

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }
listing() { modal volume ls "$VOL" "$DIR" 2>/dev/null || true; }
on_volume() { listing | grep -q "$1"; }

wait_for() {                                     # $1 = file name on the volume
  local n=0
  until on_volume "$1"; do
    n=$((n + 1))
    [ "$n" -ge "$MAX_POLLS" ] && { log "GAVE UP waiting for $1 after $n polls"; exit 1; }
    sleep 60
  done
  log "seen $1"
}

run_stage() {                                    # $1 = stage key, $2 = output file
  if on_volume "$2"; then
    log "$1: $2 already on the volume; not launching"
  else
    log "launching $1"
    modal run --detach modal27.py --stage "$1" >> "results/$1.log" 2>&1 || { log "LAUNCH of $1 FAILED (see results/$1.log)"; exit 1; }
  fi
  wait_for "$2"
  log "$1 done"
}

fetch() {                                        # $1 = file on the volume, $2 = local path
  if [ -s "$2" ]; then log "$2 already fetched"; return 0; fi
  modal volume get --force "$VOL" "$DIR/$1" "$2" >> "$LOG" 2>&1 && log "fetched $1" || { log "FETCH of $1 FAILED"; exit 1; }
}

log "run27 start"
if [ ! -s results/items_base.json ]; then
  [ -n "${ARXIV:-}" ] || { log "no results/items_base.json and ARXIV is not set; cannot build the bank"; exit 1; }
  log "building the bank (seed 2727, 400 per kind)"
  "$PY" build27.py --arxiv "$ARXIV" >> results/build.log 2>&1 && log "bank built: $(tail -1 results/build.log)" || { log "BUILD FAILED (see results/build.log)"; exit 1; }
fi
if on_volume items_base.json; then
  log "bank already on the volume"
else
  modal volume put "$VOL" results/items_base.json "$DIR/items_base.json" >> "$LOG" 2>&1 && log "uploaded" || { log "UPLOAD FAILED"; exit 1; }
fi
run_stage ol ol27.json
run_stage nla nla27.json
fetch ol27.json results/ol27.json
fetch nla27.json results/nla27.json
fetch acts27.pt results/vol/acts27.pt
if "$PY" analyze27.py >> results/analyze.log 2>&1; then log "analyzed (readers only)"; else log "ANALYZE FAILED (see results/analyze.log)"; exit 1; fi
if "$PY" judge27.py >> results/judge.log 2>&1; then log "judged: $(tail -1 results/judge.log)"; else log "JUDGE FAILED (see results/judge.log)"; exit 1; fi
if "$PY" analyze27.py >> results/analyze2.log 2>&1; then log "analyzed (judge merged)"; else log "ANALYZE 2 FAILED (see results/analyze2.log)"; exit 1; fi
log "DONE"
