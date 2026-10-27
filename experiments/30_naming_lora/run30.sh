#!/bin/bash
# Exp 30 chain, one GPU at a time: data (CPU) -> train S -> train C -> eval base -> eval S -> eval C -> fetch ->
# p5 baseline -> analyze. Idempotent: a stage whose output is on the volume is not launched again, the Modal functions
# skip finished work themselves, and a training stage relaunched after a crash resumes from its last checkpoint.
# Logs to results/pipeline.log.
#   cd experiments/30_naming_lora && nohup ./run30.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
mkdir -p results
LOG=results/pipeline.log
VOL=rfm-results
DIR=30_naming_lora
PY=../../.venv/bin/python
MAX_POLLS=260                                    # 60 s each: a little over the 4 h function timeout

log() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }
listing() { modal volume ls "$VOL" "$DIR" 2>/dev/null || true; }
have() { listing | grep -q "$1"; }               # $1 = file name on the volume

wait_for() {                                     # $1 = file name; polls every 60 s
  local n=0
  until have "$1"; do
    n=$((n + 1))
    if [ "$n" -ge "$MAX_POLLS" ]; then log "GAVE UP waiting for $1"; exit 1; fi
    sleep 60
  done
  log "seen $1"
}

run_stage() {                                    # $1 = stage, $2 = arm ("" for data), $3 = output file name
  local tag="$1${2:+_$2}"
  if have "$3"; then
    log "$tag output already on the volume; not launching"
  else
    log "launching $tag"
    if [ -n "$2" ]; then
      modal run --detach modal30.py --stage "$1" --arm "$2" >> "results/$tag.log" 2>&1 &
    else
      modal run --detach modal30.py --stage "$1" >> "results/$tag.log" 2>&1 &
    fi
  fi
  wait_for "$3"
  log "$tag done"
}

run_stage data "" data30.sha256
run_stage train S train30_S.done
run_stage train C train30_C.done
run_stage eval base eval30_base.json
run_stage eval S eval30_S.json
run_stage eval C eval30_C.json
for f in data30.json data30.sha256 train30_S.json train30_C.json eval30_base.json eval30_S.json eval30_C.json; do
  modal volume get --force "$VOL" "$DIR/$f" "results/$f" >> "$LOG" 2>&1 || log "download of $f FAILED"
done
cp results/data30.sha256 results/frozen_sha256.txt 2>/dev/null || true
log "downloaded"
$PY p5_baseline.py >> results/p5.log 2>&1 || log "p5 baseline FAILED (see results/p5.log)"
if $PY analyze30.py >> results/analyze.log 2>&1; then log "analyzed"; else log "analyze FAILED (see results/analyze.log)"; exit 1; fi
log "DONE"
