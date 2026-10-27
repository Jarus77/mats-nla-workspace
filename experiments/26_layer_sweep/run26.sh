#!/bin/bash
# Exp 26 chain: stage ol (H100) -> stage recon (H200), strictly one after the other (one GPU at a time), then
# download and analyse. Idempotent: a stage whose eight outputs are already on the volume is not launched again, and
# the Modal functions themselves skip rows (and per-layer readout files) whose output exists. Logs to results/pipeline.log.
#   cd experiments/26_layer_sweep && nohup ./run26.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
mkdir -p results
LOG=results/pipeline.log
VOL=rfm-results
DIR=26_layer_sweep
ROWS="py_string js_string latex_math py_arg json_depth md_heading md_list code_lang"
LAYERS="20 24 28 32 36 40 42 44 48 52 56 60"
PY=../../.venv/bin/python
MAX_POLLS=260                                    # 60 s each: a little over the 4 h function timeout

log() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }
listing() { modal volume ls "$VOL" "$DIR" 2>/dev/null || true; }

have_all() {                                     # $1 = file prefix; 0 iff all eight rows' files are on the volume
  local l; l=$(listing)
  for r in $ROWS; do echo "$l" | grep -q "${1}_${r}.json" || return 1; done
  return 0
}

wait_all() {                                     # $1 = file prefix; polls every 60 s, logs each file once
  local prefix=$1 seen="" l f missing n=0
  while true; do
    l=$(listing); missing=0
    for r in $ROWS; do
      f="${prefix}_${r}.json"
      if echo "$l" | grep -q "$f"; then
        case " $seen " in *" $f "*) ;; *) seen="$seen $f"; log "seen $f" ;; esac
      else missing=1; fi
    done
    [ "$missing" -eq 0 ] && return 0
    n=$((n + 1))
    if [ "$n" -ge "$MAX_POLLS" ]; then log "GAVE UP waiting for ${prefix}_*.json"; exit 1; fi
    sleep 60
  done
}

run_stage() {                                    # $1 = stage key, $2 = output prefix
  if have_all "$2"; then
    log "$1 outputs already on the volume; not launching"
  else
    log "launching $1"
    modal run --detach modal26.py --stage "$1" >> "results/$1.log" 2>&1 &
  fi
  wait_all "$2"
  log "$1 done"
}

get() {                                          # $1 = file name on the volume (under $DIR), saved under results/
  modal volume get --force "$VOL" "$DIR/$1" "results/$1" >> "$LOG" 2>&1 || log "download of $1 FAILED"
}

run_stage ol sweep26a
run_stage recon sweep26
for r in $ROWS; do
  for p in sweep26a sweep26; do get "${p}_${r}.json"; done
  for L in $LAYERS; do get "ol26_${r}_L${L}.json"; done
done
log "downloaded"
if $PY analyze26.py >> results/analyze.log 2>&1; then log "analyzed"; else log "analyze FAILED (see results/analyze.log)"; exit 1; fi
log "DONE"
