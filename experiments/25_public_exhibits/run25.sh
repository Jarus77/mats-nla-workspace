#!/bin/bash
# Exp 25 chain: stage ol (H100) -> stage recon (H200), strictly one after the other (one GPU at a time), then
# download and analyse. Idempotent: a stage whose eight outputs are already on the volume is not launched again, and
# the Modal functions themselves skip rows whose output exists. Logs to results/pipeline.log.
#   cd experiments/25_public_exhibits && nohup ./run25.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
mkdir -p results
LOG=results/pipeline.log
VOL=rfm-results
DIR=25_public_exhibits
ROWS="py_string js_string latex_math py_arg json_depth md_heading md_list code_lang"
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
    modal run --detach modal25.py --stage "$1" >> "results/$1.log" 2>&1 &
  fi
  wait_all "$2"
  log "$1 done"
}

run_stage ol ol_sel
run_stage recon recon25
for r in $ROWS; do
  for p in ol_sel nla_sel recon25; do
    modal volume get --force "$VOL" "$DIR/${p}_${r}.json" "results/${p}_${r}.json" >> "$LOG" 2>&1 || log "download of ${p}_${r}.json FAILED"
  done
done
log "downloaded"
if $PY analyze25.py >> results/analyze.log 2>&1; then log "analyzed"; else log "analyze FAILED (see results/analyze.log)"; exit 1; fi
log "DONE"
