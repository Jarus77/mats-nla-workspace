#!/bin/bash
# Exp 24 chain: launch the OL-AR reconstruction stage (detached), wait for the ten row files on the volume, pull
# them, build the tables. Idempotent: rows already in results/ are not re-fetched; a stage whose outputs already
# exist on the volume is not relaunched (and modal24.py skips finished rows itself). Logs to results/pipeline.log.
set -u
cd "$(dirname "$0")"
mkdir -p results
LOG=results/pipeline.log
PY=../../.venv/bin/python
VOL=rfm-results
DIR=24_second_decoder
ROWS="py_string js_string latex_math py_arg json_depth html_depth md_heading md_list tb_tablecol tb_paren"
MAX_POLLS=240   # 60 s each: four hours, the stage's own timeout

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

remote_listing() { modal volume ls "$VOL" "$DIR" 2>/dev/null; }

missing_remote() {   # rows whose recon24_{row}.json is not yet on the volume
  local ls; ls=$(remote_listing)
  for r in $ROWS; do grep -q "recon24_$r.json" <<<"$ls" || echo "$r"; done
}

missing_local() { for r in $ROWS; do [ -s "results/recon24_$r.json" ] || echo "$r"; done; }

log "run24 start"
if [ -z "$(missing_local)" ]; then
  log "all ten recon24 files already in results/; skipping launch and fetch"
else
  if [ -z "$(missing_remote)" ]; then
    log "all ten recon24 files already on the volume; skipping launch"
  else
    log "launching: modal run --detach modal24.py --stage recon (missing: $(missing_remote | tr '\n' ' '))"
    modal run --detach modal24.py --stage recon >> "$LOG" 2>&1 || { log "LAUNCH FAILED"; exit 1; }
  fi
  polls=0
  while [ -n "$(missing_remote)" ]; do
    polls=$((polls + 1))
    [ "$polls" -ge "$MAX_POLLS" ] && { log "GAVE UP after $polls polls; still missing: $(missing_remote | tr '\n' ' ')"; exit 1; }
    log "poll $polls: waiting for $(missing_remote | tr '\n' ' ')"
    sleep 60
  done
  for r in $(missing_local); do
    modal volume get --force "$VOL" "$DIR/recon24_$r.json" "results/recon24_$r.json" >> "$LOG" 2>&1 \
      && log "fetched recon24_$r.json" || log "FETCH FAILED recon24_$r.json"
  done
fi
"$PY" analyze24.py >> "$LOG" 2>&1 && log "table24.md written" || { log "ANALYZE FAILED"; exit 1; }
log "run24 done"
