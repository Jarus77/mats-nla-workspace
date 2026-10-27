#!/bin/bash
# Exp 29 chain, one GPU at a time: fetch the missing adapters (CPU, minutes) -> stage text (H100, the TextNLA arm)
# -> stage gen per checkpoint (H200; the first launch also runs the critic over the textnla readouts) -> download
# -> judge (laptop, OpenRouter) -> analyze. Idempotent: a stage whose outputs are already on the volume is not
# launched again, the Modal functions skip (checkpoint, row) cells whose output exists, and the judge skips complete
# cells. Logs to results/pipeline.log.
#   cd experiments/29_trajectory && nohup ./run29.sh > /dev/null 2>&1 &
# Env: SKIP_FETCH=1 to use only the adapters already on the volume; CKPTS="iter_000300 iter_000400" to restrict the
# RL checkpoints (av_base and textnla always run); JUDGE_CAP (dollars, default 15).
set -u
cd "$(dirname "$0")"
mkdir -p results
LOG=results/pipeline.log
VOL=rfm-results
DIR=29_trajectory
WVOL=rfm-weights
SNAP="hf/hub/models--ceselder--qwen3.6-27b-nla-rl/snapshots/5a13b7ec21a69fcdd0fb24d5edbe96a92aef4b9f"
ROWS="py_string js_string latex_math py_arg json_depth md_heading md_list code_lang"
PY=../../.venv/bin/python
MAX_POLLS=260                                    # 60 s each: a little over the 4 h function timeout
JUDGE_CAP=${JUDGE_CAP:-15}

log() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }
listing() { modal volume ls "$VOL" "$DIR" 2>/dev/null || true; }

have_all() {                                     # $1 = file prefix (e.g. recon29_av_base); 0 iff all eight rows are on the volume
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

launch() {                                       # $1 = log name, rest = modal run arguments
  local name=$1; shift
  log "launching $name: modal run --detach modal29.py $*"
  modal run --detach modal29.py "$@" >> "results/$name.log" 2>&1 &
}

# 1. adapters: fetch the missing iter_* into the volume snapshot (pinned revision; nothing deleted), then list them
if [ "${SKIP_FETCH:-0}" != "1" ]; then
  log "fetch: downloading missing adapters"
  modal run modal29.py --stage fetch >> results/fetch.log 2>&1 || log "fetch FAILED (see results/fetch.log); continuing with what is on the volume"
fi
ITERS=$(modal volume ls "$WVOL" "$SNAP/av_rl_adapters" 2>/dev/null | grep -o 'iter_[0-9]\{6\}' | sort -u | tr '\n' ' ')
if [ -n "${CKPTS:-}" ]; then ITERS=$CKPTS; fi
log "RL adapters on the volume: $ITERS"
[ -z "$ITERS" ] && { log "no iter_* adapters found; aborting"; exit 1; }

# 2. TextNLA arm (H100)
if have_all gen29_textnla; then log "text outputs already on the volume; not launching"; else launch text --stage text; fi
wait_all gen29_textnla; log "text done"

# 3. av_base + the critic over textnla, then every RL checkpoint (H200, one container each)
if have_all recon29_av_base && have_all recon29_textnla; then log "av_base/textnla recon on the volume; not launching"
else launch gen_av_base --stage gen --ckpts av_base,textnla; fi
wait_all recon29_av_base; wait_all recon29_textnla; log "av_base + textnla done"
for it in $ITERS; do
  if have_all "recon29_$it"; then log "$it recon on the volume; not launching"; else launch "gen_$it" --stage gen --ckpts "$it"; fi
  wait_all "recon29_$it"; log "$it done"
done

# 4. download
for ck in textnla av_base $ITERS; do
  for r in $ROWS; do
    for p in gen29 recon29; do
      modal volume get --force "$VOL" "$DIR/${p}_${ck}_${r}.json" "results/${p}_${ck}_${r}.json" >> "$LOG" 2>&1 || log "download of ${p}_${ck}_${r}.json FAILED"
    done
  done
done
log "downloaded"

# 5. judge (laptop) and analyze
if $PY judge29.py --cap "$JUDGE_CAP" >> results/judge_run.log 2>&1; then log "judged"; else log "judge FAILED (see results/judge_run.log)"; fi
if $PY analyze29.py >> results/analyze.log 2>&1; then log "analyzed"; else log "analyze FAILED (see results/analyze.log)"; exit 1; fi
log "DONE"
