#!/bin/bash
# Count-based watcher: tolerates transient status failures; stops after the call is done or 40 failed polls.
set -u
cd "$(dirname "$0")"
fails=0
while true; do
  s=$(/Users/surajprasad/miniconda3/bin/python spawn_driver.py status 2>&1 | tail -1)
  echo "$(date -u +%H:%M:%S) $s" >> results/watch.log
  case "$s" in
    done*) break ;;
    running*) fails=0 ;;
    *) fails=$((fails+1)); [ "$fails" -ge 40 ] && { echo "GAVE UP: $s" >> results/watch.log; exit 1; } ;;
  esac
  sleep 60
done
/Users/surajprasad/miniconda3/bin/python spawn_driver.py collect >> results/watch.log 2>&1
modal volume get --force rfm-results 10_operand_gate/gate.json results/gate.json >> results/watch.log 2>&1
modal volume get --force rfm-results 10_operand_gate/dirs.pt results/dirs.pt >> results/watch.log 2>&1
echo WATCH_DONE >> results/watch.log
