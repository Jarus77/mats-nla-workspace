#!/bin/bash
set -u
cd "$(dirname "$0")"
until python spawn_driver.py status | grep -q "^prep done\|^prep ERR"; do sleep 60; done
python spawn_driver.py status | head -1 > results/watch.log
grep -q "prep done" results/watch.log || { echo "PREP FAILED" >> results/watch.log; exit 1; }
python spawn_driver.py run >> results/watch.log
while python spawn_driver.py status | grep -q "running"; do sleep 90; done
python spawn_driver.py status >> results/watch.log
python spawn_driver.py collect >> results/watch.log
modal volume get --force rfm-results 09_ol_mechanism results >> results/watch.log 2>&1
echo WATCH_DONE >> results/watch.log
