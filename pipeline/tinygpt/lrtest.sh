#!/usr/bin/env bash
# Learning-rate check for the chosen size (d76L4), same 5000-step budget as the sweep.
PY="${PYTHON:-python}"   # your Python (3.10+ with torch); set PYTHON to use another one
cd "$(dirname "$0")"
for lr in "$@"; do
  "$PY" train.py --name lr_d76L4_$lr --d 76 --layers 4 --batch 512 --steps 5000 --warmup 200 --lr $lr --eval-every 1000 --max-minutes 8 > logs/lr_d76L4_$lr.log 2>&1
done
