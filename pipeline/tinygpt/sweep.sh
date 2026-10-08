#!/usr/bin/env bash
# Architecture sweep: same data, same steps (5000 x 512 x 64 tokens = 164M tokens), lr 3e-3.
# Runs sequentially (parallel processes on one GPU were ~10x slower on Windows).
# Usage: bash sweep.sh "64 4" "72 4" ...
PY="${PYTHON:-python}"   # your Python (3.10+ with torch); set PYTHON to use another one
cd "$(dirname "$0")"
for cfg in "$@"; do
  set -- $cfg
  "$PY" train.py --name sweep_d$1L$2 --d $1 --layers $2 --batch 512 --steps 5000 --warmup 200 --lr 3e-3 --eval-every 1000 --max-minutes 8 > logs/sweep_d$1L$2.log 2>&1
done
