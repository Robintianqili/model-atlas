#!/usr/bin/env bash
# Final TinyGPT run (d76 L4, lr 6e-3). Each call trains for at most 8 minutes and saves state;
# call it again until the log says "done".
PY="${PYTHON:-python}"   # your Python (3.10+ with torch); set PYTHON to use another one
cd "$(dirname "$0")"
"$PY" train.py --name final_d76L4 --d 76 --layers 4 --heads 4 --batch 512 --steps 30000 --warmup 300 --lr 6e-3 \
  --eval-every 500 --replay-steps 0,10,30,100,200,400,1000,3000,10000,30000 --max-minutes 8 >> logs/final_d76L4.log 2>&1
tail -3 logs/final_d76L4.log
