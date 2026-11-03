#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 tune.py          > results/tune_log.txt 2>&1
python3 run_grid.py 1 2  > results/grid_log.txt 2>&1
echo ALL_DONE >> results/grid_log.txt
