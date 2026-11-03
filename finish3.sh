#!/usr/bin/env bash
# Runs after the grid: each step alone, in order.
set -e
cd "$(dirname "$0")"
python3 comm_bench.py --b2b >> results/comm_log.txt 2>&1
python3 validate.py > results/validate_log.txt 2>&1
echo FINISH3_DONE >> results/validate_log.txt
