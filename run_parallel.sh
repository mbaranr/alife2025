#!/usr/bin/env bash
set -euo pipefail

TOTAL_RUNS=150
MAX_CONCURRENT=20
RUN_PY=/path/to/run.py

source /path/to/conda/activate
conda activate dgca_tasks

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs/

seq 0 $((TOTAL_RUNS-1)) | parallel -j "${MAX_CONCURRENT}" --halt soon,fail=1 \
  "python -u ${RUN_PY} --run_id {} > logs/run_{}.out 2> logs/run_{}.err"