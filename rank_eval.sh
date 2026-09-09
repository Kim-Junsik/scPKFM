#!/bin/sh
# Score the five folds with cell-eval and print the averaged table.
#
#   sh rank_eval.sh
#
# Plain commands in order. Comment out a line whose result you already have -
# run_celleval.py re-exports and re-scores rather than skipping, and each pass
# is about half an hour.
#
# One at a time on purpose: pdex builds one dense shared-memory matrix of the
# whole real set before computing DE, and two at once exhaust a container's
# 64 MB /dev/shm and die with SIGBUS (exit -7).
#
# --infer-top-gene 1000 is on every line because run.sh passes it, so the scores
# already on disk are over those genes. --gate soft on both the scoring and the
# table, or L2 and the five cell-eval columns come from different point estimates.

set -x

python scripts/run_celleval.py results/runs/main_affine_learned_f0 --gate soft --profile full --infer-top-gene 1000 --threads 16
python scripts/run_celleval.py results/runs/main_affine_learned_f1 --gate soft --profile full --infer-top-gene 1000 --threads 16
python scripts/run_celleval.py results/runs/main_affine_learned_f2 --gate soft --profile full --infer-top-gene 1000 --threads 16
python scripts/run_celleval.py results/runs/main_affine_learned_f3 --gate soft --profile full --infer-top-gene 1000 --threads 16
python scripts/run_celleval.py results/runs/main_affine_learned_f4 --gate soft --profile full --infer-top-gene 1000 --threads 16

python scripts/paper_table.py \
  results/runs/main_affine_learned_f0 \
  results/runs/main_affine_learned_f1 \
  results/runs/main_affine_learned_f2 \
  results/runs/main_affine_learned_f3 \
  results/runs/main_affine_learned_f4 \
  --gate soft --mean --n-cells 1024 --infer-top-gene 1000 --device cuda \
  --csv results/table_soft.csv
