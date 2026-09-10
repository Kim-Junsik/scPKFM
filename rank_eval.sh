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
# whole real set before computing DE. A Norman fold is ~16,000 cells over 1,000
# genes in float32, about 62 MiB, against a container's default 64 MB /dev/shm -
# so two at once, or one fold slightly larger, dies with SIGBUS (exit -7) and
# names neither pdex nor /dev/shm. Start the container with --shm-size=8g (see
# the Dockerfile) and keep these sequential anyway; the GPU is the bottleneck.
#
# --infer-top-gene 1000 is on every line because run.sh passes it, so the scores
# already on disk are over those genes. --gate soft on both the scoring and the
# table, or L2 and the five cell-eval columns come from different point estimates.

set -x

python scripts/run_celleval.py results/runs/fin_affine_learned_f0 --gate soft --profile full --infer-top-gene 1000 --threads 16
python scripts/run_celleval.py results/runs/fin_affine_learned_f1 --gate soft --profile full --infer-top-gene 1000 --threads 16
python scripts/run_celleval.py results/runs/fin_affine_learned_f2 --gate soft --profile full --infer-top-gene 1000 --threads 16
python scripts/run_celleval.py results/runs/fin_affine_learned_f3 --gate soft --profile full --infer-top-gene 1000 --threads 16
python scripts/run_celleval.py results/runs/fin_affine_learned_f4 --gate soft --profile full --infer-top-gene 1000 --threads 16

python scripts/paper_table.py \
  results/runs/fin_affine_learned_f0 \
  results/runs/fin_affine_learned_f1 \
  results/runs/fin_affine_learned_f2 \
  results/runs/fin_affine_learned_f3 \
  results/runs/fin_affine_learned_f4 \
  --gate soft --mean --n-cells 1024 --infer-top-gene 1000 --device cuda \
  --csv results/fin_table_soft.csv
