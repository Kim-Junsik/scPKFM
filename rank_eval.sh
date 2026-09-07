#!/bin/sh
# Score the rank sweep under both hurdle gates and print the two tables.
#
#   sh rank_eval.sh
#
# Nothing conditional: every command below is one you would type by hand, in
# the order you would type it. Comment out a line whose result you already have
# - run_celleval.py re-exports and re-scores rather than skipping, and each
# pass is about half an hour.
#
# --infer-top-gene 1000 is on every line because run.sh passes it, so the
# scores already on disk are over those 1,000 genes. Scoring anything here on
# all 5,032 would put two rows of one table on different gene sets.

set -x

# ---------------------------------------------------------------- sample
python scripts/run_celleval.py results/runs/pathway_affine_learned_f1 --profile full --infer-top-gene 1000
python scripts/run_celleval.py results/runs/p_r32_affine_learned_f1   --profile full --infer-top-gene 1000
python scripts/run_celleval.py results/runs/p_r64_affine_learned_f1   --profile full --infer-top-gene 1000
python scripts/run_celleval.py results/runs/p_r96_affine_learned_f1   --profile full --infer-top-gene 1000
python scripts/run_celleval.py results/runs/p_r128_affine_learned_f1  --profile full --infer-top-gene 1000
python scripts/run_celleval.py results/runs/p_rfull_affine_learned_f1 --profile full --infer-top-gene 1000

# ------------------------------------------------------------------ soft
python scripts/run_celleval.py results/runs/pathway_affine_learned_f1 --gate soft --profile full --infer-top-gene 1000
python scripts/run_celleval.py results/runs/p_r32_affine_learned_f1   --gate soft --profile full --infer-top-gene 1000
python scripts/run_celleval.py results/runs/p_r64_affine_learned_f1   --gate soft --profile full --infer-top-gene 1000
python scripts/run_celleval.py results/runs/p_r96_affine_learned_f1   --gate soft --profile full --infer-top-gene 1000
python scripts/run_celleval.py results/runs/p_r128_affine_learned_f1  --gate soft --profile full --infer-top-gene 1000
python scripts/run_celleval.py results/runs/p_rfull_affine_learned_f1 --gate soft --profile full --infer-top-gene 1000

# ---------------------------------------------------------------- tables
python scripts/paper_table.py \
  results/runs/pathway_affine_learned_f1 \
  results/runs/p_r32_affine_learned_f1 \
  results/runs/p_r64_affine_learned_f1 \
  results/runs/p_r96_affine_learned_f1 \
  results/runs/p_r128_affine_learned_f1 \
  results/runs/p_rfull_affine_learned_f1 \
  --n-cells 1024 --infer-top-gene 1000 --device cuda --csv results/rank_sample.csv

python scripts/paper_table.py \
  results/runs/pathway_affine_learned_f1 \
  results/runs/p_r32_affine_learned_f1 \
  results/runs/p_r64_affine_learned_f1 \
  results/runs/p_r96_affine_learned_f1 \
  results/runs/p_r128_affine_learned_f1 \
  results/runs/p_rfull_affine_learned_f1 \
  --gate soft --n-cells 1024 --infer-top-gene 1000 --device cuda --csv results/rank_soft.csv
