#!/bin/sh
# One configuration, end to end: build the cache if needed, train, evaluate.
#
#   sh run.sh                 build (if needed) + train + evaluate
#   sh run.sh --eval-only     skip training, re-run the evaluation
#   sh run.sh --no-celleval   skip the cell-eval scoring (~30 min) for a quick look
#
# scPKFM - single-cell Pathway Koopman Flow Matching. There is no interaction
# axis, so this trains ONE arm rather than a pair. See README.md for the method.
set -e

# --------------------------------------------------------------------- EDIT
TAG=koopman_endpoint3              # run directory is results/runs/${TAG}_f${FOLD}: the fold
                         # is appended automatically so a run can never be filed
                         # under the wrong one. Still change TAG for every
                         # experiment - scripts/train.py opens the directory with
                         # exist_ok=True and overwrites checkpoint.pt without
                         # asking. The guard below refuses when that would happen.

FOLD=1                   # 0-4. NOT interchangeable: a fixed ridge baseline moves
                         # 0.47 in resid_R2 between folds (0.5329 on fold 1, 0.0628
                         # on fold 3), which is larger than any model difference
                         # measured here. Never compare a score against a baseline
                         # from a different fold, and say which fold in the paper.

N_HVG=5000               # matches the reference pipeline's --n_top_genes
HVG_CRITERION=scanpy     # sc.pp.highly_variable_genes with its seurat default.
                         # A plain variance-to-mean ratio picks a different set and
                         # moves Control-vs-double L2 by 1.6x.
STRICT_SPLIT=true        # exclude held-out conditions from the cells that CHOOSE
                         # the gene space, on top of excluding them from stage 1
                         # and the latent standardisation. Published pipelines do
                         # not do this, so it makes the problem strictly harder -
                         # state it as such rather than hiding it.

BATCH=256
LR=1e-3
STAGE1=30                # autoencoding epochs
STAGE2=200               # flow-matching epochs
WARMUP=60                # singles-only epochs before combinations join

ENDPOINT_WEIGHT=3        # ||Phi_a(z_ctrl) - z_a||^2 over training conditions.
RESID_WEIGHT=5           # ||(Phi_ab - Phi_a - Phi_b + z0) - r_true||^2.
MMD_WEIGHT=0             # distribution-level term: MMD between the one-step
                         # endpoint estimate and the coupled targets, over the
                         # whole minibatch. Every other term here constrains a
                         # MEAN, so nothing asks the predicted population to have
                         # the right SHAPE - which is what DS and edist_rel score.
                         # scDFM trains with this (gamma=0.5); this model did not,
                         # so a table claiming a like-for-like comparison should
                         # either turn it on or say it is off.
                         # The weight does NOT transfer from theirs: their kernel
                         # runs on genes, this one on the 64-d latent. Measured
                         # here, the raw term sits near 0.02 against an fm of 1.3,
                         # so their gamma=0.5 would contribute 0.5 % of the loss -
                         # numerically off. 10 gives it roughly 20 %. Read the
                         # `mmd` column against `fm` in the log and adjust.

RESID_STEPS=2            # RK4 steps for those two terms only; inference still uses
                         # train.n_integration_steps.
                         # Use both or neither: constraining one lets the error
                         # move into the other, which is how the measured
                         # cancellation arose (singles 35 % short, composition
                         # non-additivity 8x too large, cancelling on doubles).
                         # Endpoint gets the larger weight because 88 % of the
                         # signal this task is scored on is additive.

INIT_VAE_FROM=
                         # empty = train stage 1. Otherwise a finished run whose
                         # encoder to reuse, e.g. results/runs/koopman_affine_f1.
                         # Legitimate because stage 2 freezes the encoder, so the
                         # same weights would be produced again; reusing them makes
                         # a field experiment differ from its reference in the
                         # field alone, and skips ~a third of the wall clock.
                         # NOT auto-detected on purpose: an encoder from another
                         # fold or gene space can load without error and be
                         # silently wrong. train.py compares the settings that
                         # define the encoder and refuses on a mismatch.

COMPOSITION=learned     # additive | learned
                         # additive  v = Σ u_a. First-order BCH; the default and
                         #           the only form measured so far.
                         # learned   v = Σ u_a + ρ(Σ φ(u_a)). Learns the
                         #           composition law from the generators' output
                         #           VELOCITIES instead of truncating BCH at a
                         #           fixed order. Both interaction terms that
                         #           already failed (Lie bracket, free MLP) read
                         #           perturbation IDENTITY; this reads velocities,
                         #           so an unseen pair stands on the same footing
                         #           as a seen one. Starts exactly additive.
                         # UNTESTED. Three structured terms have been added to
                         # this model and none improved it.

GENERATOR=affine         # affine | neural_field
                         # affine        u_a = s(t)·(A_a z + b_a). One operator PER
                         #               perturbation, 4,160 parameters each. This
                         #               is the Koopman form and the default.
                         # neural_field  u_a = f([z, φ(t), e_a]). One SHARED trunk;
                         #               the perturbation enters only through a
                         #               32-dim embedding, so 32 parameters each -
                         #               130x less dedicated capacity.
                         # Measured on fold 0, everything else fixed:
                         #   mlp   affine 0.3348  neural_field -0.4377  (+0.77)
                         #   pcab  affine 0.1965  neural_field -0.1346  (+0.33)
                         # Never compared on fold 1 or under the population losses,
                         # which is the point of keeping the switch.

LATENT_READOUT=dense     # dense | pathway
                         # dense   z = Linear(K*d_v -> 64). The measured setting;
                         #         the projection mixes every token, so no latent
                         #         dimension corresponds to a pathway.
                         # pathway z_k = <h_k, w> + b_k, so latent_dim BECOMES the
                         #         token count and dimension k IS pathway k. A_a is
                         #         then a [K, K] pathway interaction matrix that can
                         #         be read directly: A_a[i,j] is how much pathway j
                         #         drives pathway i under perturbation a.
                         #         This buys INTERPRETABILITY, not accuracy. The
                         #         64-d bottleneck was measured innocent (residual
                         #         share 0.575 in latent vs 0.393 in gene space), so
                         #         widening it is not expected to help and may cost.
                         #         K also counts the free tokens, which carry no
                         #         prior and are therefore not pathways - report the
                         #         annotated and free blocks separately.
GENERATOR_RANK=null      # null | integer. A_a = U_a V_a instead of a full matrix.
                         # At LATENT_READOUT=pathway the full operator is K^2 per
                         # perturbation (~74,000 at K=272, 7.5 M over 101 of them),
                         # so rank 16 brings it back to roughly the dense cost.

N_GEN=1024               # control cells transported per condition at eval time.
INFER_TOP_GENE=1000      # gene subset the reported table is scored on.
CELLEVAL=1               # score with cell-eval after training. On by default
                         # because five of the reported table's eight columns come
                         # from it, and because skipping it here means paying the
                         # transport pass twice - once now for paper_table.py and
                         # again later for cell-eval. Costs ~30 min per run.
                         # --no-celleval turns it off for a quick look.
DEVICE=cuda
# ----------------------------------------------------------------- END EDIT

# The fold is part of the run name, not something to remember to type. A score
# is meaningless without knowing which fold it came from: a fixed ridge baseline
# moves 0.47 in resid_R2 across the five, which is larger than any model
# difference measured on this task.
# The generator is in the run name too: affine and neural_field are the comparison
# this switch exists to settle, and filing both under one name would be the same
# mistake as mixing folds.
RUN=${TAG}_${GENERATOR}_${COMPOSITION}_f${FOLD}
CACHE=assets/norman_scanpy${N_HVG}_fold${FOLD}.h5ad

EVAL_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --eval-only) EVAL_ONLY=1; shift ;;
    --celleval)    CELLEVAL=1; shift ;;
    --no-celleval) CELLEVAL=0; shift ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

echo "=== configuration ==="
echo "  run=$RUN fold=$FOLD n_hvg=$N_HVG criterion=$HVG_CRITERION"
echo "  batch=$BATCH lr=$LR stage1=$STAGE1 stage2=$STAGE2 warmup=$WARMUP"
echo "  endpoint=$ENDPOINT_WEIGHT resid=$RESID_WEIGHT steps=$RESID_STEPS"
echo "  mmd=$MMD_WEIGHT"
echo "  generator=$GENERATOR composition=$COMPOSITION"
echo "  init_vae_from=${INIT_VAE_FROM:-(none - stage 1 will train)}"
echo "  readout=$LATENT_READOUT rank=$GENERATOR_RANK"
echo "  strict_split=$STRICT_SPLIT n_gen=$N_GEN device=$DEVICE"
echo ""

if [ "$EVAL_ONLY" -eq 0 ]; then
  if [ -f "results/runs/$RUN/checkpoint.pt" ]; then
    echo "" >&2
    echo "refusing to overwrite a finished run: results/runs/$RUN" >&2
    echo "  change TAG in run.sh, or move the old run out of the way:" >&2
    echo "    mv results/runs/$RUN results/runs/${RUN}_old" >&2
    echo "" >&2
    exit 1
  fi

  if [ ! -f "$CACHE" ]; then
    echo "=== building $CACHE ==="
    python data_prepare.py --set data.n_hvg=$N_HVG \
      data.hvg_criterion=$HVG_CRITERION data.cache_h5ad=$CACHE \
      data.exclude_test_from_hvg=$STRICT_SPLIT split.fold=$FOLD
    echo ""
  else
    echo "using existing cache $CACHE"
    echo ""
  fi

  echo "=== training $RUN ==="
  python scripts/train.py --tag "$RUN" --set \
    data.cache_h5ad=$CACHE data.n_hvg=$N_HVG data.hvg_criterion=$HVG_CRITERION \
    split.fold=$FOLD \
    train.batch_size=$BATCH train.lr=$LR \
    train.stage1_epochs=$STAGE1 train.stage2_epochs=$STAGE2 \
    train.single_warmup_epochs=$WARMUP train.device=$DEVICE \
    train.endpoint_weight=$ENDPOINT_WEIGHT train.mmd_weight=$MMD_WEIGHT \
    train.resid_weight=$RESID_WEIGHT train.resid_steps=$RESID_STEPS \
    model.generator=$GENERATOR model.composition=$COMPOSITION \
    ${INIT_VAE_FROM:+train.init_vae_from=$INIT_VAE_FROM} \
    model.latent_readout=$LATENT_READOUT model.generator_rank=$GENERATOR_RANK \
    eval.n_gen_cells=$N_GEN
  echo ""
fi

echo "=== metrics vs the baselines ==="
python scripts/summarise_runs.py --filter "$RUN"

echo ""
echo "=== transport diagnosis ==="
python scripts/diagnose_transport.py "results/runs/$RUN"

if [ "$CELLEVAL" -eq 1 ]; then
  echo ""
  echo "=== cell-eval ==="
  # cell-eval needs its own interpreter: its dependency set conflicts with this
  # project's, which is why the split exists at all. Build it on first use rather
  # than letting the user find out five table columns are missing after a run has
  # already finished.
  if [ ! -x ".env-celleval-linux/bin/python" ] \
     && [ ! -x ".env-celleval/bin/python" ] \
     && [ ! -f ".env-celleval/python.exe" ]; then
    echo "no cell-eval interpreter found - building one"
    sh scripts/setup_celleval_linux.sh || true
    echo ""
  fi

  # Non-fatal. `set -e` would otherwise end the script here, after the training
  # has already been paid for, and take the reported table down with it.
  python scripts/run_celleval.py "results/runs/$RUN" --profile full \
    --infer-top-gene $INFER_TOP_GENE \
    || echo "[warn] cell-eval did not run - the table's five cell-eval columns" \
            "stay '-'. Fix the interpreter, then: python scripts/run_celleval.py" \
            "results/runs/$RUN --score-only"
fi

echo ""
echo "=== reported table ==="
python scripts/paper_table.py --filter "$RUN" --n-cells $N_GEN \
  --infer-top-gene $INFER_TOP_GENE --csv "results/${RUN}_table.csv"

echo ""
echo "checks worth reading before believing any of the above:"
echo "  leakage - this must show fewer cells than the cache holds:"
echo "    grep 'cells visible to training' results/runs/$RUN/train.log"
echo "  the two extra loss terms must be ON, and end/resid must FALL:"
echo "    grep -E 'endpoint matching|composition residual' results/runs/$RUN/train.log"
echo "    grep 'stage2 epoch' results/runs/$RUN/train.log | tail -30"
echo "  the reported table's five cell-eval columns need this file:"
echo "    ls results/runs/$RUN/celleval/agg_results.csv"
echo "  stage-1 calibration - chi2 should sit near 1.0:"
echo "    grep 'stage1 epoch' results/runs/$RUN/train.log | tail -3"
