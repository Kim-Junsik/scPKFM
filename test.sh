#!/bin/sh
# Evaluate a trained run against the target line from data_prepare.py.
#
#   sh test.sh                        the most recent run
#   sh test.sh results/runs/<tag>     a specific run
#   sh test.sh --summary              every run in one table
#   sh test.sh --summary --diagnose   that table plus the gate and transport reads
#   sh test.sh --summary --celleval   join the cell-eval scores already written
#   sh test.sh <run> --celleval       also score with cell-eval 0.5.42
#
# The internal metrics are already written by training; this reads them back and
# puts them next to the baselines. --celleval additionally transports control
# cells and hands the result to the separate Python 3.11 environment.
#
# Note what --celleval means in each position: with --summary it only JOINS scores
# that already exist, because scoring six runs takes about ninety minutes and is
# not something to trigger by accident. To actually score them:
#
#   for d in results/runs/s2_*/; do python scripts/run_celleval.py "$d" --profile full; done
set -e

RUN=""
SUMMARY=0
CELLEVAL=0
DIAGNOSE=0
FILTER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --summary)  SUMMARY=1; shift ;;
    --celleval) CELLEVAL=1; shift ;;
    --diagnose) DIAGNOSE=1; shift ;;
    --filter)   FILTER="$2"; shift 2 ;;
    *)          RUN="$1"; shift ;;
  esac
done

if [ "$SUMMARY" -eq 1 ]; then
  # With --summary, --celleval joins the already-written agg_results.csv files
  # instead of scoring anything. Written out rather than ${CELLEVAL:+...} because
  # that expands on the string "0" as well as "1".
  EXTRA=""
  if [ "$CELLEVAL" -eq 1 ]; then EXTRA="--celleval"; fi
  python scripts/summarise_runs.py ${FILTER:+--filter "$FILTER"} $EXTRA

  if [ "$DIAGNOSE" -eq 1 ]; then
    # The diagnostics take run directories rather than a --filter, so the filter
    # is resolved here. With no filter they fall back to their own default (s2_*)
    # instead of every smoke run ever left in results/runs.
    RUNS=""
    if [ -n "$FILTER" ]; then
      RUNS=$(ls -d results/runs/*/ 2>/dev/null | grep -- "$FILTER" | tr '\n' ' ') || true
    fi
    # Gate first: if the interaction term is numerically absent, the table above
    # is comparing a model against itself and nothing else here means anything.
    echo ""
    python scripts/diagnose_gate.py $RUNS
    echo ""
    python scripts/diagnose_transport.py $RUNS
  fi
  exit 0
fi

if [ -z "$RUN" ]; then
  RUN=$(ls -dt results/runs/*/ 2>/dev/null | head -1)
  if [ -z "$RUN" ]; then
    echo "no run found under results/runs - train one first:  sh train.sh"
    exit 1
  fi
  echo "using the most recent run: $RUN"
fi

python scripts/summarise_runs.py --run "$RUN"

if [ "$CELLEVAL" -eq 1 ]; then
  echo ""
  python scripts/run_celleval.py "$RUN" --profile minimal
fi
