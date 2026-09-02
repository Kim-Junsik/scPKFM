"""Read back what training already measured and put it next to the baselines.

Nothing is recomputed here - training writes results.json, data_prepare.py writes
results/baselines_*.json, and this only joins them so a run can be read against
the line it had to beat.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# `identity` is edist_rel with transport switched off - what doing NOTHING but
# encode-decode scores on the same scale. It is a reference point, not a bound:
# a working model comes in below it.
HEADER = (f"{'run':34s} {'backbone':11s} {'latent':>7s} {'coupling':11s} "
          f"{'resid_R2':>9s} {'edist':>8s} {'identity':>9s}")

# cell-eval writes one agg_results.csv per run and nothing joins them, so a sweep
# leaves six files and no table. These are the columns worth putting side by side;
# anything else cell-eval produced is named at the bottom rather than dropped
# silently. Order is the display order, and a column missing from the profile that
# was run is skipped rather than shown empty.
CELLEVAL_COLUMNS = (
    ("mse", "mse"),
    ("mae", "mae"),
    ("pearson_delta", "pearson_d"),
    ("discrimination_score_l1", "PDS"),
    ("de_spearman_lfc_sig", "DE_spear"),
    ("overlap_at_N", "overlap"),
)


def load_baselines() -> dict:
    """(dataset, method) -> baseline results."""
    out = {}
    for path in glob.glob("results/baselines_*.json"):
        stem = os.path.basename(path)[len("baselines_"):-len(".json")]
        dataset, _, method = stem.rpartition("_")
        out[(dataset, method)] = json.load(open(path))["results"]
    return out


def row(name: str, payload: dict) -> str:
    model = payload["config"]["model"]
    r = payload["results"]
    return (f"{name[:33]:34s} {model.get('backbone', 'mlp'):11s} "
            f"{model.get('latent_dim', 0):7d} "
            f"{payload['config']['train'].get('coupling', '-'):11s} "
            f"{r['resid_R2_pooled']:9.4f} {r['edist_rel']:8.4f} "
            # Older result files only carry the deprecated name.
            f"{r.get('edist_rel_identity', r.get('edist_rel_autoencoder_floor', float('nan'))):9.4f}")


def load_celleval(run_dir: str) -> dict[str, str] | None:
    """The `mean` row of a run's agg_results.csv, or None if it was never scored.

    cell-eval writes the file as statistic-per-row (count, mean, std, quartiles),
    so the mean over the fold's test conditions is one row rather than a column.
    """
    path = os.path.join(run_dir, "celleval", "agg_results.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record.get("statistic") == "mean":
                return {k: v for k, v in record.items() if k != "statistic"}
    return None


def print_celleval(entries: list[tuple[str, dict | None]]) -> None:
    scored = [(name, values) for name, values in entries if values]
    if not scored:
        print("\nno cell-eval results found. score a run with:"
              "\n  python scripts/run_celleval.py results/runs/<tag> --profile full")
        return

    present = [(key, label) for key, label in CELLEVAL_COLUMNS
               if any(key in values for _, values in scored)]
    header = f"{'run':34s} " + " ".join(f"{label:>11s}" for _, label in present)
    print("\n" + header)
    print("-" * len(header))
    for name, values in scored:
        cells = []
        for key, _ in present:
            raw = values.get(key)
            try:
                cells.append(f"{float(raw):11.4f}")
            except (TypeError, ValueError):
                cells.append(f"{'-':>11s}")
        print(f"{name[:33]:34s} " + " ".join(cells))

    missing = [name for name, values in entries if not values]
    if missing:
        print(f"not scored: {', '.join(missing)}")

    shown = {key for key, _ in present}
    extra = sorted({k for _, values in scored for k in values} - shown)
    if extra:
        print(f"also in agg_results.csv: {', '.join(extra)}")


def print_target_line(baselines: dict, dataset: str, method: str) -> None:
    ridge = baselines.get((dataset, method), {}).get("ridge_additive")
    if not ridge:
        print(f"\ntarget ({dataset}, {method}): none - run data_prepare.py first")
        return
    print(f"\ntarget ({dataset}, {method}, ridge_additive):  "
          f"resid_R2 > {ridge['resid_R2_pooled']:.4f}   "
          f"edist_rel < {ridge['edist_rel']:.4f}   "
          f"(n={ridge['n_evaluated']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="a single run directory")
    parser.add_argument("--filter", default=None,
                        help="only runs whose name contains this, e.g. s2_mlp")
    parser.add_argument("--celleval", action="store_true",
                        help="also join each run's celleval/agg_results.csv")
    args = parser.parse_args()

    baselines = load_baselines()
    if not baselines:
        print("[warn] no baselines found - run: python data_prepare.py")

    paths = ([os.path.join(args.run, "results.json")] if args.run
             else sorted(glob.glob("results/runs/*/results.json")))
    paths = [p for p in paths if os.path.exists(p)]
    if args.filter:
        paths = [p for p in paths if args.filter in os.path.basename(os.path.dirname(p))]
    if not paths:
        print("no results.json found. train first:  sh train.sh")
        return

    print(HEADER)
    print("-" * len(HEADER))
    seen = set()
    entries = []
    for path in paths:
        payload = json.load(open(path))
        cache = payload["config"]["data"]["cache_h5ad"]
        seen.add((os.path.splitext(os.path.basename(cache))[0],
                  payload["config"]["split"]["method"]))
        run_dir = os.path.dirname(path)
        name = os.path.basename(run_dir)
        print(row(name, payload))
        entries.append((name, run_dir))

    for dataset, method in sorted(seen):
        print_target_line(baselines, dataset, method)

    if args.celleval:
        print_celleval([(name, load_celleval(run_dir)) for name, run_dir in entries])

    print("\nidentity is the autoencoder with transport switched off - what doing"
          "\nnothing scores. edist equal to identity means the flow did nothing, so"
          "\ncomparing interactions in that state says nothing. It is a reference"
          "\npoint and NOT a bound: a working model comes in well below it.")


if __name__ == "__main__":
    main()
