"""Does the field transport the cells far enough, and in the right direction?

    python scripts/diagnose_transport.py                    every s2_* run
    python scripts/diagnose_transport.py results/runs/diag  one run
    python scripts/diagnose_transport.py --device cuda      once the sweep is done

resid_R2 near 0 has two causes that the summary table cannot tell apart: the model
reproduced additivity, or the model did nothing. The displacement ratio separates
them. A ratio near 1 with a high cosine means the field lands where it should; a
ratio well under 1 means it stops short, and every downstream metric is then
measuring a model that never left the control population.

Reads checkpoint.pt only, on cpu by default, so it does not disturb a running sweep.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.diagnostics import (condition_groups, load_run, measure_transport,
                                  subsample)

HEADER = (f"{'run':32s} {'group':14s} {'n':>4s} {'lat_ratio':>10s} {'lat_cos':>8s} "
          f"{'gene_ratio':>11s} {'gene_cos':>9s}")


def summarise(rows: list[dict]) -> dict:
    """Median, not mean: a single condition whose true displacement is nearly
    zero sends the ratio to a huge value and would carry the average with it."""
    return {key: float(np.nanmedian([r[key] for r in rows]))
            for key in ("latent_ratio", "latent_cos", "gene_ratio", "gene_cos")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=None, help="run directories")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cells", type=int, default=256,
                        help="control cells transported per condition")
    parser.add_argument("--max-per-group", type=int, default=40,
                        help="cap per group; 0 for all")
    args = parser.parse_args()

    runs = args.runs or sorted(glob.glob("results/runs/s2_*"))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "checkpoint.pt"))]
    if not runs:
        print("no finished run found (checkpoint.pt missing). still training?")
        return

    print(HEADER)
    print("-" * len(HEADER))
    for run_dir in runs:
        config, data, stats, fold, vae, field = load_run(run_dir, args.device)
        rng = np.random.default_rng(config["eval"]["seed"])
        groups = condition_groups(data, stats, fold, config["split"]["method"])

        name = os.path.basename(run_dir.rstrip("/\\"))
        for group, conditions in groups.items():
            conditions = subsample(conditions, args.max_per_group, rng)
            if not conditions:
                continue
            rows = measure_transport(vae, field, data, stats, conditions, config,
                                     rng, args.device, args.n_cells)
            if not rows:
                continue
            s = summarise(rows)
            print(f"{name[:31]:32s} {group:14s} {len(rows):4d} "
                  f"{s['latent_ratio']:10.3f} {s['latent_cos']:8.3f} "
                  f"{s['gene_ratio']:11.3f} {s['gene_cos']:9.3f}")
            name = ""  # only label the first row of each run

    print("\nratio = ||predicted mean displacement|| / ||true mean displacement||,"
          "\nmedian over conditions. 1.0 is correct length; below 1 is under-"
          "\ntransport, above 1 overshoot. cos is direction, and separates 'too"
          "\nshort' from 'wrong way'. Latent is where the field lives; a large"
          "\ngap between the latent and gene rows puts the loss in the decoder.")


if __name__ == "__main__":
    main()
