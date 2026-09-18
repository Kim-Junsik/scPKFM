"""Does the learned composition help or hurt on combinations it never saw?

    python scripts/rho_off_probe.py results/dev/s1c/manifest.json results/dev/p0c/manifest.json \
        results/dev/c1n/manifest.json --arms base ot --device cuda

Every finished arm run of the manifests is scored twice on its validation
conditions - as trained, and with rho switched off (v = sum_a u_a) - on the same
cells, and the paired difference off - on is summarised per family and block.
Nothing is retrained, and only validation conditions are read.

Negative means the combinations are predicted better WITHOUT rho: the correction
learned on training combinations does not transfer. The single block does not
move by construction (rho never acts on a single), so it is reported as a check.

The measurement behind it (2026-09-18, p0c cv0 seed 0): rho helped
Panobinostat+SRT2104 (-0.12) and hurt SRT3025+Cediranib (+0.40) and
Givinostat+Dasatinib (+0.31). One run; this script asks the same of all of them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dev_score import BLOCKS, family, find_run, score_run, summarise, weighted_l2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--arms", nargs="+", default=None, help="only these arms")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-cells", type=int, default=1024)
    args = parser.parse_args()

    from src.eval import diagnostics

    jobs = []
    for path in args.manifests:
        with open(path, encoding="utf-8") as handle:
            jobs += [job for job in json.load(handle)["jobs"] if job["kind"] == "arm"]
    if args.arms:
        jobs = [job for job in jobs if job["arm"] in set(args.arms)]

    rows, current = [], None
    print(f"{'run tag':34s} {'family':17s} {'L2 on':>8s} {'L2 off':>8s} "
          f"{'dbl on':>8s} {'dbl off':>8s}")
    for job in sorted(jobs, key=lambda j: (j["dataset"], j["group"], j["arm"], j["seed"])):
        run_dir = find_run(job["tag"])
        if run_dir is None:
            continue
        group = (job["dataset"], job["group"])
        if group != current:
            diagnostics._DATASETS.clear()
            diagnostics._FOLDS.clear()
            current = group
        on = score_run(run_dir, args.device, args.n_cells)
        off = score_run(run_dir, args.device, args.n_cells, rho_off=True)
        row = {"tag": job["tag"], "family": family(job["dataset"], job["group"]),
               "arm": job["arm"], "on": on, "off": off}
        rows.append(row)
        print(f"{job['tag'][:33]:34s} {row['family']:17s} {weighted_l2(on):8.4f} "
              f"{weighted_l2(off):8.4f} {on['double']:8.4f} {off['double']:8.4f}")

    print("\n=== rho off - rho on, paired per run (negative = better without rho) ===")
    for fam in sorted({r["family"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows if r["family"] == fam}):
            chosen = [r for r in rows if r["family"] == fam and r["arm"] == arm]
            overall = summarise([weighted_l2(r["off"]) - weighted_l2(r["on"]) for r in chosen])
            print(f"  {fam:17s} arm {arm:8s} L2 {overall['mean']:+.4f} +- "
                  f"{overall['se']:.4f} (n={overall['n']})")
            for block in BLOCKS:
                values = [r["off"][block] - r["on"][block] for r in chosen
                          if r["on"][block] is not None]
                if values:
                    s = summarise(values)
                    se = f"{s['se']:.4f}" if math.isfinite(s["se"]) else "nan"
                    print(f"      {block:7s} {s['mean']:+.4f} +- {se} (n={s['n']})")


if __name__ == "__main__":
    main()
