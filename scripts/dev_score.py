"""Score development runs and decide by the pre-registered rule.

    python scripts/dev_score.py results/dev/p0n/manifest.json results/dev/p0c/manifest.json \
        --baseline base --device cuda

For every finished ARM run of the given manifests: L2 on its validation conditions,
computed by paper_table.compute_l2 - the reported metric's own code, over the
scanpy-HVG genes of the validation set plus control, soft gate, 1,024 transported
cells. Encoder runs are not scored. Manifests can be combined, so a later
experiment can pair its arms with an earlier experiment's baseline runs when both
used the same encoders.

Noise: the baseline arm's seed-to-seed SD within each validation fold, pooled.

Comparison: every other arm is paired with the baseline run of the same dataset,
validation fold and seed; delta = L2(arm) - L2(baseline). Per dataset, the mean
delta and its standard error over pairs.

Decision rule, fixed in the phase-0 design before any arm was run:
  adopt an arm if, on at least one dataset, mean delta <= -2 SE,
  and on no dataset mean delta > +1 SE.
Ties between adoptable arms go to the simpler, closer-to-default arm, by hand.

Required seeds: the pairs needed for 2 SE to drop below --mde, from the observed
SD of the paired deltas - or sqrt(2) x the baseline noise while no pairs exist -
converted to seeds per validation fold.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def find_run(tag: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(ROOT, "results", "runs", f"{tag}_*", "checkpoint.pt")))
    if len(hits) > 1:
        raise SystemExit(f"tag {tag} matches {len(hits)} finished runs: {hits}")
    return os.path.dirname(hits[0]) if hits else None


def pooled_sd(groups: list[list[float]]) -> tuple[float, int]:
    """Pooled within-group SD and its degrees of freedom."""
    num, dof = 0.0, 0
    for values in groups:
        if len(values) >= 2:
            num += float(np.var(values, ddof=1)) * (len(values) - 1)
            dof += len(values) - 1
    return (math.sqrt(num / dof) if dof else float("nan")), dof


def summarise(deltas: list[float]) -> dict:
    n = len(deltas)
    mean = float(np.mean(deltas)) if n else float("nan")
    sd = float(np.std(deltas, ddof=1)) if n >= 2 else float("nan")
    se = sd / math.sqrt(n) if n >= 2 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "se": se}


def decide(per_dataset: dict[str, dict]) -> str:
    stats = [s for s in per_dataset.values() if s["n"] >= 2 and math.isfinite(s["se"])]
    if len(stats) < len(per_dataset) or not stats:
        return "insufficient pairs"
    improved = any(s["mean"] <= -2 * s["se"] for s in stats)
    hurt = any(s["mean"] > s["se"] for s in stats)
    if improved and not hurt:
        return "ADOPT"
    return "reject (hurts a dataset)" if hurt else "reject (no 2 SE improvement)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--baseline", required=True, help="arm the others are paired with")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-cells", type=int, default=1024)
    parser.add_argument("--mde", type=float, default=0.05,
                        help="smallest L2 improvement the experiment must resolve")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    from paper_table import compute_l2
    from src.eval import diagnostics

    jobs = []
    for path in args.manifests:
        with open(path, encoding="utf-8") as handle:
            jobs += [job for job in json.load(handle)["jobs"] if job["kind"] == "arm"]

    records, missing, current = [], [], None
    print(f"{'run tag':34s} {'dataset':13s} {'group':6s} {'arm':10s} {'seed':>4s} {'L2':>8s}")
    for job in sorted(jobs, key=lambda j: (j["dataset"], j["group"], j["arm"], j["seed"])):
        run_dir = find_run(job["tag"])
        if run_dir is None:
            missing.append(job["tag"])
            continue
        group = (job["dataset"], job["group"])
        if group != current:
            # Validation folds differ in cache AND split; drop both caches so a
            # later fold can never be scored with an earlier fold's split or cells.
            diagnostics._DATASETS.clear()
            diagnostics._FOLDS.clear()
            current = group
        l2 = compute_l2(run_dir, args.device, args.n_cells, "soft", 1000, "double")
        records.append({**{k: job[k] for k in ("tag", "dataset", "group", "arm", "seed")},
                        "run": os.path.basename(run_dir), "l2": l2})
        print(f"{job['tag'][:33]:34s} {job['dataset']:13s} {job['group']:6s} "
              f"{job['arm']:10s} {job['seed']:4d} {l2:8.4f}")
    if missing:
        print(f"\nnot finished yet ({len(missing)}): {', '.join(missing)}")

    by_key = {(r["dataset"], r["group"], r["arm"], r["seed"]): r["l2"] for r in records}
    datasets = sorted({r["dataset"] for r in records})
    arms = sorted({r["arm"] for r in records if r["arm"] != args.baseline})

    print(f"\n=== noise: baseline '{args.baseline}' across seeds ===")
    noise = {}
    for dataset in datasets:
        groups = sorted({r["group"] for r in records if r["dataset"] == dataset})
        per_group = [[by_key[k] for k in sorted(by_key)
                      if k[0] == dataset and k[1] == g and k[2] == args.baseline] for g in groups]
        sd, dof = pooled_sd(per_group)
        noise[dataset] = (sd, len(groups))
        means = [f"{g} {np.mean(v):.4f} (n={len(v)})" for g, v in zip(groups, per_group) if v]
        print(f"  {dataset:13s} pooled SD {sd:.4f} (dof {dof})   " + "   ".join(means))

    print("\n=== paired deltas vs baseline (negative = better) ===")
    decisions = {}
    sd_deltas = {dataset: [] for dataset in datasets}
    for arm in arms:
        per_dataset = {}
        for dataset in datasets:
            deltas = [by_key[k] - by_key[(k[0], k[1], args.baseline, k[3])]
                      for k in sorted(by_key)
                      if k[0] == dataset and k[2] == arm
                      and (k[0], k[1], args.baseline, k[3]) in by_key]
            per_dataset[dataset] = summarise(deltas)
            if math.isfinite(per_dataset[dataset]["sd"]):
                sd_deltas[dataset].append(per_dataset[dataset]["sd"])
        decisions[arm] = decide(per_dataset)
        cells = "   ".join(f"{d}: {s['mean']:+.4f} +- {s['se']:.4f} (n={s['n']})"
                           for d, s in per_dataset.items())
        print(f"  {arm:10s} {cells}   -> {decisions[arm]}")

    print(f"\n=== seeds needed per validation fold to resolve {args.mde} L2 (2 SE) ===")
    for dataset in datasets:
        sigma, n_groups = noise[dataset]
        if sd_deltas[dataset]:
            sd, source = max(sd_deltas[dataset]), "largest paired-delta SD"
        else:
            sd, source = math.sqrt(2) * sigma, "sqrt(2) x baseline SD"
        if not math.isfinite(sd):
            print(f"  {dataset:13s} unknown - need at least two baseline seeds per fold")
            continue
        pairs = math.ceil((2 * sd / args.mde) ** 2)
        print(f"  {dataset:13s} SD {sd:.4f} ({source}) -> {pairs} pairs "
              f"= {math.ceil(pairs / max(n_groups, 1))} seeds x {n_groups} fold(s)")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["tag", "run", "dataset", "group",
                                                        "arm", "seed", "l2"])
            writer.writeheader()
            writer.writerows(records)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
