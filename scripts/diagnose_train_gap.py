"""Is the transport error a fitting problem or a generalisation problem?

    python scripts/diagnose_train_gap.py results/runs/fin_combosciplex_scdfm7_affine_learned_f0 \
        --device cuda --gate soft --infer-top-gene 1000

For TRAINING and held-out conditions alike it reports the model's L2 (eq. 15, the
reported table's metric) next to the decoder's ceiling on the same conditions -
the L2 a perfect flow would still pay, since the real cells are encoded and
decoded with no transport at all.

  training conditions far above their ceiling  ->  the objective is not reaching
      the quantity L2 scores, even where it is supervised. The endpoint term
      matches Phi(mean z_ctrl) in latent space, while L2 scores the gene-space
      mean of every decoded transported cell; these differ whenever the field or
      the decoder is nonlinear. Matching the decoded population mean targets it.
  training conditions near their ceiling, held-out far above  ->  generalisation;
      a better-matched training objective would not help.

The ceiling and the model are scored on the same genes and the same conditions,
with the same per-condition subsample of real cells for the ceiling.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.diagnostics import (condition_groups, load_run, measure_transport,
                                  scdfm_eval_genes, subsample)
from src.eval.predict import _head_aux


@torch.no_grad()
def ceiling(vae, data, stats, conditions, genes, rng, device, n_cells) -> list[float]:
    out = []
    for condition in conditions:
        cells = data.cells(condition)
        take = rng.choice(cells.shape[0], size=min(n_cells, cells.shape[0]), replace=False)
        x = torch.as_tensor(cells[take], device=device)
        z, _ = vae.encode_z(x)
        decoded = vae.reconstruction(vae.decode_z(z), **_head_aux(vae, x))
        mean = decoded.mean(dim=0).cpu().numpy()
        truth = stats.mean[condition]
        if genes is not None:
            mean, truth = mean[genes], truth[genes]
        out.append(float(np.linalg.norm(mean - truth)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gate", default=None, choices=["soft", "hard", "sample"])
    parser.add_argument("--n-cells", type=int, default=1024)
    parser.add_argument("--infer-top-gene", type=int, default=None)
    parser.add_argument("--max-per-group", type=int, default=40,
                        help="cap per group, so Norman's 101 training singles stay quick")
    args = parser.parse_args()

    header = (f"{'run':34s} {'group':14s} {'n':>4s} {'model L2':>9s} "
              f"{'ceiling':>8s} {'transport adds':>15s}")
    print(header)
    print("-" * len(header))
    for run_dir in args.runs:
        config, data, stats, fold, vae, field = load_run(run_dir, args.device, args.gate)
        rng = np.random.default_rng(config["eval"]["seed"])
        genes = (scdfm_eval_genes(data, fold, args.infer_top_gene)
                 if args.infer_top_gene else None)
        groups = condition_groups(data, stats, fold, config["split"]["method"])
        name = os.path.basename(run_dir.rstrip("/\\"))
        for group in ("train singles", "train doubles", "test singles", "test doubles"):
            conditions = subsample(groups[group], args.max_per_group, rng)
            conditions = [c for c in conditions
                          if c != data.control_condition and stats.has(c)]
            if not conditions:
                continue
            rows = measure_transport(vae, field, data, stats, conditions, config,
                                     rng, args.device, args.n_cells, genes=genes)
            model = float(np.mean([r["l2"] for r in rows]))
            floor = float(np.mean(ceiling(vae, data, stats, conditions, genes, rng,
                                          args.device, args.n_cells)))
            print(f"{name[:33]:34s} {group:14s} {len(conditions):4d} {model:9.4f} "
                  f"{floor:8.4f} {model - floor:15.4f}")
            name = ""


if __name__ == "__main__":
    main()
