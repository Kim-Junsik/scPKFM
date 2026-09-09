"""How much of a run's L2 belongs to the autoencoder and how much to the flow.

    python scripts/diagnose_bottleneck.py results/runs/main_affine_learned_f1 \
        --device cuda --gate soft --infer-top-gene 1000

The transport is replaced by the ground truth: the REAL perturbed cells are encoded
and decoded, and the same eq. (15) norm is taken against the same condition mean on
the same gene subset paper_table.py uses. Whatever L2 survives that substitution is
the decoder's floor - the number a perfect flow would still pay - so the difference
against the run's reported L2 localises the error to one half of the model.

Without this the two halves are not separable, and the tempting reading of a large
L2 ("reconstruction is bad, change the recon loss") is exactly the one the number
can refute: measured on the five main_* folds the ceiling is 1.61 while the runs
score 2.23, i.e. the flow contributes 0.62 and the decoder alone would already beat
the 1.70 this project is compared against.

A sampling floor is printed alongside. It is the L2 of the n_cells subsample's own
mean against the full condition mean, and it is 0 whenever a condition has fewer
cells than --n-cells - which is the case for every Norman test double at 1024. A
non-zero value there means the ceiling below carries subsampling noise.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.diagnostics import condition_groups, load_run, scdfm_eval_genes
from src.eval.predict import _head_aux


@torch.no_grad()
def bottleneck(run_dir: str, device: str, n_cells: int, gate: str | None,
               infer_top_gene: int | None) -> dict:
    config, data, stats, fold, vae, _ = load_run(run_dir, device, gate)
    rng = np.random.default_rng(config["eval"]["seed"])
    conditions = condition_groups(data, stats, fold,
                                  config["split"]["method"])["test doubles"]
    genes = scdfm_eval_genes(data, fold, infer_top_gene) if infer_top_gene else None

    sampling, ceiling = [], []
    for condition in conditions:
        if condition == data.control_condition or not stats.has(condition):
            continue
        cells = data.cells(condition)
        take = rng.choice(cells.shape[0], size=min(n_cells, cells.shape[0]),
                          replace=False)
        sample = cells[take]
        truth = stats.mean[condition]
        if genes is not None:
            truth = truth[genes]
        x = torch.as_tensor(sample, device=device)
        # The flow is skipped entirely: encode the truth, decode it back. z1_hat
        # is replaced by z1_true, which is what "perfect transport" means here.
        z, _ = vae.encode_z(x)
        reconstructed = vae.reconstruction(vae.decode_z(z), **_head_aux(vae, x))
        recon_mean = reconstructed.mean(dim=0).cpu().numpy()
        sample_mean = sample.mean(0)
        if genes is not None:
            recon_mean, sample_mean = recon_mean[genes], sample_mean[genes]
        sampling.append(float(np.linalg.norm(sample_mean - truth)))
        ceiling.append(float(np.linalg.norm(recon_mean - truth)))

    return {"n": len(ceiling), "sampling": float(np.mean(sampling)),
            "ceiling": float(np.mean(ceiling))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cells", type=int, default=1024)
    parser.add_argument("--gate", default=None, choices=["soft", "hard", "sample"])
    parser.add_argument("--infer-top-gene", type=int, default=None,
                        help="restrict to the scanpy-HVG subset scDFM scores on")
    args = parser.parse_args()

    print(f"{'run':40}{'n':>4}{'sampling':>11}{'ceiling':>10}")
    ceilings = []
    for run_dir in args.runs:
        row = bottleneck(run_dir, args.device, args.n_cells, args.gate,
                         args.infer_top_gene)
        ceilings.append(row["ceiling"])
        print(f"{os.path.basename(run_dir):40}{row['n']:4d}"
              f"{row['sampling']:11.4f}{row['ceiling']:10.4f}")
    if len(ceilings) > 1:
        print(f"{'MEAN':40}{'':4}{'':11}{np.mean(ceilings):10.4f}")


if __name__ == "__main__":
    main()
