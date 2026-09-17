"""Where the direction of a predicted shift is lost: in the flow or in the decoder.

    python scripts/diagnose_decoder.py results/runs/<validation run> --device cuda
    python scripts/diagnose_decoder.py results/runs/fin_affine_learned_f1 --group train --device cuda

For each condition, from control cells transported by the model and from the
condition's own cells encoded and decoded (the decoder ceiling: a perfect flow):

  latent_cos    cos(mean z_hat - mean z0, mean z_true - mean z0), the flow alone
  ceiling_cos   cos(decode(z_true) mean - control mean, true shift)
  model_cos     cos(decode(z_hat) mean - control mean, true shift)
  dec_share     (1 - ceiling_cos) / (1 - model_cos): the part of the model's
                direction loss a perfect flow would still pay
  ceiling_l2    eq. (15) L2 of the ceiling; model_l2 the same for the model
  ceil/ctrl     ceiling_l2 / ||true shift||: the ceiling relative to how far the
                condition moves, comparable between singles and combinations
  top50_cos / rest_cos
                model cosine on the 50 genes with the largest true shift and on
                the remaining evaluation genes; top50_energy is their share of
                the true shift's squared norm
  gate_cos / mag_cos
                first-order split of the model's shift into the detection channel,
                (p1 - p0) * m0, and the magnitude channel, p0 * (m1 - m0), on
                population means of the hurdle head's probability p and magnitude m

All on the scanpy-HVG evaluation genes of the run's scored set (1,000), soft gate.

--group scored (the default for validation runs) measures the conditions the run
scores. On a table run those are the reported test set, so there the default is
--group train and --group scored is refused: a design decision must never read
test conditions. --group train measures training singles and combinations, whose
cells the encoder has seen.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.diagnostics import (condition_groups, load_run, scdfm_eval_genes,  # noqa: E402
                                  subsample)
from src.models.flow import integrate  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / scale if scale else float("nan")


@torch.no_grad()
def head_means(vae, z: torch.Tensor, chunk: int = 256) -> dict[str, np.ndarray]:
    """Population means of the soft point estimate, gate probability and magnitude."""
    soft, prob, mag = [], [], []
    for i in range(0, z.shape[0], chunk):
        params = vae.decode_z(z[i:i + chunk])
        p = torch.sigmoid(params["gate_logit"])
        soft.append(p * params["magnitude"])
        prob.append(p)
        mag.append(params["magnitude"])
    return {name: torch.cat(parts).mean(dim=0).cpu().numpy()
            for name, parts in (("soft", soft), ("p", prob), ("m", mag))}


def measure(vae, field, data, stats, condition: str, genes: np.ndarray, n_cells: int,
            n_steps: int, rng, device: str) -> dict:
    naming = data.naming
    control = data.cells(data.control_condition)
    own = data.cells(condition)
    x0 = torch.as_tensor(control[rng.choice(control.shape[0], size=min(n_cells, control.shape[0]),
                                            replace=False)], device=device)
    x1 = torch.as_tensor(own[rng.choice(own.shape[0], size=min(n_cells, own.shape[0]),
                                        replace=False)], device=device)
    perturbations = [data.pert_index[g] for g in naming.genes(condition)]
    with torch.no_grad():
        z0, _ = vae.encode_z(x0)
        z_true, _ = vae.encode_z(x1)
        z_hat = integrate(field, z0, perturbations, n_steps)

    origin = z0.mean(dim=0)
    latent_cos = cosine((z_hat.mean(dim=0) - origin).cpu().numpy(),
                        (z_true.mean(dim=0) - origin).cpu().numpy())
    base, ceiling, model = (head_means(vae, z) for z in (z0, z_true, z_hat))

    ctrl = stats.control[genes]
    truth = stats.mean[condition][genes]
    shift = truth - ctrl
    ceiling_shift = ceiling["soft"][genes] - ctrl
    model_shift = model["soft"][genes] - ctrl

    top = np.argsort(-np.abs(shift))[:50]
    rest = np.setdiff1d(np.arange(len(genes)), top)
    p0, m0 = base["p"][genes], base["m"][genes]
    gate_part = (model["p"][genes] - p0) * m0
    mag_part = p0 * (model["m"][genes] - m0)

    ceiling_cos = cosine(ceiling_shift, shift)
    model_cos = cosine(model_shift, shift)
    shift_norm = float(np.linalg.norm(shift))
    return {
        "condition": condition,
        "block": "single" if naming.is_single(condition) else "double",
        "n_cells": int(own.shape[0]),
        "latent_cos": latent_cos,
        "ceiling_cos": ceiling_cos,
        "model_cos": model_cos,
        "dec_share": (1 - ceiling_cos) / (1 - model_cos) if model_cos < 1 else float("nan"),
        "shift_norm": shift_norm,
        "ceiling_l2": float(np.linalg.norm(ceiling["soft"][genes] - truth)),
        "model_l2": float(np.linalg.norm(model["soft"][genes] - truth)),
        "ceil_over_ctrl": float(np.linalg.norm(ceiling["soft"][genes] - truth)) / shift_norm
        if shift_norm else float("nan"),
        "top50_energy": float((shift[top] ** 2).sum() / max((shift ** 2).sum(), 1e-12)),
        "top50_cos": cosine(model_shift[top], shift[top]),
        "rest_cos": cosine(model_shift[rest], shift[rest]),
        "gate_cos": cosine(gate_part, shift),
        "mag_cos": cosine(mag_part, shift),
        "gate_norm_share": float(np.linalg.norm(gate_part)
                                 / max(np.linalg.norm(gate_part) + np.linalg.norm(mag_part), 1e-12)),
    }


COLUMNS = ["latent_cos", "ceiling_cos", "model_cos", "dec_share", "ceiling_l2", "model_l2",
           "ceil_over_ctrl", "top50_energy", "top50_cos", "rest_cos", "gate_cos", "mag_cos",
           "gate_norm_share"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--group", choices=["scored", "train"], default=None,
                        help="default: scored for validation runs, train otherwise")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-cells", type=int, default=512)
    parser.add_argument("--limit", type=int, default=40,
                        help="subsample training conditions per block (0 = all)")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    config, data, stats, fold, vae, field = load_run(args.run_dir, args.device, "soft")
    if field is None:
        raise SystemExit("this checkpoint holds no field")
    validation = bool(config["split"].get("validation"))
    group = args.group or ("scored" if validation else "train")
    if group == "scored" and not validation:
        raise SystemExit("--group scored on a non-validation run reads a reported test set. "
                         "Refused; use --group train or a validation run.")

    rng = np.random.default_rng(config["eval"]["seed"])
    groups = condition_groups(data, stats, fold, config["split"]["method"])
    if group == "scored":
        conditions = groups["test doubles"] + groups["test singles"]
    else:
        conditions = (subsample(groups["train doubles"], args.limit, rng)
                      + subsample(groups["train singles"], args.limit, rng))
    genes = scdfm_eval_genes(data, fold, 1000)
    n_steps = config["train"]["n_integration_steps"]

    rows = []
    header = f"{'condition':30s} {'block':6s} " + " ".join(f"{c[:10]:>10s}" for c in COLUMNS)
    print(f"group: {group}   ({len(conditions)} conditions, {len(genes)} genes)\n{header}")
    for condition in conditions:
        row = measure(vae, field, data, stats, condition, genes, args.n_cells, n_steps,
                      rng, args.device)
        rows.append(row)
        print(f"{condition[:29]:30s} {row['block']:6s} "
              + " ".join(f"{row[c]:10.3f}" for c in COLUMNS))

    print("\nmedian by block")
    print(f"{'block':30s} {'n':>6s} " + " ".join(f"{c[:10]:>10s}" for c in COLUMNS))
    for block in ("double", "single", None):
        chosen = [r for r in rows if block is None or r["block"] == block]
        if not chosen:
            continue
        print(f"{block or 'all':30s} {len(chosen):6d} "
              + " ".join(f"{float(np.nanmedian([r[c] for r in chosen])):10.3f}"
                         for c in COLUMNS))
    print("\nreading: dec_share above 0.5 means most of the direction loss survives a "
          "perfect flow;\nceil/ctrl compares ceilings between blocks on the scale of how far "
          "each condition moves.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
