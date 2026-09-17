"""How much of a combination's transport the learned composition term carries.

    python scripts/diagnose_composition.py results/runs/fin_combosciplex_scdfm7_affine_learned_f0 \
        --device cuda
    python scripts/diagnose_composition.py results/runs/<validation run> --scored --device cuda

The field is
    v(z,t,{a})   = u_a
    v(z,t,{a,b}) = u_a + u_b + rho(phi(u_a) + phi(u_b))
so rho never sees a single. A drug trained ONLY inside combinations - both test
singles of Table 3 - gets its u_a from combination losses alone, and nothing forces
u_a rather than rho to carry that drug's own effect. If rho carries a large part
of the training combinations' transport, predicting such a drug alone (u_a with rho
switched off) is extrapolating out of how it was fitted. That is the
identifiability question behind a rho penalty, and this script measures it before
any such penalty is trained.

Per combination, from the same control cells:
  v_ratio     mean ||rho|| / mean ||u_a + u_b|| along the full-field trajectory
  v_cos       mean cosine between rho and u_a + u_b (negative: rho opposes the sum)
  disp_share  ||mean z_full - mean z_add|| / ||mean z_full - mean z0||, where
              z_add integrates the field with rho switched off
  l2_full     eq. (15) L2 of the full prediction against the condition mean
  l2_add      the same with rho switched off
Combinations are grouped by how many of their two drugs have a training single.

Training combinations by default. --scored adds the conditions the run scores, and
is refused unless the run is a validation run: on a table run those are the
reported test set, and a design decision must never read them.
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
from src.eval.predict import _head_aux  # noqa: E402
from src.models.flow import integrate  # noqa: E402


class AdditiveOnly:
    """The same generators with rho switched off: v = sum_a u_a."""

    def __init__(self, field):
        self.field = field

    def __call__(self, z, t, perturbations):
        velocity = self.field.generator(z, t, perturbations[0])
        for pert in perturbations[1:]:
            velocity = velocity + self.field.generator(z, t, pert)
        return velocity


@torch.no_grad()
def velocity_split(field, z0: torch.Tensor, perturbations: list[int], n_steps: int):
    """Mean ||rho||, mean ||sum u||, mean cosine, over the full-field RK4 grid."""
    z, dt = z0, 1.0 / n_steps
    rho_norm, sum_norm, cosines = [], [], []
    for step in range(n_steps):
        t = torch.full((1,), step * dt, device=z.device, dtype=z.dtype)
        velocities = [field.generator(z, t, pert) for pert in perturbations]
        additive = torch.stack(velocities).sum(dim=0)
        correction = field.compose(velocities)
        rho_norm.append(float(correction.norm(dim=1).mean()))
        sum_norm.append(float(additive.norm(dim=1).mean()))
        cosines.append(float(torch.nn.functional.cosine_similarity(
            correction, additive, dim=1).mean()))
        # advance along the full field, as inference does
        k1 = field(z, t, perturbations)
        k2 = field(z + 0.5 * dt * k1, t + 0.5 * dt, perturbations)
        k3 = field(z + 0.5 * dt * k2, t + 0.5 * dt, perturbations)
        k4 = field(z + dt * k3, t + dt, perturbations)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return float(np.mean(rho_norm)), float(np.mean(sum_norm)), float(np.mean(cosines))


@torch.no_grad()
def decoded_mean(vae, z: torch.Tensor, x_aux: torch.Tensor, chunk: int = 256) -> np.ndarray:
    parts = [vae.reconstruction(vae.decode_z(z[i:i + chunk]), **_head_aux(vae, x_aux))
             for i in range(0, z.shape[0], chunk)]
    return torch.cat(parts).mean(dim=0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-cells", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0,
                        help="subsample the training combinations (0 = all)")
    parser.add_argument("--scored", action="store_true",
                        help="also measure the scored combinations (validation runs only)")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    config, data, stats, fold, vae, field = load_run(args.run_dir, args.device, "soft")
    if field is None:
        raise SystemExit("this checkpoint holds no field")
    if getattr(field, "composition_kind", "additive") != "learned":
        raise SystemExit("composition is additive: there is no rho to measure")
    if getattr(field, "anchored", False):
        raise SystemExit("anchored runs drop sum u_a for combinations; this measurement "
                         "does not apply")
    if args.scored and not config["split"].get("validation"):
        raise SystemExit("--scored reads the conditions this run is scored on, which for a "
                         "non-validation run is a reported test set. Refused.")

    rng = np.random.default_rng(config["eval"]["seed"])
    groups = condition_groups(data, stats, fold, config["split"]["method"])
    naming = data.naming
    anchored_drugs = {naming.genes(c)[0] for c in groups["train singles"]}
    todo = [("train", c) for c in subsample(groups["train doubles"], args.limit, rng)]
    if args.scored:
        todo += [("scored", c) for c in groups["test doubles"]]
    genes = scdfm_eval_genes(data, fold, 1000)
    n_steps = config["train"]["n_integration_steps"]
    additive_field = AdditiveOnly(field)
    control = data.cells(data.control_condition)

    rows = []
    print(f"{'set':6s} {'combination':34s} {'singles':>7s} {'v_ratio':>8s} {'v_cos':>7s} "
          f"{'disp_share':>10s} {'l2_full':>8s} {'l2_add':>8s}")
    for which, condition in todo:
        perturbations = [data.pert_index[g] for g in naming.genes(condition)]
        pick = rng.choice(control.shape[0], size=min(args.n_cells, control.shape[0]),
                          replace=False)
        x0 = torch.as_tensor(control[pick], device=args.device)
        with torch.no_grad():
            z0, _ = vae.encode_z(x0)
            z_full = integrate(field, z0, perturbations, n_steps)
            z_add = integrate(additive_field, z0, perturbations, n_steps)
        rho_norm, sum_norm, v_cos = velocity_split(field, z0, perturbations, n_steps)
        origin = z0.mean(dim=0)
        displacement = float((z_full.mean(dim=0) - origin).norm())
        by_rho = float((z_full.mean(dim=0) - z_add.mean(dim=0)).norm())
        truth = stats.mean[condition][genes]
        full_mean = decoded_mean(vae, z_full, x0)[genes]
        add_mean = decoded_mean(vae, z_add, x0)[genes]
        n_anchored = sum(g in anchored_drugs for g in naming.genes(condition))
        row = {"set": which, "condition": condition, "anchored_singles": n_anchored,
               "v_ratio": rho_norm / max(sum_norm, 1e-12), "v_cos": v_cos,
               "disp_share": by_rho / max(displacement, 1e-12),
               "l2_full": float(np.linalg.norm(full_mean - truth)),
               "l2_add": float(np.linalg.norm(add_mean - truth))}
        rows.append(row)
        print(f"{which:6s} {condition[:33]:34s} {n_anchored:7d} {row['v_ratio']:8.3f} "
              f"{row['v_cos']:7.3f} {row['disp_share']:10.3f} {row['l2_full']:8.4f} "
              f"{row['l2_add']:8.4f}")

    print("\nsummary (median over combinations)")
    print(f"  {'set':6s} {'singles in pair':>15s} {'n':>3s} {'v_ratio':>8s} {'v_cos':>7s} "
          f"{'disp_share':>10s} {'l2_full':>8s} {'l2_add':>8s}")
    for which in ("train", "scored"):
        for anchored in (None, 2, 1, 0):
            chosen = [r for r in rows if r["set"] == which
                      and (anchored is None or r["anchored_singles"] == anchored)]
            if not chosen:
                continue
            label = "all" if anchored is None else str(anchored)
            med = {k: float(np.median([r[k] for r in chosen]))
                   for k in ("v_ratio", "v_cos", "disp_share", "l2_full", "l2_add")}
            print(f"  {which:6s} {label:>15s} {len(chosen):3d} {med['v_ratio']:8.3f} "
                  f"{med['v_cos']:7.3f} {med['disp_share']:10.3f} {med['l2_full']:8.4f} "
                  f"{med['l2_add']:8.4f}")
    print("\nreading: disp_share near 0 means rho barely moves the prediction and a rho "
          "penalty\nhas little to act on; a large share, largest where neither drug has a "
          "training\nsingle, is the identifiability risk for single-drug prediction.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
