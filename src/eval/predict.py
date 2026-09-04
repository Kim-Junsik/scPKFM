"""Turn a trained model into the same quantities the baselines are scored on.

Prediction is TRANSPORT, not generation: control cells are encoded, carried along
the velocity field to t=1, and decoded. Nothing is sampled from a prior, so the
predicted population inherits the control population's own heterogeneity.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.dataset import condition_genes
from ..models.flow import integrate
from . import metrics


def _head_aux(vae, x: torch.Tensor) -> dict:
    """Extra inputs a head may need at reconstruction time.

    Only the zinb head ever wanted anything (a library size), and this build has
    only the hurdle head, so there is nothing to pass. Kept as a function because
    three call sites spread it into vae.reconstruction(...), and because a head
    that needs auxiliary inputs would plug in here.
    """
    return {}


@torch.no_grad()
def autoencode(vae, cells: np.ndarray, device: str) -> np.ndarray:
    """decode(encode(x)) with no transport at all.

    This is the floor of the whole latent approach: a prediction cannot be closer
    to the truth than the autoencoder's own reconstruction of the control cells,
    because every prediction passes through the same bottleneck. The gene-space
    baselines never pay this cost, so comparing a latent model against them
    without reporting this number would hide where an error actually comes from.
    """
    vae.eval()
    x = torch.as_tensor(cells, device=device)
    mu, _ = vae.encode(x)
    return vae.reconstruction(vae.decode(mu), **_head_aux(vae, x)).cpu().numpy()


@torch.no_grad()
def predict_cells(vae, field, control_cells: np.ndarray, condition: str,
                  pert_index: dict[str, int], n_steps: int,
                  device: str, anchor: dict | None = None) -> np.ndarray:
    """`anchor` is the table from eval.baselines.anchor_deltas.

    THE choke point for anchoring at inference: every scoring path in this
    repository ends up here, so applying the shift once here is what keeps
    training and inference building z0 the same way. A condition absent from the
    table (every single, and every combination under anchor=none) transports the
    raw control cells, unchanged.

    Left None it falls back to `field.anchor_table`, which scripts/train.py and
    diagnostics.load_run both attach. That fallback is the reason celleval.export,
    diagnostics.measure_transport and eval_scdfm_style did not each need the table
    threaded through their own signatures: forgetting to pass it at one of those
    call sites would silently score an anchored model as if it were unanchored,
    and the number would look like a training failure rather than a plumbing bug.
    """
    vae.eval()
    field.eval()
    if anchor is None:
        anchor = getattr(field, "anchor_table", None)
    perturbations = [pert_index[g] for g in condition_genes(condition)]
    shift = (anchor or {}).get(condition)
    if shift is not None:
        control_cells = control_cells + shift.astype(control_cells.dtype, copy=False)
    x0 = torch.as_tensor(control_cells, device=device)
    z0, _ = vae.encode_z(x0)
    z1 = integrate(field, z0, perturbations, n_steps)
    return vae.reconstruction(vae.decode_z(z1), **_head_aux(vae, x0)).cpu().numpy()


def evaluate_model(vae, field, data, stats, folds, method, config,
                   rng: np.random.Generator, anchor: dict | None = None) -> dict:
    """Same protocol as scripts/run_baselines.py so the numbers are comparable."""
    from ..eval import baselines as baseline_module

    device = config["train"]["device"]
    n_gen = config["eval"]["n_gen_cells"]
    n_steps = config["train"]["n_integration_steps"]
    control_cells = data.cells(data.control_condition)

    numerator, denominator = 0.0, 0.0
    edists, de20s, per_double, floors = [], [], [], []

    for fold in folds:
        test_doubles = [c for c in fold["test"] if data.naming.is_double(c)]
        for double in test_doubles:
            a, b = data.naming.genes(double)
            single_a, single_b = stats.single_of(a), stats.single_of(b)
            if not all(stats.has(c) for c in (double, single_a, single_b)):
                continue

            pick = rng.choice(control_cells.shape[0],
                              size=min(n_gen, control_cells.shape[0]), replace=False)
            control_sample = control_cells[pick]
            predicted = predict_cells(vae, field, control_sample, double,
                                      data.pert_index, n_steps, device, anchor)

            m_hat = predicted.mean(axis=0)
            m_ab, m_a = stats.mean[double], stats.mean[single_a]
            m_b, m_ctrl = stats.mean[single_b], stats.control
            e_noise = sum(float((stats.var[c] / max(stats.n[c], 1)).sum())
                          for c in (double, single_a, single_b,
                                    stats.control_condition))

            r = metrics.residual(m_ab, m_a, m_b, m_ctrl)
            r_hat = metrics.residual(m_hat, m_a, m_b, m_ctrl)
            numerator += float((r_hat - r) @ (r_hat - r)) - e_noise
            denominator += float(r @ r) - e_noise
            per_double.append(metrics.residual_r2(m_hat, m_ab, m_a, m_b, m_ctrl, e_noise))
            de20s.append(metrics.de20_pearson(m_hat - m_ctrl, m_ab - m_ctrl))
            edists.append(metrics.edist_rel(
                predicted, data.cells(double), control_sample,
                power=config["eval"]["edist_power"], device=config["eval"]["device"]))
            # Same comparison with transport switched off, to separate
            # "the flow is wrong" from "the autoencoder cannot represent cells".
            floors.append(metrics.edist_rel(
                autoencode(vae, control_sample, device), data.cells(double),
                control_sample, power=config["eval"]["edist_power"],
                device=config["eval"]["device"]))

    scored = int(np.sum(~np.isnan(per_double))) if per_double else 0
    return {
        "n_evaluated": len(edists),
        # THE headline. Pooled, so no per-condition denominator can blow up.
        "resid_R2_pooled": 1.0 - numerator / denominator if edists else float("nan"),
        # Per-condition mean, over the doubles whose residual clears their own
        # noise floor (see metrics.residual_r2). n says how many that was: a mean
        # over a handful of conditions is a different statement from the pooled
        # value and the two should not be quoted interchangeably.
        "resid_R2_mean": float(np.nanmean(per_double)) if scored else float("nan"),
        "resid_R2_mean_n": scored,
        "edist_rel": float(np.nanmean(edists)) if edists else float("nan"),
        "r_de20": float(np.nanmean(de20s)) if edists else float("nan"),
        # NOT a floor: this is the control run through encode-decode with no
        # transport at all, relative to the raw control, so it is what IDENTITY
        # scores on the same scale as edist_rel. A perfect autoencoder gives
        # exactly 1.0 and the excess over 1 is reconstruction distortion measured
        # in units of the control-to-double distance. A model is expected to come
        # in far below it - pcab_lie_commutator scores 0.5295 against 1.0406 - so
        # it bounds nothing, and it is not the autoencoder's ceiling on resid_R2.
        # That ceiling is a different measurement: scripts/diagnose_bottleneck.py.
        "edist_rel_identity": float(np.nanmean(floors)) if floors else float("nan"),
        # DEPRECATED alias, kept so older result files and readers still line up.
        "edist_rel_autoencoder_floor": float(np.nanmean(floors)) if floors else float("nan"),
    }
