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
                  device: str, naming, anchor: dict | None = None,
                  alpha: tuple[str, float] | None = None) -> np.ndarray:
    """`naming` parses the condition; `anchor` is eval.baselines.anchor_deltas.

    naming is positional and has no default on purpose. It used to fall back to
    the module-level 'ctrl' convention, which is right for Norman and wrong for
    combosciplex, where the control is spelled control+control: the fallback
    turned 'control+Panobinostat' into two perturbations and raised
    KeyError: 'control' from inside stage 2, a thousand lines from the cause.

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

    `alpha` is the global magnitude correction, (mode, value) - see apply_alpha. It
    rides the same choke point and the same `field.<attr>` fallback for the same
    reason, so every scoring path gets it or none does. Pass ("none", 1.0) to force
    raw predictions, which is what fit_alpha needs.
    """
    vae.eval()
    field.eval()
    if anchor is None:
        anchor = getattr(field, "anchor_table", None)
    perturbations = [pert_index[g] for g in condition_genes(condition, naming)]
    source = control_cells
    shift = (anchor or {}).get(condition)
    if shift is not None:
        control_cells = control_cells + shift.astype(control_cells.dtype, copy=False)
    x0 = torch.as_tensor(control_cells, device=device)
    z0, _ = vae.encode_z(x0)
    z1 = integrate(field, z0, perturbations, n_steps)
    predicted = vae.reconstruction(vae.decode_z(z1),
                                  **_head_aux(vae, x0)).cpu().numpy()
    if alpha is None:
        alpha = getattr(field, "magnitude_alpha", None)
    # `source` and not `control_cells`: the displacement alpha corrects is measured
    # from the control population, and under an anchor control_cells has already
    # been shifted by w_A + w_B, which is part of the displacement, not its origin.
    return apply_alpha(predicted, source, alpha)


def apply_alpha(predicted: np.ndarray, control_cells: np.ndarray,
                alpha: tuple[str, float] | None) -> np.ndarray:
    """Scale the predicted displacement by a global factor. `alpha` is (mode, value).

    The model's displacement is systematically SHORT - ratio 0.646 on training
    singles, the conditions flow matching supervises most directly - and one scalar
    fitted on training conditions moved 5-fold Norman L2 2.2482 -> 2.1418 with DS
    0.7750 -> 0.8956. A per-condition oracle alpha only reaches 1.94 from 2.25, so
    this is close to all a magnitude correction can buy; the rest is direction.

    The two modes differ by exactly (alpha - 1) times the CENTRED displacement:

        mean_i = p_i + (alpha - 1) * mean(p - c)      one shift for every cell
        cell_i = p_i + (alpha - 1) * (p_i - c_i)      each cell's own

    Both give the same population mean, so L2 cannot tell them apart. mean leaves
    the predicted population's shape exactly as the decoder produced it; cell also
    scales how much the displacement VARIES between cells, which is the smaller part
    of the spread here - a transported population inherits the control population's
    heterogeneity, and both modes carry that through untouched. Expect the two to
    score close on edist_rel and DS rather than far apart; both are post-hoc, so one
    checkpoint scores both and the comparison is free.

    Clamped at zero: these are log1p values and cannot be negative, the same reason
    HurdleHead.point_estimate clamps its sampled magnitude.
    """
    if alpha is None:
        return predicted
    mode, value = alpha
    if mode == "none" or value == 1.0:
        return predicted
    if mode == "mean":
        delta = predicted.mean(axis=0) - control_cells.mean(axis=0)
        corrected = predicted + (value - 1.0) * delta
    elif mode == "cell":
        corrected = control_cells + value * (predicted - control_cells)
    else:
        raise ValueError(f"unknown eval.magnitude_alpha {mode!r}")
    return np.clip(corrected, 0.0, None)


@torch.no_grad()
def fit_alpha(vae, field, data, stats, train_conditions: list[str], config,
              rng: np.random.Generator, anchor: dict | None = None) -> float:
    """The least-squares global scale, fitted on TRAINING conditions only.

        alpha = sum_c <d_hat_c, d_c> / sum_c ||d_hat_c||^2

    which is the scalar minimising sum_c ||alpha * d_hat_c - d_c||^2 - the same
    quantity L2 is computed from, so the fit targets the metric rather than a proxy.
    Reading a held-out condition here would make the correction illegitimate, so the
    loop takes `train_conditions` and nothing else; the measured alpha_train
    (1.16-1.38) came out close to the test-optimal alpha, which is why one scalar
    fitted this way transfers.

    Run with eval.magnitude_alpha unset, i.e. on the RAW predictions: fitting on
    already-corrected ones would compound the factor.
    """
    device = config["train"]["device"]
    n_gen = config["eval"]["n_gen_cells"]
    n_steps = config["train"]["n_integration_steps"]
    control_cells = data.cells(data.control_condition)

    numerator, denominator = 0.0, 0.0
    for condition in train_conditions:
        if data.naming.is_control(condition) or not stats.has(condition):
            continue
        pick = rng.choice(control_cells.shape[0],
                          size=min(n_gen, control_cells.shape[0]), replace=False)
        control_sample = control_cells[pick]
        predicted = predict_cells(vae, field, control_sample, condition,
                                  data.pert_index, n_steps, device, data.naming,
                                  anchor, alpha=("none", 1.0))
        d_hat = predicted.mean(axis=0) - control_sample.mean(axis=0)
        d = stats.delta(condition)
        numerator += float(d_hat @ d)
        denominator += float(d_hat @ d_hat)
    if denominator <= 0.0:
        return 1.0
    return numerator / denominator


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
                                      data.pert_index, n_steps, device,
                                      data.naming, anchor)

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
