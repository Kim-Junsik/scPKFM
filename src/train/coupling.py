"""Minibatch optimal-transport coupling between control and perturbed latents.

Control and perturbed cells are unpaired, so flow matching needs a rule for which
source cell transports to which target cell. Random pairing would define a
straight path between two arbitrary cells and teach the field to average over
transports that no cell actually makes - the coupling has to carry meaning.

Unbalanced OT is the default because the two marginals are not exchangeable:
perturbation kills and creates cells, so mass is not conserved between the control
and perturbed populations. `random` exists only as an ablation control.
"""

from __future__ import annotations

import numpy as np
import ot
import torch


def _cost_matrix(source: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        cost = torch.cdist(source, target, p=2) ** 2
        return (cost / cost.max().clamp(min=1e-12)).double().cpu().numpy()


def coupling_plan(source: torch.Tensor, target: torch.Tensor, method: str,
                  reg: float, reg_marginal: float) -> np.ndarray:
    n, m = source.shape[0], target.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)

    if method == "random":
        return np.outer(a, b)
    cost = _cost_matrix(source, target)
    if method == "ot":
        return ot.emd(a, b, cost)
    if method == "uot":
        return ot.unbalanced.sinkhorn_unbalanced(a, b, cost, reg=reg, reg_m=reg_marginal)
    raise ValueError(f"unknown coupling {method!r}")


def sample_pairs(source: torch.Tensor, target: torch.Tensor, method: str,
                 reg: float, reg_marginal: float,
                 rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw index pairs from the transport plan, treating it as a joint law.

    Sampling rather than taking a hard assignment keeps the plan's mass structure:
    a source cell that transports to several targets should be trained toward all
    of them in proportion, not toward its arg-max alone.
    """
    plan = coupling_plan(source, target, method, reg, reg_marginal)
    flat = plan.reshape(-1)
    total = flat.sum()
    if not np.isfinite(total) or total <= 0:  # degenerate plan -> fall back to random
        flat = np.full_like(flat, 1.0 / flat.size)
    else:
        flat = flat / total

    n_pairs = source.shape[0]
    picks = rng.choice(flat.size, size=n_pairs, p=flat, replace=True)
    rows, columns = np.divmod(picks, plan.shape[1])
    return (source[torch.as_tensor(rows, device=source.device)],
            target[torch.as_tensor(columns, device=target.device)])
