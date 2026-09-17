"""A perturbation similarity graph that couples the Koopman operators.

    w_ab = max(0, (s_ab - threshold) / (1 - threshold)),   w_aa = 0

s is a symmetric similarity in [0, 1] over perturbation names - for combosciplex
the ECFP4 Tanimoto of the drugs (assets/drugs/tanimoto_ecfp4_2048.csv, built by
scripts/drug_similarity.py from structure alone). The graph is used two ways,
selected by model.operator_graph_mode:

  penalty  the stage-2 loss adds, per edge, the relative squared distance between
           the two perturbations' operators (operator_graph_penalty below). The
           field itself is unchanged, so an operator with plenty of data keeps it
           while a data-poor one is pulled toward its structural neighbours.
  mix      each perturbation's operator is the weighted mean of its own and its
           neighbours' (AffineGenerator.operator), so a condition's data trains
           its neighbours' parameters too.

Threshold 0.25 was fixed on 2026-09-17 before any run: it sits above the 95th
percentile (0.227) of all 136 drug pairs, and leaves exactly six edges -
Alvespimycin-Tanespimycin 0.79, SRT1720-SRT2104 0.33, Dacinostat-Panobinostat
0.25, Givinostat-PCI-34051 0.14, SRT2104-SRT3025 0.09, SRT1720-SRT3025 0.07.
A perturbation without an edge is modelled exactly as without the graph.
"""

from __future__ import annotations

import os

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRAPH_MODES = ("penalty", "mix")


def resolve(path: str) -> str:
    """The path as given, or relative to the repository root."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(ROOT, path)


def load_similarity(path: str) -> tuple[list[str], np.ndarray]:
    """Names from the header line and the square similarity matrix below it."""
    path = resolve(path)
    with open(path, encoding="utf-8") as handle:
        names = handle.readline().strip().split(",")
    matrix = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if matrix.shape != (len(names), len(names)):
        raise ValueError(f"{path}: {len(names)} names but a {matrix.shape} matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError(f"{path}: similarity matrix is not symmetric")
    return names, matrix


def operator_graph(path: str, threshold: float, perturbations: list[str]) -> np.ndarray:
    """[P, P] edge weights in the model's perturbation order.

    Every perturbation of the model must be named in the file: a silently missing
    row would train that perturbation as isolated and look like a result.
    """
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"operator_graph_threshold must be in [0, 1), got {threshold}")
    names, similarity = load_similarity(path)
    index = {name: i for i, name in enumerate(names)}
    missing = [p for p in perturbations if p not in index]
    if missing:
        raise ValueError(f"{path} has no similarity for perturbation(s) {missing}; "
                         f"an operator graph is defined for the drugs it names only")
    order = [index[p] for p in perturbations]
    s = similarity[np.ix_(order, order)]
    weights = np.clip((s - threshold) / (1.0 - threshold), 0.0, None)
    np.fill_diagonal(weights, 0.0)
    return weights.astype(np.float32)


def describe(weights: np.ndarray, perturbations: list[str]) -> str:
    edges = [(float(weights[i, j]), perturbations[i], perturbations[j])
             for i in range(len(perturbations)) for j in range(i + 1, len(perturbations))
             if weights[i, j] > 0]
    if not edges:
        return "no edges - every operator is isolated"
    return ", ".join(f"{a}-{b} {w:.3f}" for w, a, b in sorted(edges, reverse=True))


def operator_graph_penalty(generator, eps: float = 1e-12) -> torch.Tensor:
    """Sum over edges of w_ab * relative squared distance of the two operators.

        w_ab * ( |A_a - A_b|^2 / sg(|A_a|^2 + |A_b|^2)  +  |b_a - b_b|^2 / sg(|b_a|^2 + |b_b|^2) )

    Uses each perturbation's OWN operator, never the mixed one. The denominators
    are detached (sg): they set the scale, so the penalty reads the same whatever
    size the operators reach, but the model cannot lower it by inflating them.
    Zero when every operator is zero, as at initialisation.
    """
    edges = generator.graph_edges()
    total = generator.b.new_zeros(())
    for a, b, weight in edges:
        op_a, op_b = generator.own_matrix(a), generator.own_matrix(b)
        bias_a, bias_b = generator.b[a], generator.b[b]
        op_term = (op_a - op_b).pow(2).sum() / ((op_a.pow(2).sum()
                                                 + op_b.pow(2).sum()).detach() + eps)
        bias_term = (bias_a - bias_b).pow(2).sum() / ((bias_a.pow(2).sum()
                                                       + bias_b.pow(2).sum()).detach() + eps)
        total = total + weight * (op_term + bias_term)
    return total


def penalty_ramp(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """0 during the singles warm-up, then linear to 1 over `ramp_epochs`.

    Decided while implementing, before any run: during warm-up only the drugs with
    a training single move, so a live penalty would pull, say, Panobinostat's
    operator toward Dacinostat's, which is still exactly zero. The ramp gives every
    operator data first.
    """
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)
