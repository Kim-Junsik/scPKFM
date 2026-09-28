"""The identifiability changes, asserted where they are supposed to hold.

Three pieces, all aimed at one measured failure: 13 of combosciplex's 17 drugs
never appear alone in training, both drugs behind its two test singles are of that
kind, and the combination data fix u_A + u_B + rho without fixing how it splits -
so those operators are free. scripts/rho_off_probe.py measured the consequence:
held-out Dasatinib alone pointed OPPOSITE its true shift (cosine -0.26, -0.31).

  model.composition_orthogonal   rho may add directions the generators do not
                                 produce, never rescale the ones they do
  baselines.pseudo_single_shifts the ridge estimate of a drug's own effect, for the
                                 singles training does not contain
  eval.magnitude_alpha           the post-hoc global magnitude correction

These are structural properties, not accuracy claims: each one must hold without
training, so a failure means the construction is wrong.

    python -m pytest tests/test_identifiability.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.data.conventions import ConditionNaming
from src.eval import baselines
from src.eval.predict import apply_alpha
from src.models.flow import PKFMField, _orthogonalise

LATENT = 8
N_PERTURBATIONS = 6
BATCH = 4


def build(orthogonal: bool, latent_dim: int = LATENT) -> PKFMField:
    config = config_module.load([f"model.latent_dim={latent_dim}",
                                 "model.composition=learned",
                                 f"model.composition_orthogonal={orthogonal}"])
    torch.manual_seed(0)
    field = PKFMField(config, N_PERTURBATIONS)
    # rho's output layer is zero-initialised, so every test below would pass for
    # the wrong reason on an untrained field. Same for the generators.
    with torch.no_grad():
        field.compose.rho[-1].weight.normal_(0.0, 0.3)
        field.compose.rho[-1].bias.normal_(0.0, 0.3)
        field.generator.a.normal_(0.0, 0.2)
        field.generator.b.normal_(0.0, 0.2)
    return field


@pytest.fixture
def state():
    torch.manual_seed(1)
    return torch.randn(BATCH, LATENT), torch.rand(BATCH)


# --------------------------------------------------------------- orthogonal rho

def test_orthogonal_rho_leaves_the_additive_projection_alone(state):
    """THE POINT. The field's component along each u_a is exactly u_a's own.

    This is what identifies the operator of a drug seen only in combinations: the
    part of the displacement inside span{u_a} can only come from the generators, so
    a well-sampled partner's operator plus the data pins the data-poor one.
    """
    z, t = state
    field = build(orthogonal=True)
    perturbations = [0, 3]
    velocity = field(z, t, perturbations)
    additive = sum(field.generator(z, t, p) for p in perturbations)

    for pert in perturbations:
        u = field.generator(z, t, pert)
        unit = u / u.norm(dim=-1, keepdim=True)
        got = (velocity * unit).sum(-1)
        want = (additive * unit).sum(-1)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_orthogonal_rho_still_changes_the_field(state):
    """It restricts rho, it does not delete it: the complement is not empty.

    At latent_dim 8 with two perturbations the span is 2-dimensional, so six
    directions remain. If this failed, the test above would be passing because the
    correction had been zeroed rather than projected.
    """
    z, t = state
    field = build(orthogonal=True)
    perturbations = [0, 3]
    velocity = field(z, t, perturbations)
    additive = sum(field.generator(z, t, p) for p in perturbations)
    assert (velocity - additive).norm() > 1e-3


def test_orthogonal_rho_is_a_noop_at_initialisation():
    """rho starts at zero, so turning the flag on cannot move step 0.

    That is why it needs no warm-up and why a run with it on is comparable to one
    without from the first step.
    """
    torch.manual_seed(1)
    z, t = torch.randn(BATCH, LATENT), torch.rand(BATCH)
    config = config_module.load([f"model.latent_dim={LATENT}",
                                 "model.composition=learned",
                                 "model.composition_orthogonal=True"])
    torch.manual_seed(0)
    field = PKFMField(config, N_PERTURBATIONS)
    with torch.no_grad():  # generators only; leave rho at its zero init
        field.generator.a.normal_(0.0, 0.2)
        field.generator.b.normal_(0.0, 0.2)
    additive = sum(field.generator(z, t, p) for p in [0, 3])
    torch.testing.assert_close(field(z, t, [0, 3]), additive)


def test_orthogonal_rho_keeps_the_structural_guarantees(state):
    """Control is zero, a single is exactly its generator, order does not matter."""
    z, t = state
    field = build(orthogonal=True)
    torch.testing.assert_close(field(z, t, []), torch.zeros_like(z))
    torch.testing.assert_close(field(z, t, [2]), field.generator(z, t, 2))
    torch.testing.assert_close(field(z, t, [1, 4]), field(z, t, [4, 1]))


def test_orthogonalise_survives_parallel_velocities():
    """Two generators producing the same velocity must not blow the projection up.

    The second Gram-Schmidt residual is zero there, and normalising it would turn
    numerical noise into a projection direction.
    """
    torch.manual_seed(0)
    u = torch.randn(BATCH, LATENT)
    correction = torch.randn(BATCH, LATENT)
    out = _orthogonalise(correction, [u, u.clone()])
    assert torch.isfinite(out).all()
    # One direction removed, and it is u's.
    torch.testing.assert_close((out * u).sum(-1), torch.zeros(BATCH),
                               rtol=0, atol=1e-5)


def test_orthogonalise_removes_the_whole_span():
    """With independent velocities the result is orthogonal to every one of them."""
    torch.manual_seed(0)
    velocities = [torch.randn(BATCH, LATENT) for _ in range(3)]
    out = _orthogonalise(torch.randn(BATCH, LATENT), velocities)
    for u in velocities:
        torch.testing.assert_close((out * u).sum(-1), torch.zeros(BATCH),
                                   rtol=0, atol=1e-5)


# ------------------------------------------------------------- pseudo singles

class _Stats:
    """The slice of ConditionMeans that pseudo_single_shifts reads."""

    def __init__(self, naming, deltas: dict[str, np.ndarray], n_genes: int):
        self.naming = naming
        self._deltas = deltas
        self.mean = {c: np.zeros(n_genes) for c in deltas}
        self.n = {c: 100 for c in deltas}

    def has(self, condition: str) -> bool:
        return condition in self._deltas

    def delta(self, condition: str) -> np.ndarray:
        return self._deltas[condition]


def _combosciplex_shaped(n_genes: int = 5):
    """Training conditions with combosciplex's shape, at toy size.

    Givinostat is well sampled (a single and two combinations), X appears in
    exactly one training condition next to it, and Y only in a combination - the
    structure of every test condition in Table 3.
    """
    naming = ConditionNaming(control="control", separator="+")
    rng = np.random.default_rng(0)
    truth = {g: rng.normal(size=n_genes) for g in ("Givinostat", "X", "Y")}
    train = ["control+Givinostat", "Givinostat+X", "Givinostat+Y"]
    deltas = {c: sum(truth[g] for g in naming.genes(c)) for c in train}
    deltas["control+control"] = np.zeros(n_genes)
    return naming, truth, train, _Stats(naming, deltas, n_genes)


def test_pseudo_singles_cover_exactly_the_drugs_without_a_training_single():
    naming, _, train, stats = _combosciplex_shaped()
    shifts = baselines.pseudo_single_shifts(stats, train)
    assert set(shifts) == {"X+control", "Y+control"}
    # Givinostat has its own training single and is already supervised directly; a
    # weaker estimate must not compete with a measured condition mean.
    assert "Givinostat+control" not in shifts


def test_pseudo_single_shift_recovers_the_effect_it_is_standing_in_for():
    """With an exactly additive system the ridge estimate is the true single effect.

    Not a claim that biology is additive - it is the claim that this estimate is
    the right TARGET, i.e. that it identifies u_X from combinations alone rather
    than encoding something else. alpha shrinks it, so the direction is what is
    asserted, not the length.
    """
    naming, truth, train, stats = _combosciplex_shaped()
    shifts = baselines.pseudo_single_shifts(stats, train, alpha=1e-6)
    for gene, key in (("X", "X+control"), ("Y", "Y+control")):
        got, want = shifts[key], truth[gene]
        cosine = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want)))
        assert cosine > 0.99, f"{gene}: cosine {cosine}"


def test_pseudo_singles_read_no_held_out_condition():
    """Adding a held-out condition to stats must not change a single shift.

    The estimate has to be legitimate: stats holds the mean of EVERY condition
    because the metrics need them, so the discipline is which keys are read.
    """
    naming, _, train, stats = _combosciplex_shaped()
    before = baselines.pseudo_single_shifts(stats, train)
    rng = np.random.default_rng(1)
    stats._deltas["X+Y"] = rng.normal(size=5) * 10.0
    stats.mean["X+Y"] = np.zeros(5)
    stats.n["X+Y"] = 100
    after = baselines.pseudo_single_shifts(stats, train)
    assert set(before) == set(after)
    for key in before:
        np.testing.assert_allclose(before[key], after[key])


def test_pseudo_singles_are_empty_when_every_drug_has_one():
    """Norman's additive split: every single is a training condition, so no term."""
    naming = ConditionNaming(control="ctrl", separator="+")
    rng = np.random.default_rng(0)
    truth = {g: rng.normal(size=4) for g in ("A", "B")}
    train = ["A+ctrl", "B+ctrl", "A+B"]
    deltas = {c: sum(truth[g] for g in naming.genes(c)) for c in train}
    deltas["ctrl"] = np.zeros(4)
    assert baselines.pseudo_single_shifts(_Stats(naming, deltas, 4), train) == {}


# ------------------------------------------------------------- magnitude alpha

def test_alpha_none_is_the_identity():
    predicted = np.array([[1.0, 2.0], [3.0, 0.0]])
    control = np.zeros_like(predicted)
    np.testing.assert_allclose(apply_alpha(predicted, control, None), predicted)
    np.testing.assert_allclose(apply_alpha(predicted, control, ("mean", 1.0)), predicted)
    np.testing.assert_allclose(apply_alpha(predicted, control, ("cell", 1.0)), predicted)


def test_alpha_mean_scales_the_mean_displacement_and_not_the_spread():
    rng = np.random.default_rng(0)
    control = rng.normal(2.0, 1.0, size=(64, 5))
    predicted = control + 1.0 + rng.normal(0.0, 0.3, size=(64, 5))
    out = apply_alpha(predicted, control, ("mean", 1.5))
    want = 1.5 * (predicted.mean(0) - control.mean(0))
    np.testing.assert_allclose(out.mean(0) - control.mean(0), want, atol=1e-10)
    # THE reason mode=mean exists: edist_rel and DS read the spread, and this mode
    # must not touch it.
    np.testing.assert_allclose(out.std(0), predicted.std(0), atol=1e-10)


def test_alpha_cell_differs_from_mean_by_the_centred_displacement():
    """The exact difference between the two modes, so neither is assumed.

        mean_i = p_i + (a-1) * mean(p - c)
        cell_i = p_i + (a-1) * (p_i - c_i)

    so they differ by (a-1) times the CENTRED displacement. mode=cell therefore
    scales only how much the displacement varies between cells - not the
    heterogeneity the prediction inherited from the control population, which both
    modes carry through untouched. That is why the two scores come out close on a
    transport model: the inherited spread dominates.
    """
    rng = np.random.default_rng(0)
    control = rng.normal(2.0, 1.0, size=(64, 5))
    predicted = control + 1.0 + rng.normal(0.0, 0.3, size=(64, 5))
    displacement = predicted - control
    centred = displacement - displacement.mean(0)

    by_mean = apply_alpha(predicted, control, ("mean", 1.5))
    by_cell = apply_alpha(predicted, control, ("cell", 1.5))
    np.testing.assert_allclose(by_cell - by_mean, 0.5 * centred, atol=1e-10)
    # Same population mean either way: only the per-cell part is treated differently.
    np.testing.assert_allclose(by_cell.mean(0), by_mean.mean(0), atol=1e-10)


def test_alpha_never_emits_a_negative_expression():
    """log1p values cannot be negative; scaling a downward shift could get there."""
    control = np.full((8, 3), 0.1)
    predicted = np.zeros((8, 3))
    for mode in ("mean", "cell"):
        out = apply_alpha(predicted, control, (mode, 3.0))
        assert (out >= 0.0).all(), mode


def test_alpha_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="magnitude_alpha"):
        apply_alpha(np.zeros((2, 2)), np.zeros((2, 2)), ("linear", 1.2))
