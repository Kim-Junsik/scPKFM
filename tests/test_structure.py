"""The claims that must hold by construction, asserted numerically.

These are not accuracy tests. Each one checks a property the architecture is
supposed to guarantee without training, so a failure means the composition is
wrong rather than that the model needs more epochs.

    python -m pytest tests/test_structure.py -v
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.models.flow import PKFMField, integrate

LATENT = 8
N_PERTURBATIONS = 6
BATCH = 4


COMPOSITIONS = ["additive", "learned"]


def build(latent_dim: int = LATENT, composition: str = "additive") -> PKFMField:
    config = config_module.load([f"model.latent_dim={latent_dim}",
                                 f"model.composition={composition}"])
    torch.manual_seed(0)
    field = PKFMField(config, N_PERTURBATIONS)
    # rho's output layer is zero-initialised so training starts additive, which
    # would make every composition test below pass for the wrong reason. Break it.
    if composition == "learned":
        with torch.no_grad():
            field.compose.rho[-1].weight.normal_(0.0, 0.2)
            field.compose.rho[-1].bias.normal_(0.0, 0.2)
    # A_a and b_a are zero-initialised on purpose, so an untrained field is the
    # null field and every test below would pass trivially. Fill them.
    with torch.no_grad():
        field.generator.a.normal_(0.0, 0.2)
        field.generator.b.normal_(0.0, 0.2)
    return field


@pytest.fixture
def state():
    torch.manual_seed(1)
    return torch.randn(BATCH, LATENT), torch.rand(BATCH)


@pytest.mark.parametrize("composition", COMPOSITIONS)
def test_control_is_the_zero_field(state, composition):
    """v(z, t, {}) = 0 exactly, not approximately.

    Control is not a learned condition: an empty perturbation set has to produce
    no motion at all, or the model can drift the control population and every
    delta measured against it is offset.
    """
    z, t = state
    assert torch.equal(build(composition=composition)(z, t, []), torch.zeros_like(z))


@pytest.mark.parametrize("composition", COMPOSITIONS)
def test_single_perturbation_is_its_own_generator(state, composition):
    """v(z, t, {a}) = u_a(z, t): no pair term may appear for a single.

    The learned composition must be gated on |S| >= 2 for this to hold. Applying
    it to a single would corrupt exactly the conditions the endpoint loss
    supervises most directly.
    """
    z, t = state
    field = build(composition=composition)
    assert torch.allclose(field(z, t, [2]), field.generator(z, t, 2), atol=1e-6)


def test_composition_is_the_sum_of_generators(state):
    """v(z, t, {a, b}) = u_a + u_b, with nothing added on top.

    This build has no interaction term - see src/models/flow.py for the seven
    settings that decided that - so the composed field is exactly the sum. The
    non-additivity of the flow MAP comes from the integrator, not from here.
    """
    z, t = state
    field = build()
    expected = field.generator(z, t, 1) + field.generator(z, t, 4)
    assert torch.allclose(field(z, t, [1, 4]), expected, atol=1e-6)


@pytest.mark.parametrize("composition", COMPOSITIONS)
def test_permutation_invariance(state, composition):
    """{a, b} and {b, a} are the same condition and must give the same field.

    For the learned composition this is what the sum-of-phi (DeepSets) form buys:
    the pooled representation cannot see the order.
    """
    z, t = state
    field = build(composition=composition)
    assert torch.allclose(field(z, t, [1, 4]), field(z, t, [4, 1]), atol=1e-6)


@pytest.mark.parametrize("composition", COMPOSITIONS)
def test_no_parameter_is_indexed_by_a_pair(composition):
    """Every parameter is per-perturbation, never per-pair.

    This is what makes an unseen combination expressible: there is no parameter
    that only a seen pair could have fitted. An earlier version of this project
    stored a [P, P] table for its interaction term; a pair never seen in training
    kept its initial value, so on all 37 evaluation combinations the term was
    numerically zero. The size check below is the cheap guard against that
    returning.
    """
    field = build(composition=composition)
    for name, parameter in field.named_parameters():
        assert N_PERTURBATIONS ** 2 not in parameter.shape, (
            f"{name} has a dimension of size P^2 = {N_PERTURBATIONS ** 2}, "
            "which is how a per-pair table looks")


def test_the_flow_map_is_non_additive_even_though_the_field_is(state):
    """Phi_ab != Phi_a + Phi_b - z0, with an additive velocity field.

    The point of the whole design. Integrating a state-dependent field is a
    nonlinear operation, so composition non-additivity is already present without
    any algebraic interaction term - which is why adding one was measured to be
    redundant rather than helpful.
    """
    z, _ = state
    field = build()
    both = integrate(field, z, [1, 4], n_steps=8)
    only_a = integrate(field, z, [1], n_steps=8)
    only_b = integrate(field, z, [4], n_steps=8)
    residual = both - only_a - only_b + z
    assert residual.abs().max() > 1e-3, (
        "the flow map came out additive; with A_a and A_b commuting this test is "
        "vacuous, so check that build() actually randomised the operators")


def test_integration_returns_the_start_for_the_control():
    """Integrating the empty set moves nothing, at any step count."""
    torch.manual_seed(2)
    z = torch.randn(BATCH, LATENT)
    field = build()
    for n_steps in (1, 5, 20):
        assert torch.allclose(integrate(field, z, [], n_steps), z, atol=1e-7)


def test_learned_composition_starts_exactly_additive(state):
    """At init the learned term contributes nothing at all.

    rho's output layer is zero, so a run with composition=learned begins
    numerically identical to composition=additive. Any difference in the final
    score is then attributable to the term rather than to a different
    initialisation - which is the only way the ablation means anything.
    """
    z, t = state
    config = config_module.load([f"model.latent_dim={LATENT}",
                                 "model.composition=learned"])
    torch.manual_seed(0)
    field = PKFMField(config, N_PERTURBATIONS)
    with torch.no_grad():
        field.generator.a.normal_(0.0, 0.2)
        field.generator.b.normal_(0.0, 0.2)
    expected = field.generator(z, t, 1) + field.generator(z, t, 4)
    assert torch.allclose(field(z, t, [1, 4]), expected, atol=1e-7)


def test_learned_composition_actually_changes_the_field(state):
    """Once rho is nonzero the composition is no longer the plain sum.

    The guard against a term that is wired up but silently inert - which is how
    the [P, P] interaction table failed: it existed, it trained, and it was
    numerically zero on every evaluated combination.
    """
    z, t = state
    field = build(composition="learned")
    plain = field.generator(z, t, 1) + field.generator(z, t, 4)
    assert not torch.allclose(field(z, t, [1, 4]), plain, atol=1e-4)
