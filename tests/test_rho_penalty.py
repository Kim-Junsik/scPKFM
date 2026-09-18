"""The composition-magnitude penalty and the rho-off scoring switch.

    python -m pytest tests/test_rho_penalty.py -v
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.models.flow import PKFMField
from src.train.loop import rho_penalty

LATENT = 8
N = 5


def field(composition="learned", seed=0):
    config = config_module.load([f"model.latent_dim={LATENT}", f"model.composition={composition}",
                                 "model.generator_rank=3"])
    torch.manual_seed(seed)
    built = PKFMField(config, N)
    with torch.no_grad():
        for p in (built.generator.u, built.generator.v, built.generator.b):
            p.normal_(0.0, 0.3)
    return built


def batch():
    torch.manual_seed(1)
    return torch.randn(6, LATENT), torch.rand(6), torch.randn(6, LATENT)


def test_penalty_is_zero_while_rho_is_the_zero_initialised_map():
    z, t, target = batch()
    assert float(rho_penalty(field(), z, t, [0, 1], target)) == 0.0


def test_penalty_is_rhos_share_of_the_needed_velocity():
    model = field()
    with torch.no_grad():
        model.compose.rho[-1].weight.normal_(0.0, 0.3)
    z, t, target = batch()
    velocities = [model.generator(z, t, p) for p in (0, 1)]
    correction = model.compose(velocities)
    expected = correction.pow(2).sum(1).mean() / target.pow(2).sum(1).mean()
    assert float(rho_penalty(model, z, t, [0, 1], target)) == pytest.approx(float(expected),
                                                                          rel=1e-5)


def test_penalty_stays_bounded_when_the_generators_are_zero():
    """Right after the singles warm-up a combination-only drug's operator is ~0.
    Dividing by |sum u|^2 made the term 9e5 in a smoke run; the data-fixed
    denominator keeps it at rho's size relative to the needed displacement."""
    model = field()
    with torch.no_grad():
        for p in (model.generator.u, model.generator.b):
            p.zero_()
        model.compose.rho[-1].bias.normal_(0.0, 0.3)
    z, t, target = batch()
    value = float(rho_penalty(model, z, t, [0, 1], target))
    assert 0.0 < value < 10.0


def test_penalty_trains_rho_only():
    """The generators' velocities enter detached: the penalty shrinks rho and leaves
    the sharing-out of the fit to flow matching."""
    model = field()
    with torch.no_grad():
        model.compose.rho[-1].weight.normal_(0.0, 0.3)
    z, t, target = batch()
    rho_penalty(model, z, t, [0, 1], target).backward()
    assert model.compose.rho[-1].weight.grad.abs().sum() > 0
    assert model.compose.phi[0].weight.grad.abs().sum() > 0
    for p in (model.generator.u, model.generator.v, model.generator.b):
        assert p.grad is None or p.grad.abs().sum() == 0


def test_switching_composition_off_leaves_singles_and_drops_rho():
    model = field()
    with torch.no_grad():
        model.compose.rho[-1].weight.normal_(0.0, 0.3)
    z, t, _ = batch()
    single_before = model(z, t[:1], [2])
    pair_before = model(z, t[:1], [0, 1])
    model.composition_kind = "additive"  # what dev_score.score_run(rho_off=True) does
    assert torch.allclose(model(z, t[:1], [2]), single_before)
    additive = model.generator(z, t[:1], 0) + model.generator(z, t[:1], 1)
    assert torch.allclose(model(z, t[:1], [0, 1]), additive)
    assert not torch.allclose(pair_before, additive)
