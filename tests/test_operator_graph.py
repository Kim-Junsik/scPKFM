"""The operator graph: its construction, and that it changes nothing it should not.

    python -m pytest tests/test_operator_graph.py -v
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.models import similarity
from src.models.flow import PKFMField

DRUG_CSV = "assets/drugs/tanimoto_ecfp4_2048.csv"
DRUGS = ["Alvespimycin", "Carmofur", "Cediranib", "Crizotinib", "Curcumin", "Dacinostat",
         "Danusertib", "Dasatinib", "Givinostat", "PCI-34051", "Panobinostat", "Pirarubicin",
         "SRT1720", "SRT2104", "SRT3025", "Sorafenib", "Tanespimycin"]
LATENT = 8
RANK = 3


@pytest.fixture
def toy_csv(tmp_path):
    """Four perturbations: a-b similar, c-d weakly similar, the rest background."""
    names = ["a", "b", "c", "d"]
    s = np.array([[1.0, 0.85, 0.1, 0.1],
                  [0.85, 1.0, 0.1, 0.1],
                  [0.1, 0.1, 1.0, 0.40],
                  [0.1, 0.1, 0.40, 1.0]])
    path = tmp_path / "toy.csv"
    np.savetxt(path, s, delimiter=",", header=",".join(names), comments="", fmt="%.4f")
    return str(path), names


def field(perturbations, graph=None, mode="penalty", rank=RANK, seed=0):
    overrides = [f"model.latent_dim={LATENT}", "model.composition=learned",
                 f"model.generator_rank={'null' if rank is None else rank}"]
    if graph:
        overrides += [f"model.operator_graph={graph}", f"model.operator_graph_mode={mode}"]
    config = config_module.load(overrides)
    torch.manual_seed(seed)
    built = PKFMField(config, len(perturbations), perturbations=perturbations)
    with torch.no_grad():  # zero-initialised factors would make every test trivial
        for name, parameter in built.generator.named_parameters():
            if name in ("u", "v", "b", "a"):
                parameter.normal_(0.0, 0.3)
        built.compose.rho[-1].weight.normal_(0.0, 0.2)
    return built


# ---------------------------------------------------------------- the graph itself
def test_real_drug_graph_has_the_six_preregistered_edges():
    weights = similarity.operator_graph(DRUG_CSV, 0.25, DRUGS)
    index = {d: i for i, d in enumerate(DRUGS)}
    edges = {(DRUGS[i], DRUGS[j]): round(float(weights[i, j]), 3)
             for i in range(len(DRUGS)) for j in range(i + 1, len(DRUGS)) if weights[i, j] > 0}
    assert edges == {("Alvespimycin", "Tanespimycin"): 0.794, ("SRT1720", "SRT2104"): 0.333,
                     ("Dacinostat", "Panobinostat"): 0.246, ("Givinostat", "PCI-34051"): 0.138,
                     ("SRT2104", "SRT3025"): 0.094, ("SRT1720", "SRT3025"): 0.067}
    assert np.allclose(weights, weights.T) and not np.diag(weights).any()
    # the drugs with no analogue stay isolated
    for drug in ("Crizotinib", "Curcumin", "Sorafenib", "Dasatinib"):
        assert not weights[index[drug]].any(), drug


def test_graph_follows_the_model_perturbation_order(toy_csv):
    path, names = toy_csv
    forward = similarity.operator_graph(path, 0.25, names)
    backward = similarity.operator_graph(path, 0.25, names[::-1])
    assert np.allclose(backward, forward[::-1, ::-1])


def test_a_perturbation_missing_from_the_file_is_refused(toy_csv):
    path, names = toy_csv
    with pytest.raises(ValueError, match="no similarity"):
        similarity.operator_graph(path, 0.25, names + ["e"])


def test_threshold_out_of_range_is_refused(toy_csv):
    path, names = toy_csv
    with pytest.raises(ValueError, match="threshold"):
        similarity.operator_graph(path, 1.0, names)


def test_graph_needs_names_and_an_affine_generator(toy_csv):
    path, names = toy_csv
    config = config_module.load([f"model.latent_dim={LATENT}", f"model.operator_graph={path}"])
    with pytest.raises(ValueError, match="perturbation names"):
        PKFMField(config, len(names))
    shared = config_module.load([f"model.latent_dim={LATENT}", f"model.operator_graph={path}",
                                 "model.generator=shared_basis"])
    with pytest.raises(ValueError, match="couples affine operators"):
        PKFMField(shared, len(names), perturbations=names)


# ---------------------------------------------------------------- equivalences
def _same_outputs(first, second, perturbation_sets, n=5):
    torch.manual_seed(1)
    z = torch.randn(n, LATENT)
    t = torch.full((1,), 0.3)
    for perturbations in perturbation_sets:
        assert torch.allclose(first(z, t, perturbations), second(z, t, perturbations),
                              atol=1e-6), perturbations


@pytest.mark.parametrize("rank", [None, RANK])
def test_penalty_mode_leaves_the_field_unchanged(toy_csv, rank):
    path, names = toy_csv
    plain = field(names, rank=rank)
    graphed = field(names, graph=path, mode="penalty", rank=rank)
    graphed.load_state_dict({**plain.state_dict(), "generator.graph": graphed.generator.graph})
    _same_outputs(plain, graphed, [[0], [1], [2, 3], [0, 1]])


@pytest.mark.parametrize("rank", [None, RANK])
def test_mix_with_no_edges_is_the_plain_field(toy_csv, rank):
    path, names = toy_csv
    plain = field(names, rank=rank)
    config_overrides = [f"model.latent_dim={LATENT}", "model.composition=learned",
                        f"model.generator_rank={rank}" if rank else "model.generator_rank=null",
                        f"model.operator_graph={path}", "model.operator_graph_mode=mix",
                        "model.operator_graph_threshold=0.9"]  # above every pair
    isolated = PKFMField(config_module.load(config_overrides), len(names), perturbations=names)
    assert not isolated.generator.graph.any()
    isolated.load_state_dict({**plain.state_dict(), "generator.graph": isolated.generator.graph})
    _same_outputs(plain, isolated, [[0], [1], [2, 3], [0, 1]])


def test_mix_is_the_weighted_mean_of_own_operators(toy_csv):
    path, names = toy_csv
    mixed = field(names, graph=path, mode="mix")
    gen = mixed.generator
    w = float(gen.graph[0, 1])
    assert w == pytest.approx((0.85 - 0.25) / 0.75, abs=1e-4)
    z = torch.randn(3, LATENT)
    expected = (gen.own_operator(z, 0) + w * gen.own_operator(z, 1)) / (1 + w)
    assert torch.allclose(gen.operator(z, 0), expected, atol=1e-6)
    assert torch.allclose(gen.bias(0), (gen.b[0] + w * gen.b[1]) / (1 + w), atol=1e-6)
    # matrix() is the operator the field applies
    assert torch.allclose(z @ gen.matrix(0).T, gen.operator(z, 0), atol=1e-5)
    # the mixing is linear in z: still one Koopman operator
    z2 = torch.randn(3, LATENT)
    assert torch.allclose(gen.operator(2 * z + z2, 0),
                          2 * gen.operator(z, 0) + gen.operator(z2, 0), atol=1e-5)


def test_mixed_data_trains_the_neighbour(toy_csv):
    path, names = toy_csv
    mixed = field(names, graph=path, mode="mix")
    loss = mixed(torch.randn(4, LATENT), torch.full((1,), 0.5), [0]).pow(2).sum()
    loss.backward()
    assert mixed.generator.u.grad[1].abs().sum() > 0      # neighbour b learns from a
    assert mixed.generator.u.grad[2].abs().sum() == 0     # c is not a neighbour of a


# ---------------------------------------------------------------- penalty
def test_penalty_is_zero_at_initialisation_and_for_equal_operators(toy_csv):
    path, names = toy_csv
    config = config_module.load([f"model.latent_dim={LATENT}", f"model.generator_rank={RANK}",
                                 f"model.operator_graph={path}"])
    fresh = PKFMField(config, len(names), perturbations=names)
    assert float(similarity.operator_graph_penalty(fresh.generator)) == 0.0
    trained = field(names, graph=path)
    with torch.no_grad():
        for p in (trained.generator.u, trained.generator.v, trained.generator.b):
            p[1].copy_(p[0])
            p[3].copy_(p[2])
    assert float(similarity.operator_graph_penalty(trained.generator)) == pytest.approx(0.0, abs=1e-8)


def test_penalty_pulls_a_data_free_operator_to_its_neighbour(toy_csv):
    """With no data term, minimising the penalty alone moves an operator toward its
    neighbour's: the zero-shot operator is interpolated from structure."""
    path, names = toy_csv
    model = field(names, graph=path, rank=None)
    gen = model.generator
    target = gen.own_matrix(0).detach().clone()
    optimiser = torch.optim.Adam([gen.a], lr=0.02)

    def distance():
        return float((gen.own_matrix(1) - target).norm() / target.norm())

    before = distance()
    for _ in range(300):
        optimiser.zero_grad()
        similarity.operator_graph_penalty(gen).backward()
        gen.a.grad[0] = 0  # operator a is held by its data
        optimiser.step()
    assert distance() < 0.1 * before


def test_ramp_is_off_during_warmup_then_linear():
    assert similarity.penalty_ramp(59, 60, 100) == 0.0
    assert similarity.penalty_ramp(60, 60, 100) == pytest.approx(0.01)
    assert similarity.penalty_ramp(109, 60, 100) == pytest.approx(0.5)
    assert similarity.penalty_ramp(500, 60, 100) == 1.0
    assert similarity.penalty_ramp(60, 60, 0) == 1.0


# ---------------------------------------------------------------- checkpoints
def test_graph_travels_in_the_checkpoint(toy_csv):
    path, names = toy_csv
    trained = field(names, graph=path, mode="mix")
    state = copy.deepcopy(trained.state_dict())
    # rebuilt at another threshold: the loaded buffer and neighbour lists must win
    config = config_module.load([f"model.latent_dim={LATENT}", "model.composition=learned",
                                 f"model.generator_rank={RANK}", f"model.operator_graph={path}",
                                 "model.operator_graph_mode=mix",
                                 "model.operator_graph_threshold=0.9"])
    reloaded = PKFMField(config, len(names), perturbations=names)
    assert not reloaded.generator.graph.any()
    reloaded.load_state_dict(state)
    assert torch.equal(reloaded.generator.graph, trained.generator.graph)
    assert reloaded.generator.graph_edges() == trained.generator.graph_edges()
    _same_outputs(trained, reloaded, [[0], [1], [2, 3]])


def test_plain_checkpoints_still_load():
    names = ["a", "b", "c"]
    plain = field(names)
    assert "generator.graph" not in plain.state_dict()
    again = field(names, seed=5)
    again.load_state_dict(plain.state_dict())
    _same_outputs(plain, again, [[0], [1, 2]])
