"""Per-perturbation generators u_a(z, t) acting on the latent space.

Two forms, selected by `model.generator`. Both fill the same slot: given a
perturbation and a latent state, return the velocity.

    affine        u_a(z,t) = s(t) * (A_a z + b_a)     - the default
    neural_field  u_a(z,t) = f([z, phi(t), e_a])      - the comparison arm

THE AFFINE GENERATOR

    u_a(z,t) = s(t) * (A_a z + b_a)

A linear (affine) ODE in the latent space: the encoder lifts cells into a 64-d
space and the dynamics advance by a per-perturbation linear operator there, which
is the Koopman / DMD-with-control structure.

BE PRECISE ABOUT THE CLAIM. This is Koopman in FORM, not a Koopman autoencoder.
Deep Koopman trains the lifting map jointly with the linear operator so that the
latent dynamics ARE linear - that is the point of those models. Here stage 1
trains the encoder for reconstruction alone and stage 2 freezes it, so the latent
is a space that reconstructs well, not one built to linearise the dynamics.
Write it as "linear (Koopman-form) dynamics on a fixed reconstruction latent;
the representation is not trained to linearise", or a reviewer will say it first.

s(t) is a scalar, so it modulates speed without bending the trajectory: the path
is that of the autonomous linear system under a time reparameterisation.

WHY IT IS THE DEFAULT. The neural_field arm below lost 6-0 across two backbones
and three interaction settings. The axis that mattered was not depth but
PER-PERTURBATION CAPACITY: affine gives each perturbation its own 64x64 operator
(4,160 parameters) where the shared trunk gives it a 32-dim embedding and makes
101 perturbations share one function. The cost is 3.5x more parameters in the
field (421 K against 120 K), which belongs in the paper's table - though at 6 % of
the 5.2 M encoder it is not where the model's capacity lives.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Sinusoidal features so the field can vary along the transport path."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("time_embed_dim must be even")
        self.dim = dim
        half = dim // 2
        self.register_buffer("frequencies", torch.exp(
            torch.linspace(0.0, 6.0, half)), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.reshape(-1, 1) * self.frequencies.reshape(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class AffineGenerator(nn.Module):
    """u_a(z,t) = A_a(t) z + b_a(t), with the time dependence factored out.

    `rank` factorises A_a = U_a V_a instead of storing it whole. That matters at
    latent_readout=pathway, where the operator is [K, K] with K around 272 -
    74,000 parameters per perturbation against 4,160 at latent_dim=64.
    """

    def __init__(self, n_perturbations: int, latent_dim: int, time_embed_dim: int,
                 rank: int | None = None, graph=None, graph_mode: str | None = None):
        """`graph` is an optional [P, P] perturbation similarity graph (see
        src/models/similarity.py). With graph_mode="mix" every operator and bias is
        the weighted mean of the perturbation's own and its neighbours'; with
        "penalty" the field is unchanged and the graph is only read by the loss.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank
        self.time = TimeEmbedding(time_embed_dim)
        if graph is None:
            self.register_buffer("graph", None)
            self.graph_mode = None
        else:
            from .similarity import GRAPH_MODES
            if graph_mode not in GRAPH_MODES:
                raise ValueError(f"operator_graph_mode must be one of {GRAPH_MODES}, "
                                 f"got {graph_mode!r}")
            graph = torch.as_tensor(graph, dtype=torch.float32)
            if graph.shape != (n_perturbations, n_perturbations):
                raise ValueError(f"operator graph is {tuple(graph.shape)}, expected "
                                 f"({n_perturbations}, {n_perturbations})")
            # A buffer, so the graph a run trained with travels in its checkpoint.
            self.register_buffer("graph", graph)
            self.graph_mode = graph_mode
            self._cache_neighbours()
            # Loading a checkpoint replaces the buffer; the neighbour lists follow.
            self.register_load_state_dict_post_hook(
                lambda module, incompatible: module._cache_neighbours())
        # Start at zero: at initialisation every generator is the null field, so
        # the model begins as "predict no change" rather than as noise.
        if rank is None:
            # A_a, the Koopman generator: ONE [D, D] operator per perturbation,
            # applied as a matrix product in operator() below. It is a plain
            # nn.Parameter, which is why this reads like a network weight - the
            # thing that makes it Koopman rather than a layer is that nothing
            # nonlinear ever touches z.
            self.a = nn.Parameter(torch.zeros(n_perturbations, latent_dim, latent_dim))
        else:
            # A_a = U_a V_a still starts at zero, but only ONE factor may be zero.
            # With both at zero the product's gradient vanishes for both and the
            # operator never leaves the origin; zeroing U alone leaves dL/dU
            # proportional to V, so U moves first and V follows - the same
            # argument as a zero-initialised output layer.
            self.u = nn.Parameter(torch.zeros(n_perturbations, latent_dim, rank))
            self.v = nn.Parameter(torch.randn(n_perturbations, rank, latent_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_perturbations, latent_dim))
        self.time_scale = nn.Sequential(nn.Linear(time_embed_dim, 32), nn.GELU(),
                                        nn.Linear(32, 1))

    def operator(self, z: torch.Tensor, pert: int) -> torch.Tensor:
        """A_a z. THIS IS THE KOOPMAN PART, and it is one matrix product.

        Easy to mistake for a neural network layer, because `self.a` is an
        nn.Parameter like any weight. The difference is what is NOT here: no
        activation, no second layer, no bias applied elementwise through a
        nonlinearity. z enters once, linearly, and leaves.

        That is the whole claim. `dz/dt = A_a z + b_a` is a LINEAR ODE, so its
        flow map is a matrix exponential,

            Phi_a^t(z) = exp(t A_a) z + (integral of the constant term),

        and A_a is a finite-dimensional Koopman generator for perturbation a. A
        network with an activation in this path would give a general nonlinear
        field, the flow would have no closed form, and none of that would hold -
        that is exactly what NeuralFieldGenerator below is.

        The low-rank branch factorises A_a = U_a V_a. Still one linear map; only
        its storage changed.

        With a mixing graph, A_a = (A_a_own + sum_b w_ab A_b_own) / (1 + sum_b w_ab):
        a convex combination of linear maps, so still one linear map.
        """
        if self.graph_mode != "mix" or not self._neighbours[pert]:
            return self.own_operator(z, pert)
        out, total = self.own_operator(z, pert), 1.0
        for other, weight in self._neighbours[pert]:
            out = out + weight * self.own_operator(z, other)
            total += weight
        return out / total

    def own_operator(self, z: torch.Tensor, pert: int) -> torch.Tensor:
        """A_a z from perturbation a's own parameters, ignoring any graph."""
        if self.rank is None:
            return z @ self.a[pert].T
        return (z @ self.v[pert].T) @ self.u[pert].T

    def bias(self, pert: int) -> torch.Tensor:
        """b_a, mixed over the graph exactly like the operator."""
        if self.graph_mode != "mix" or not self._neighbours[pert]:
            return self.b[pert]
        out, total = self.b[pert], 1.0
        for other, weight in self._neighbours[pert]:
            out = out + weight * self.b[other]
            total += weight
        return out / total

    def _cache_neighbours(self) -> None:
        weights = self.graph.detach().cpu()
        self._neighbours = [[(int(j), float(weights[i, j]))
                             for j in torch.nonzero(weights[i]).flatten().tolist()]
                            for i in range(weights.shape[0])]

    def graph_edges(self) -> list[tuple[int, int, float]]:
        """(a, b, w) for every edge, each pair once. Empty without a graph."""
        if self.graph is None:
            return []
        return [(a, b, w) for a, pairs in enumerate(self._neighbours)
                for b, w in pairs if b > a]

    def forward(self, z: torch.Tensor, t: torch.Tensor, pert: int) -> torch.Tensor:
        # `time_scale` IS an MLP, and it is the one thing here that could break
        # the linearity claim - so note what it takes: TIME ONLY. It never sees z,
        # and it returns a SCALAR. For any fixed t the map z -> u is still
        # z |-> s(t)(A_a z + b_a), linear in z. The trajectory is therefore the
        # autonomous linear system's, traversed at a varying speed; feeding z into
        # this MLP would make the field nonlinear and void the Koopman reading.
        scale = 1.0 + self.time_scale(self.time(t))
        return scale * (self.operator(z, pert) + self.bias(pert))

    def matrix(self, pert: int) -> torch.Tensor:
        """A_a as a dense [D, D] matrix, whichever way it is stored - the operator
        the field actually applies, so mixed when the graph mixes.

        At latent_readout=pathway this is the figure: entry [i, j] is how much
        pathway j drives pathway i under perturbation a.
        """
        if self.graph_mode != "mix" or not self._neighbours[pert]:
            return self.own_matrix(pert)
        out, total = self.own_matrix(pert), 1.0
        for other, weight in self._neighbours[pert]:
            out = out + weight * self.own_matrix(other)
            total += weight
        return out / total

    def own_matrix(self, pert: int) -> torch.Tensor:
        """Perturbation a's own dense operator, ignoring any graph."""
        if self.rank is None:
            return self.a[pert]
        return self.u[pert] @ self.v[pert]


class SharedBasisGenerator(nn.Module):
    """u_a(z,t) = s(t) * (U diag(c_a) V z + P_a Q_a z + b_a).

    A Koopman operator per perturbation, built mostly from modes every
    perturbation shares, plus a small private part.

    WHY. On combosciplex the transport error is almost entirely generalisation
    (scripts/diagnose_train_gap.py): training conditions sit 0.20-0.26 L2 above
    the decoder's ceiling, held-out ones 1.10-1.32. Every test combination pairs
    Panobinostat with a drug seen in ONE training condition, and AffineGenerator
    hands that drug a dedicated rank-64 operator - 33,282 parameters fitted from
    one condition. Here most of each operator comes from shared modes and a
    perturbation only chooses how much of each mode it uses.

    STILL KOOPMAN. The field is linear in z for every perturbation, so
    A_a = U diag(c_a) V + P_a Q_a is one generator and the flow map is its matrix
    exponential, as for AffineGenerator. Composition stays in the same algebra:
    the shared parts of a and b add as U diag(c_a + c_b) V, so an unseen pair is
    built from coefficients each drug learned in whatever conditions it appeared.

    CAPACITY. m + 2*K*p + K parameters per perturbation, plus 2*K*m shared. At
    K=258, m=64, p=8 that is 4,450 per drug against 33,282; on Norman (K~312),
    5,368 against 40,248 - still above the 4,160 per perturbation of the affine
    generator that beat the shared neural field 6-0, so the capacity that won
    there is not given up.

    m=0 with p=r is AffineGenerator(rank=r) term for term, which
    tests/test_structure.py asserts: the current model is the special case.

    INITIALISATION. U and P_a start at zero, so the field is the null field at
    step 0 like every other generator here. V and Q_a start small and random so
    the products still pass gradient (see AffineGenerator). c_a starts at one:
    every perturbation begins using every mode equally, so the basis first learns
    the response the perturbations have in common and the coefficients
    differentiate from there. U diag(c) V is invariant to U -> kU, c -> c/k, so
    read A_a as a whole rather than U or c_a alone.
    """

    def __init__(self, n_perturbations: int, latent_dim: int, time_embed_dim: int,
                 shared_rank: int, private_rank: int):
        super().__init__()
        if shared_rank < 0 or private_rank < 0 or shared_rank + private_rank == 0:
            raise ValueError(
                f"shared_basis needs shared_rank + private_rank > 0, both >= 0 "
                f"(got shared_rank={shared_rank}, private_rank={private_rank})")
        self.latent_dim = latent_dim
        self.shared_rank = shared_rank
        self.private_rank = private_rank
        self.time = TimeEmbedding(time_embed_dim)
        if shared_rank:
            self.basis_u = nn.Parameter(torch.zeros(latent_dim, shared_rank))
            self.basis_v = nn.Parameter(torch.randn(shared_rank, latent_dim) * 0.02)
            self.coef = nn.Parameter(torch.ones(n_perturbations, shared_rank))
        if private_rank:
            self.private_u = nn.Parameter(
                torch.zeros(n_perturbations, latent_dim, private_rank))
            self.private_v = nn.Parameter(
                torch.randn(n_perturbations, private_rank, latent_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_perturbations, latent_dim))
        self.time_scale = nn.Sequential(nn.Linear(time_embed_dim, 32), nn.GELU(),
                                        nn.Linear(32, 1))

    def operator(self, z: torch.Tensor, pert: int) -> torch.Tensor:
        """A_a z, computed through the factors without forming A_a."""
        out = torch.zeros_like(z)
        if self.shared_rank:
            out = out + ((z @ self.basis_v.T) * self.coef[pert]) @ self.basis_u.T
        if self.private_rank:
            out = out + (z @ self.private_v[pert].T) @ self.private_u[pert].T
        return out

    def forward(self, z: torch.Tensor, t: torch.Tensor, pert: int) -> torch.Tensor:
        # time_scale sees time only and returns a scalar, so for fixed t the map
        # z -> u is still linear - the same argument as AffineGenerator.forward.
        scale = 1.0 + self.time_scale(self.time(t))
        return scale * (self.operator(z, pert) + self.b[pert])

    def matrix(self, pert: int) -> torch.Tensor:
        """A_a as a dense [D, D] matrix: the pathway interaction figure."""
        full = torch.zeros(self.latent_dim, self.latent_dim,
                           device=self.b.device, dtype=self.b.dtype)
        if self.shared_rank:
            full = full + (self.basis_u * self.coef[pert]) @ self.basis_v
        if self.private_rank:
            full = full + self.private_u[pert] @ self.private_v[pert]
        return full


class NeuralFieldGenerator(nn.Module):
    """u_a(z,t) = f([z, phi(t), e_a]) with f SHARED and e_a learned per perturbation.

    The comparison arm for the Koopman generator, and the axis it isolates is not
    depth but per-perturbation capacity: every perturbation goes through the same
    trunk and is distinguished only by a 32-dim embedding, where AffineGenerator
    hands each one its own operator. At latent_dim=64 that is 32 parameters per
    perturbation against 4,160 - a factor of 130.

    Measured on fold 0 with everything else held fixed, affine won both backbones:

        mlp    affine  0.3348   neural_field -0.4377   (+0.77)
        pcab   affine  0.1965   neural_field -0.1346   (+0.33)

    Note the two arms differ in TWO ways at once - dedicated vs shared capacity,
    and linear vs nonlinear form - so the attribution is not isolated. A third arm
    (a dedicated MLP per perturbation) would separate them and has never been
    built. The reading that capacity is what mattered rests on the more expressive
    arm being the one that lost.
    """

    def __init__(self, n_perturbations: int, latent_dim: int, hidden: list[int],
                 time_embed_dim: int, embed_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.time = TimeEmbedding(time_embed_dim)
        self.embedding = nn.Embedding(n_perturbations, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        sizes = [latent_dim + time_embed_dim + embed_dim, *hidden]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(sizes[-1], latent_dim)
        # Zero-initialised head, so this arm also starts as exactly the null field
        # and the two generators are comparable from step one.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor, pert: int) -> torch.Tensor:
        batch = z.shape[0]
        time_features = self.time(t).expand(batch, -1)
        embedding = self.embedding.weight[pert].reshape(1, -1).expand(batch, -1)
        return self.head(self.trunk(torch.cat([z, time_features, embedding], dim=-1)))


def build_generator(config: dict, n_perturbations: int,
                    latent_dim: int | None = None,
                    perturbations: list[str] | None = None) -> nn.Module:
    """`latent_dim` overrides the config value.

    latent_readout=pathway makes the encoder's latent width K rather than
    model.latent_dim, and the field has to match the encoder it is paired with,
    so callers pass vae.latent_dim rather than trusting the config.

    `perturbations` are the names in index order (data.perturbations). They are
    required when model.operator_graph is set, since the graph is keyed by name.
    """
    model_cfg = config["model"]
    width = model_cfg["latent_dim"] if latent_dim is None else latent_dim
    kind = model_cfg["generator"]
    graph_path = model_cfg.get("operator_graph")
    if graph_path and kind != "affine":
        raise ValueError(f"model.operator_graph couples affine operators; generator "
                         f"{kind!r} has none")
    if kind == "affine":
        graph = None
        if graph_path:
            if perturbations is None or len(perturbations) != n_perturbations:
                raise ValueError("model.operator_graph needs the perturbation names in "
                                 "index order; pass data.perturbations")
            from .similarity import operator_graph
            graph = operator_graph(graph_path, model_cfg.get("operator_graph_threshold", 0.25),
                                   list(perturbations))
        return AffineGenerator(n_perturbations, width, model_cfg["time_embed_dim"],
                               model_cfg.get("generator_rank"), graph,
                               model_cfg.get("operator_graph_mode", "penalty") if graph_path
                               else None)
    if kind == "shared_basis":
        if model_cfg.get("generator_rank") is not None:
            raise ValueError(
                "generator_rank sizes the affine generator; shared_basis is sized by "
                "model.shared_rank and model.private_rank. Set generator_rank to null "
                "(drop --generator-rank).")
        return SharedBasisGenerator(n_perturbations, width, model_cfg["time_embed_dim"],
                                    model_cfg["shared_rank"], model_cfg["private_rank"])
    if kind == "neural_field":
        if model_cfg.get("generator_rank") is not None:
            raise ValueError(
                "generator_rank factorises the affine operator and has no meaning "
                "for neural_field; set it to null.")
        return NeuralFieldGenerator(n_perturbations, width,
                                    model_cfg["generator_hidden"],
                                    model_cfg["time_embed_dim"])
    raise ValueError(f"unknown generator {kind!r}")
