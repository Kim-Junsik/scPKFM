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
                 rank: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank
        self.time = TimeEmbedding(time_embed_dim)
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
        """
        if self.rank is None:
            return z @ self.a[pert].T
        return (z @ self.v[pert].T) @ self.u[pert].T

    def forward(self, z: torch.Tensor, t: torch.Tensor, pert: int) -> torch.Tensor:
        # `time_scale` IS an MLP, and it is the one thing here that could break
        # the linearity claim - so note what it takes: TIME ONLY. It never sees z,
        # and it returns a SCALAR. For any fixed t the map z -> u is still
        # z |-> s(t)(A_a z + b_a), linear in z. The trajectory is therefore the
        # autonomous linear system's, traversed at a varying speed; feeding z into
        # this MLP would make the field nonlinear and void the Koopman reading.
        scale = 1.0 + self.time_scale(self.time(t))
        return scale * (self.operator(z, pert) + self.b[pert])

    def matrix(self, pert: int) -> torch.Tensor:
        """A_a as a dense [D, D] matrix, whichever way it is stored.

        At latent_readout=pathway this is the figure: entry [i, j] is how much
        pathway j drives pathway i under perturbation a.
        """
        if self.rank is None:
            return self.a[pert]
        return self.u[pert] @ self.v[pert]


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
                    latent_dim: int | None = None) -> nn.Module:
    """`latent_dim` overrides the config value.

    latent_readout=pathway makes the encoder's latent width K rather than
    model.latent_dim, and the field has to match the encoder it is paired with,
    so callers pass vae.latent_dim rather than trusting the config.
    """
    model_cfg = config["model"]
    width = model_cfg["latent_dim"] if latent_dim is None else latent_dim
    kind = model_cfg["generator"]
    if kind == "affine":
        return AffineGenerator(n_perturbations, width, model_cfg["time_embed_dim"],
                               model_cfg.get("generator_rank"))
    if kind == "neural_field":
        if model_cfg.get("generator_rank") is not None:
            raise ValueError(
                "generator_rank factorises the affine operator and has no meaning "
                "for neural_field; set it to null.")
        return NeuralFieldGenerator(n_perturbations, width,
                                    model_cfg["generator_hidden"],
                                    model_cfg["time_embed_dim"])
    raise ValueError(f"unknown generator {kind!r}")
