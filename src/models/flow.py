"""The composed velocity field and its integrator.

    v(z, t, S) = sum_{a in S} u_a(z, t)

Three properties hold by construction rather than by training, so each is a
numerical assertion (tests/test_structure.py) instead of a hope:

  1. control invariance      v(z, t, {}) = 0            - empty sum
  2. single supervision      v(z, t, {a}) = u_a(z,t)    - no pair term exists
  3. permutation invariance  a sum does not depend on order

WHY THERE IS NO INTERACTION TERM

The obvious design adds an explicit algebraic term for pairs - a Lie bracket
Lambda_ab [u_a, u_b], or an unconstrained MLP over the two perturbation
embeddings. Both were built and measured, over seven configurations, two
backbones, three gene spaces and six metrics. Neither helped, and in the five
settings where the model works at all (resid_R2 > 0) the plain sum won all five.

The reason is mechanical rather than a tuning failure. Integrating a
state-dependent field is a nonlinear operation, so an ADDITIVE VELOCITY FIELD
ALREADY INDUCES A NON-ADDITIVE FLOW MAP:

    Phi_ab(z) != Phi_a(z) + Phi_b(z) - z

Measured: a model with no interaction term produces a composition non-additivity
of 0.99 of the additive sum, and its residual points along the true one with
cosine 0.46. Adding the Lie bracket moved that cosine to 0.44 - slightly worse.
The explicit term is redundant with the integrator, not too weak.

Those two forms are therefore absent here. They still exist in the full-axis
repository this was pruned from, if the ablation table is ever needed.

WHAT IS STILL OPEN

`model.composition = learned` adds a correction that neither of those was:

    v(z,t,S) = sum_a u_a  +  rho( sum_a phi(u_a(z,t)) )

Both failed terms were conditioned on perturbation IDENTITY - a gate indexed by
which genes were hit, or an MLP over their embeddings. This one is conditioned on
the VELOCITIES the generators produced, and never learns anything per
perturbation. In BCH terms the additive default truncates at first order and the
Lie bracket at second; this truncates nowhere and learns the remainder.

It is untested. Three structured terms have now been added to this model and none
improved it, so treat it as a hypothesis with a switch, not as the default.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .generators import build_generator


class ValueComposition(nn.Module):
    """A learned composition law over the generators' OUTPUT VELOCITIES.

        v(z,t,S) = sum_a u_a  +  rho( sum_a phi(u_a(z,t)) )

    Composing two linear generators is BCH:

        log(e^A e^B) = A + B + 1/2 [A,B] + (higher order)

    `additive` truncates at first order. A Lie bracket term truncates at second,
    which was built and measured across seven settings without helping. This
    truncates nowhere and learns the correction instead.

    WHAT MAKES IT DIFFERENT FROM THE INTERACTION TERM THAT ALREADY LOST. That one
    read the perturbation EMBEDDINGS - it was told which genes were hit and had to
    infer the correction from identity. This reads the VELOCITIES the generators
    actually produced at this point in latent space. An unseen combination has no
    familiar pair of identities, but its velocities are computed the same way as
    any other, so the correction is defined on the same footing as in training.

    Four properties survive by construction, all asserted in tests:

      - phi is applied per perturbation and SUMMED, so the result cannot depend on
        the order of S (DeepSets)
      - no parameter is indexed by a perturbation, let alone a pair
      - rho's output layer starts at zero, so training begins exactly additive and
        any gain is attributable to this term rather than to a different init
      - the caller applies it only for |S| >= 2, so a single is still exactly its
        own generator
    """

    def __init__(self, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim))
        nn.init.zeros_(self.rho[-1].weight)
        nn.init.zeros_(self.rho[-1].bias)

    def forward(self, velocities: list[torch.Tensor]) -> torch.Tensor:
        pooled = self.phi(velocities[0])
        for velocity in velocities[1:]:
            pooled = pooled + self.phi(velocity)
        return self.rho(pooled)


class PKFMField(nn.Module):
    """scPKFM's composed field: the sum of per-perturbation generators.

    Nothing is indexed by a PAIR of perturbations. That is what makes an unseen
    combination expressible at all: there are no parameters that only a seen pair
    could have fitted. An earlier version stored a [P, P] table for the pair term
    and it trained fine while being exactly wrong - a pair never seen in training
    kept its initial value, so on all 37 evaluation combinations the term was
    numerically zero and the model was identical to this one.
    """

    def __init__(self, config: dict, n_perturbations: int,
                 latent_dim: int | None = None):
        """`latent_dim` overrides the config value.

        At latent_readout=pathway the encoder's width is the number of pathway
        tokens, not model.latent_dim, and the field must match the encoder it is
        paired with. Callers pass vae.latent_dim; leaving it None keeps the
        config value, which is correct for the dense readout.
        """
        super().__init__()
        model_cfg = config["model"]
        self.latent_dim = (model_cfg["latent_dim"] if latent_dim is None
                           else latent_dim)
        self.generator = build_generator(config, n_perturbations, self.latent_dim)

        self.composition_kind = model_cfg.get("composition", "additive")
        if self.composition_kind == "learned":
            self.compose = ValueComposition(self.latent_dim,
                                            model_cfg.get("composition_hidden", 128))
        elif self.composition_kind != "additive":
            raise ValueError(f"unknown composition {self.composition_kind!r}")

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                perturbations: list[int]) -> torch.Tensor:
        if not perturbations:  # control: structurally the zero field
            return torch.zeros_like(z)

        per_perturbation = [self.generator(z, t, pert) for pert in perturbations]
        velocity = per_perturbation[0]
        for single in per_perturbation[1:]:
            velocity = velocity + single

        # Only a genuine combination has a composition law to correct. Applying it
        # to a single would break v(z,t,{a}) = u_a, which the loss supervises
        # directly and the tests assert.
        if self.composition_kind == "learned" and len(perturbations) >= 2:
            velocity = velocity + self.compose(per_perturbation)
        return velocity


def integrate(field: PKFMField, z0: torch.Tensor, perturbations: list[int],
              n_steps: int) -> torch.Tensor:
    """RK4 from t=0 to t=1. Fixed step: the fields are smooth and an adaptive
    solver would make runs non-reproducible for no measurable accuracy gain -
    re-integrating one checkpoint at 20 and at 100 steps moved resid_R2 from
    -0.9577 to -0.9825, i.e. not at all relative to the errors being chased."""
    z = z0
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t0 = torch.full((1,), step * dt, device=z.device, dtype=z.dtype)
        half = t0 + 0.5 * dt
        full = t0 + dt
        k1 = field(z, t0, perturbations)
        k2 = field(z + 0.5 * dt * k1, half, perturbations)
        k3 = field(z + 0.5 * dt * k2, half, perturbations)
        k4 = field(z + dt * k3, full, perturbations)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z
