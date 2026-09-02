"""Interchangeable representation backbones.

The paper's contribution is the latent dynamics, not the encoder. Showing that the
same LieCFM field works on top of several standard representations is therefore a
strength rather than a distraction: it says the result is a property of the
dynamics, not of one lucky architecture. All three satisfy one interface, so
src/train and src/eval never learn which is loaded.

    mlp          plain MLP VAE. Cheap, and the current reference point.
    transformer  Perceiver-style: gene embeddings are compressed to a small set of
                 latent tokens by cross-attention, then self-attention runs over
                 THOSE. Full gene-to-gene self-attention is O(G^2) - 9.4 M pairs
                 per cell at G=3,074 and 371 M at the full 19,264 - which does not
                 fit the 8.6 GB budget at any useful batch size.
    scvi         MLP encoder with an explicit library-size head and a ZINB decoder,
                 i.e. the standard scRNA-seq generative model. Usable because raw
                 counts are exactly recoverable (src/data/counts.py).

Interface:
    encode(x)                     -> mu, logvar
    decode(z)                     -> head parameter dict
    reconstruction(params, ...)   -> point estimate in log1p space
    loss(params, x, ...)          -> scalar, parts
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .heads import build_head


def mlp_stack(sizes: list[int], dropout: float, final_activation: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if final_activation or i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class BaseBackbone(nn.Module):
    """Shared plumbing: reparameterisation, decoding and the loss dispatch."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        # Flow matching is run on a STANDARDISED latent. Measured on a stage-1
        # checkpoint, the raw latent collapses to ||std|| = 0.0146: conditions are
        # still separated there (centre shift / spread = 1.3), but the target
        # velocity z1 - z0 is then ~1e-2 and the matching loss ~1e-5, so the
        # gradient signal sits at the level of numerical noise. Rescaling is a
        # pure reparameterisation - encode_z and decode_z invert each other - so
        # it changes conditioning without changing the model class.
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

    @torch.no_grad()
    def fit_latent_scale(self, x: torch.Tensor, chunk: int = 512) -> tuple[float, float]:
        """Standardisation statistics over `x`, encoded in chunks.

        The caller hands this 8,192 cells while training runs at batch 256, so an
        unchunked forward is the largest single allocation in the whole run by a
        factor of 32 - it is what ran the gpu out of memory on pcab at 5,000
        genes, before stage 2 had taken a single step.

        Chunking changes nothing numerically: only the latents are kept, and
        8,192 x 64 floats is 2 MB, so mean and std are still taken over every row
        at once after the concatenation.
        """
        was_training = self.training
        self.eval()
        mu = torch.cat([self.encode(x[start:start + chunk])[0]
                        for start in range(0, x.shape[0], chunk)], dim=0)
        self.latent_mean.copy_(mu.mean(dim=0))
        self.latent_std.copy_(mu.std(dim=0).clamp(min=1e-6))
        self.train(was_training)
        return float(mu.std(dim=0).norm()), float(self.latent_std.mean())

    def encode_z(self, x: torch.Tensor):
        """Standardised latent - the space the velocity field lives in."""
        mu, logvar = self.encode(x)
        return (mu - self.latent_mean) / self.latent_std, logvar

    def decode_z(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decode(z * self.latent_std + self.latent_mean)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.decoder(z))

    def reconstruction(self, params: dict, **aux) -> torch.Tensor:
        return self.head.point_estimate(params, **aux)

    def loss(self, params: dict, x: torch.Tensor, **aux):
        return self.head.loss(params, x, **aux)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterise(mu, logvar)), mu, logvar


def build_backbone(config: dict, n_genes: int, gene_names=None) -> BaseBackbone:
    """`gene_names` is only needed by pcab, which has to align its K x G mask to
    the gene axis. Column order is adata.var_names as given - that single
    convention replaces the old tokenizer/vocab alignment, and a mismatch here
    would silently misalign every mask row against the wrong genes."""
    kind = config["model"]["backbone"]
    if kind != "pcab":
        raise ValueError(
            f"backbone {kind!r} is not in this build. Only P-CAB/E-RCA is kept "
            "here; the mlp, transformer and scvi backbones live in the original "
            "full-axis repository this was pruned from.")
    if gene_names is None:
        raise ValueError("backbone=pcab needs gene_names to build the mask")
    if len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names has {len(gene_names)} entries but n_genes is {n_genes}")
    from .pcab import PCABBackbone  # local: pcab imports from this module
    from ..data.kegg import build_prior
    prior, _ = build_prior(config, np.asarray(gene_names))
    return PCABBackbone(config, n_genes, prior)
