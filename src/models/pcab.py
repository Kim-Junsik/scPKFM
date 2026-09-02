"""P-CAB encoder and E-RCA decoder (stage 3).

P-CAB replaces the MLP encoder with K learned pathway tokens that cross-attend to
genes. Full gene-to-gene self-attention is O(G^2) - 9.4 M pairs per cell at
G=3,074 and 371 M at the full 19,264 - so this is not a Transformer and should not
be described as one; the cost is O(G*K).

The mask carries the biological prior:

    M_total = act( M_prior + alpha * M_residual )        in (-1, 1)^{K x G}

and is applied as a MULTIPLICATIVE GATE, not as a logit bias:

    A     = softmax_g( Q K^T / sqrt(d_k) )      non-negative, rows sum to 1
    H     = ( A (*) M_total ) V

The distinction matters. Added to a softmax logit, a negative entry only means
"attend less" - the value aggregation stays a convex combination of V, so
repression is not in the representable function space at all. As a multiplicative
gate, M_total is a SIGNED connection strength and a negative entry genuinely
subtracts. That is the whole reason tanh replaced the original softmax mask.

It also removes the need for a scale on the mask: as a logit bias the prior had to
overcome the sheer count of unconnected genes (a 50-gene pathway against 3,024
others needs lambda ~ 8 before it bites), whereas a multiplicative gate kills them
by multiplying by ~0.

Measured for this gene space: K = 171 annotated pathways survive the filters at
0.87 % density, and 53 of the 101 perturbation targets sit in NO usable pathway -
the developmental TF block KEGG does not catalogue. `n_free_tokens` rows with an
all-zero prior are appended for those; there the mask reduces to act(alpha *
M_residual) and the model builds its own tokens.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import BaseBackbone, mlp_stack
from .heads import HurdleHead


class PathwayMask(nn.Module):
    """M_total = act(M_prior + alpha * M_residual), shared by encoder and decoder.

    Sharing is not just parameter thrift. For a repressive edge the decoder says
    "pathway up -> gene down" and the encoder says "gene observed low -> pathway
    active", and both are the same negative weight; separate masks could learn
    contradictory signs for one edge.
    """

    def __init__(self, prior: np.ndarray, alpha: float, activation: str,
                 mode: str, self_loop: bool):
        super().__init__()
        self.register_buffer("prior", torch.as_tensor(prior, dtype=torch.float32))
        self.residual = nn.Parameter(torch.zeros_like(self.prior))
        self.alpha = alpha
        self.activation = activation
        self.mode = mode
        self.self_loop = self_loop

    def forward(self) -> torch.Tensor:
        if self.mode == "prior_only":
            raw = self.prior
        elif self.mode == "residual_only":
            raw = self.alpha * self.residual
        elif self.mode == "hybrid":
            raw = self.prior + self.alpha * self.residual
        else:
            raise ValueError(f"unknown mask_mode {self.mode!r}")

        if self.activation == "tanh":
            return torch.tanh(raw)
        if self.activation == "sigmoid":
            # Unsigned control arm: isolates how much the SIGN contributes.
            return torch.sigmoid(raw)
        raise ValueError(f"unknown mask_activation {self.activation!r}")

    def l1(self) -> torch.Tensor:
        return self.residual.abs().mean()

    def learned_edges(self) -> torch.Tensor:
        """M_total minus the activated prior: edges the data added or removed.

        This is the interpretability figure - the model returns a corrected
        annotation rather than only consuming one.
        """
        with torch.no_grad():
            activated = (torch.tanh(self.prior) if self.activation == "tanh"
                         else torch.sigmoid(self.prior))
            return self.forward() - activated


class GeneWiseHurdleHead(HurdleHead):
    """Same as HurdleHead but reading a [B, G, d_v] tensor.

    A Flatten + Linear output layer would be G * K * d_v = 23.7 billion parameters
    at the full gene set. A shared direction plus a per-gene bias is ~19 K, and
    loses nothing: gene identity already lives in h, not in the projection.
    """

    def __init__(self, d_value: int, n_genes: int, bce_weight: float,
                 gate_mode: str, magnitude_mode: str):
        nn.Module.__init__(self)
        self.bce_weight = bce_weight
        self.gate_mode = gate_mode
        self.magnitude_mode = magnitude_mode
        self.gate_w = nn.Parameter(torch.randn(d_value) * 0.02)
        self.gate_b = nn.Parameter(torch.zeros(n_genes))
        self.magnitude_w = nn.Parameter(torch.randn(d_value) * 0.02)
        self.magnitude_b = nn.Parameter(torch.zeros(n_genes))
        if magnitude_mode == "gaussian":
            self.scale_w = nn.Parameter(torch.randn(d_value) * 0.02)
            self.scale_b = nn.Parameter(torch.zeros(n_genes))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        params = {
            "gate_logit": torch.einsum("bgd,d->bg", h, self.gate_w) + self.gate_b,
            "magnitude": F.softplus(
                torch.einsum("bgd,d->bg", h, self.magnitude_w) + self.magnitude_b),
        }
        if self.magnitude_mode == "gaussian":
            params["log_scale"] = (
                torch.einsum("bgd,d->bg", h, self.scale_w) + self.scale_b).clamp(-6.0, 2.0)
        return params


class PCABBackbone(BaseBackbone):
    def __init__(self, config: dict, n_genes: int, prior: np.ndarray):
        model_cfg = config["model"]
        n_tokens = prior.shape[0]
        # pathway readout makes the latent one scalar PER TOKEN, so its width is
        # K and not model.latent_dim. Everything downstream reads self.latent_dim
        # rather than the config for exactly this reason.
        self.readout = model_cfg.get("latent_readout", "dense")
        if self.readout not in ("dense", "pathway"):
            raise ValueError(f"unknown latent_readout {self.readout!r}")
        super().__init__(n_tokens if self.readout == "pathway"
                         else model_cfg["latent_dim"])
        d_key, d_value = model_cfg["d_key"], model_cfg["d_value"]
        self.n_tokens, self.d_key, self.n_genes = n_tokens, d_key, n_genes
        self.combine = model_cfg["mask_combine"]

        self.mask = PathwayMask(prior, model_cfg["mask_alpha"],
                                model_cfg["mask_activation"], model_cfg["mask_mode"],
                                model_cfg["mask_self_loop"])
        if not model_cfg["mask_share_enc_dec"]:
            self.decoder_mask = PathwayMask(prior, model_cfg["mask_alpha"],
                                            model_cfg["mask_activation"],
                                            model_cfg["mask_mode"],
                                            model_cfg["mask_self_loop"])

        self.queries = nn.Parameter(torch.randn(n_tokens, d_key) * 0.02)
        self.gene_key = nn.Parameter(torch.randn(n_genes, d_key) * 0.02)
        self.gene_value = nn.Parameter(torch.randn(n_genes, d_value) * 0.02)
        self.encoder_norm = nn.LayerNorm(d_value)
        if self.readout == "pathway":
            # One shared direction plus a per-token bias, the same shape of
            # readout the gene-wise head uses. A dense K*d_v -> K layer would
            # re-mix the tokens and undo the point of this mode.
            self.mu_w = nn.Parameter(torch.randn(d_value) * 0.02)
            self.mu_b = nn.Parameter(torch.zeros(n_tokens))
            self.logvar_w = nn.Parameter(torch.randn(d_value) * 0.02)
            self.logvar_b = nn.Parameter(torch.zeros(n_tokens))
        else:
            self.to_mu = nn.Linear(n_tokens * d_value, self.latent_dim)
            self.to_logvar = nn.Linear(n_tokens * d_value, self.latent_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))  # only used by logit_bias

        # --- decoder: latent -> pathway token values -> gene-wise readout ---
        # One SHARED small MLP evaluated per token, conditioned on a token
        # embedding. Going latent -> (n_tokens * d_value) with a dense layer is
        # the same blow-up the output layer had: 631 M parameters at K=277,
        # d_v=64. This is ~30 K and expresses the same thing.
        self.token_embedding = nn.Parameter(torch.randn(n_tokens, d_key) * 0.02)
        # pathway mode hands each token its OWN latent scalar, so the per-token
        # input is 1 + d_key wide rather than latent_dim + d_key.
        per_token_in = (1 if self.readout == "pathway" else self.latent_dim) + d_key
        self.from_latent = mlp_stack([per_token_in, 2 * d_value, d_value],
                                     model_cfg["dropout"], final_activation=True)
        self.gene_query = nn.Parameter(torch.randn(n_genes, d_key) * 0.02)
        self.token_key = nn.Parameter(torch.randn(n_tokens, d_key) * 0.02)
        self.decoder_norm = nn.LayerNorm(d_value)

        head_kind = model_cfg["decoder_head"]
        if head_kind == "hurdle":
            self.head = GeneWiseHurdleHead(d_value, n_genes, model_cfg["hurdle_bce_weight"],
                                           model_cfg["hurdle_gate"],
                                           model_cfg["hurdle_magnitude"])
        else:
            raise ValueError(f"decoder_head {head_kind!r} is not supported by pcab")

    def encode(self, x: torch.Tensor):
        """Cell-specific attention over genes, gated by the pathway mask.

        Keys and values are the gene embeddings scaled by THIS cell's expression,
        which is what makes the attention pattern cell-specific; the mask alone
        would hand every cell the same pattern, which is what the masked-linear
        -decoder papers do.

        Two algebraic simplifications keep it affordable. Because the scaling is a
        per-gene scalar,

            score[b,k,g] = (Q gene_key^T)[k,g] * x[b,g]

        so the [K, G] factor is batch-independent and is computed once instead of
        once per cell; and the value scaling folds into the attention weights, so
        neither [B, G, d_k] nor [B, G, d_v] is ever materialised.
        """
        base = (self.queries @ self.gene_key.T) / (self.d_key ** 0.5)   # [K, G]
        scores = base.unsqueeze(0) * x.unsqueeze(1)                     # [B, K, G]
        mask = self.mask()

        if self.combine == "gate":
            weights = torch.softmax(scores, dim=-1) * mask.unsqueeze(0)
        elif self.combine == "logit_bias":
            # Control arm: the original additive formulation, kept so the paper
            # can show what the multiplicative gate actually buys.
            weights = torch.softmax(scores + self.logit_scale * mask.unsqueeze(0), dim=-1)
        else:
            raise ValueError(f"unknown mask_combine {self.combine!r}")

        h = self.encoder_norm((weights * x.unsqueeze(1)) @ self.gene_value)
        if self.readout == "pathway":
            mu = torch.einsum("bkd,d->bk", h, self.mu_w) + self.mu_b
            logvar = torch.einsum("bkd,d->bk", h, self.logvar_w) + self.logvar_b
            return mu, logvar.clamp(-10.0, 10.0)
        flat = h.reshape(x.shape[0], -1)
        return self.to_mu(flat), self.to_logvar(flat).clamp(-10.0, 10.0)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = z.shape[0]
        embedded = self.token_embedding.unsqueeze(0).expand(batch, -1, -1)
        if self.readout == "pathway":
            # token k reads its own coordinate, mirroring the encoder
            joined = torch.cat([z.unsqueeze(-1), embedded], dim=-1)
        else:
            joined = torch.cat([
                z.unsqueeze(1).expand(batch, self.n_tokens, -1), embedded], dim=-1)
        tokens = self.from_latent(joined)
        # Gene queries against token keys do not depend on the cell, so this
        # [G, K] attention is computed once and broadcast over the batch.
        scores = (self.gene_query @ self.token_key.T) / (self.d_key ** 0.5)
        mask = (self.mask() if not hasattr(self, "decoder_mask") else self.decoder_mask())
        if self.combine == "gate":
            weights = torch.softmax(scores, dim=-1) * mask.T
        else:
            weights = torch.softmax(scores + self.logit_scale * mask.T, dim=-1)
        h = weights.unsqueeze(0) @ tokens                               # [B, G, d_v]
        return self.head(self.decoder_norm(h))

    def mask_penalty(self) -> torch.Tensor:
        penalty = self.mask.l1()
        if hasattr(self, "decoder_mask"):
            penalty = penalty + self.decoder_mask.l1()
        return penalty
