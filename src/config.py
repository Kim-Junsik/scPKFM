"""Central configuration.

Every value an experiment might vary lives here, never as a literal inside the
code. Runs record the full resolved config next to their results, and any key can
be overridden from the command line with dot notation:

    python scripts/build_data.py --set data.n_hvg=5000 data.hvg_criterion=dispersion
"""

from __future__ import annotations

import copy
import json
from typing import Any

DEFAULTS: dict[str, Any] = {
    "data": {
        "raw_h5ad": "data/norman/norman.h5ad",
        "kegg_dir": "assets/kegg",
        "pathway_min_genes": 10,
        "pathway_max_genes": 300,
        "drop_disease_pathways": True,
        "cache_h5ad": "assets/norman_modeled.h5ad",
        # X in the source file is ALREADY log1p-normalised (values 0.305-6.405,
        # non-integer, no raw-count layer). Never re-apply normalize_total/log1p.
        "assume_prenormalised": True,
        "n_hvg": 3000,  # null selects every gene
        "hvg_criterion": "raw_variance",  # raw_variance | dispersion | scanpy
        # scanpy calls sc.pp.highly_variable_genes with its seurat default,
        # which is what scDFM uses. dispersion is a plain variance/mean ratio
        # and picks a different set - the two differ by 1.6x in Control L2.
        "force_include_targets": True,
        # Exclude the fold's held-out conditions from the cells that CHOOSE the
        # gene space. scDFM does not do this - its HVG call runs over the whole
        # dataset - so turning it on makes our problem strictly harder than the
        # published one, and the cache becomes fold-specific. Off by default for
        # that reason; run.sh turns it on because a zero-shot claim needs it.
        "exclude_test_from_hvg": False,
        # Condition naming. Norman writes 'AHR+FEV' / 'AHR+ctrl' / 'ctrl';
        # other datasets use 'control' or 'DMSO'. Read through
        # src/data/conventions.py, never hardcoded.
        "control_label": "ctrl",
        "condition_separator": "+",
        "chunk_size": 10000,
    },
    "split": {
        # The additive folds ship WITH the dataset, so there is nothing to
        # regenerate and nothing to copy: assets/splits_additive.pkl was
        # byte-identical to this file. The combinations folds are a deterministic
        # function of the additive ones and are derived at load time rather than
        # cached, so no split artifact can drift out of sync with its source.
        # reference_pkl: the file that ships with the dataset (Norman).
        # obs_column:    a column of the cache's obs, for datasets that carry
        #                their split per cell instead (combosciplex).
        # generated:     make one deterministically from a seed, for a dataset
        #                that ships no split at all. Nothing is written to disk.
        "source": "reference_pkl",  # reference_pkl | obs_column | generated
        "reference_pkl": "data/norman/split_results.pkl",
        "obs_key": "split",
        "obs_test_value": "test",
        "generate_scheme": "combinations",  # doubles | combinations | group
        "generate_seed": 0,
        "generate_test_fraction": 0.3,
        "n_folds": 5,
        "group_key": "cell_type",  # for generate_scheme=group
        "method": "additive",  # additive | combinations
        "fold": 0,
    },
    "eval": {
        # power=1 is Szekely's energy distance. power=2 collapses to
        # 2*||mean_x - mean_y||^2 and would make the metric blind to everything
        # beyond first moments.
        "edist_power": 1,
        "n_gen_cells": 256,  # control cells the predicted shift is applied to
        # PROVISIONAL. Picking this by looking at test performance would not be
        # legitimate, so it is pinned rather than tuned; select it by inner CV on
        # the training doubles before quoting a final target line.
        "ridge_alpha": 1.0,
        "ridge_weight_by_cells": False,
        "device": "cuda",
        "seed": 0,
    },
    "model": {
        "latent_dim": 64,
        # --- representation backbone (ablation axis: the dynamics claim should
        #     hold on top of any of these, not just one) ---
        # Only P-CAB/E-RCA is built here. The mlp, transformer and scvi
        # backbones exist in the full-axis repository this was pruned from.
        "backbone": "pcab",
        "hidden": [1024, 512],
        "dropout": 0.1,
        # Decoder output head. 41.2 % of the data is exactly zero and an mse head
        # produces exact zeros 0.000 % of the time, so this matters for the
        # population-level metric.
        # hurdle only: 41 % of entries are exactly zero, and a plain MSE head
        # fixes a constant sigma, which collapses cell-to-cell variance.
        "decoder_head": "hurdle",
        "hurdle_bce_weight": 1.0,
        # How the binary detection event is realised at inference.
        # sample is the right default for a distribution-level metric; soft is
        # optimal for mean-only metrics. See HurdleHead.point_estimate.
        "hurdle_gate": "sample",  # soft | hard | sample
        # point pins the magnitude to its conditional mean (what plain MSE does);
        # gaussian learns a dispersion so the magnitude can be drawn as well.
        "hurdle_magnitude": "gaussian",  # point <- MSE | gaussian
        # --- latent dynamics ---
        # affine        u_a = s(t)*(A_a z + b_a). One 64x64 operator PER
        #               perturbation: 4,160 parameters each.
        # neural_field  u_a = f([z, phi(t), e_a]). One SHARED trunk, and the
        #               perturbation enters only through a 32-dim embedding: 32
        #               parameters each, a factor of 130 less dedicated capacity.
        # Measured on fold 0 with everything else fixed, affine won on both
        # backbones (mlp 0.3348 vs -0.4377, pcab 0.1965 vs -0.1346). The two arms
        # differ in capacity AND in linear-vs-nonlinear form at once, so that
        # attribution is not isolated - the reading that capacity mattered rests
        # on the more expressive arm being the one that lost.
        "generator": "affine",  # affine | neural_field
        "generator_hidden": [256, 256],  # neural_field trunk only
        # How the per-perturbation fields combine.
        # additive  v = sum_a u_a. First-order BCH, and the measured default.
        # learned   v = sum_a u_a + rho(sum_a phi(u_a)). Learns the composition
        #           law from the generators' OUTPUT VELOCITIES rather than
        #           truncating BCH at a fixed order. Distinct from the interaction
        #           term that already lost, which read perturbation embeddings -
        #           identity - where this reads the velocities themselves, so an
        #           unseen pair is on the same footing as a seen one.
        #           Starts exactly additive (rho's output layer is zero).
        "composition": "additive",  # additive | learned
        "composition_hidden": 128,
        # Where the flow STARTS for a combination. none transports a control cell
        # and the field must produce the whole displacement, 88 % of which is the
        # additive part. The other two shift the control cell in GENE SPACE first,
        # so the field only has to produce what additivity gets wrong - which is
        # exactly what resid_R2 measures.
        #
        #   none      z0 = encode(x_ctrl)
        #   additive  z0 = encode(x_ctrl + (m_A - m_ctrl) + (m_B - m_ctrl))
        #   ridge     z0 = encode(x_ctrl + w_A + w_B)
        #
        # Both shifts are built from TRAINING conditions only: in the additive
        # split every single is a training condition and ridge fits over the
        # training conditions alone, so no evaluated double is read. Under the
        # combinations split the singles of an evaluated double are held out and
        # neither shift exists - anchoring is defined for the additive split only.
        #
        # ValueComposition's output layer starts at zero, so at step 0 the field
        # is the zero field and the prediction IS the anchor: 0.000 resid_R2 for
        # additive, 0.533 for ridge (fold 1). Losing to the baseline the anchor
        # came from therefore requires getting WORSE than the starting point.
        "anchor": "none",  # none | additive | ridge
        # --- Koopman latent dynamics ---
        # u_a(z,t) = s(t) * (A_a z + b_a): a linear ODE in the latent space,
        # one 64x64 operator per perturbation. Koopman in FORM only - the
        # encoder is trained for reconstruction and frozen, not trained to
        # linearise the dynamics. There is no interaction term; see
        # src/models/flow.py for the measurements behind that.
        "time_embed_dim": 32,
        # dense    z = Linear(K*d_v -> latent_dim). The measured configuration.
        #          The projection mixes every pathway token, so no latent
        #          dimension corresponds to a pathway and A_a cannot be read.
        # pathway  z_k = <h_k, w> + b_k, one scalar per token, so latent_dim
        #          BECOMES K and dimension k IS pathway k. A_a then is a [K, K]
        #          pathway interaction matrix: A_a[i,j] is how much pathway j
        #          drives pathway i under perturbation a, which is readable.
        #          Note K includes n_free_tokens rows with no prior at all
        #          (53 of 101 perturbation targets sit in no usable KEGG
        #          pathway), and those dimensions are NOT pathways - report the
        #          annotated and free blocks separately.
        #          This buys interpretability, not accuracy: the 64-d bottleneck
        #          was measured innocent (residual share 0.575 in latent vs
        #          0.393 in gene space), so widening it is not expected to help.
        "latent_readout": "dense",  # dense | pathway
        # Low-rank factorisation of the Koopman operator, A_a = U_a V_a.
        # None keeps the full [latent_dim, latent_dim] operator. It exists for
        # latent_readout=pathway, where a full operator is K^2 ~ 74,000 per
        # perturbation (7.5 M over 101) against 4,160 at latent_dim=64.
        # rank 16 brings that back to roughly the dense-readout cost.
        "generator_rank": None,
        # --- P-CAB mask (stage 3) ---
        "n_pathway_tokens": None,  # None = however many KEGG pathways survive
        "n_free_tokens": 101,
        "d_key": 64,
        "d_value": 64,
        "mask_combine": "gate",  # gate (multiplicative) | logit_bias (control)
        "mask_mode": "hybrid",  # hybrid | prior_only | residual_only
        "mask_activation": "tanh",  # tanh (signed) | sigmoid (unsigned control)
        "mask_alpha": 1.0,
        "mask_share_enc_dec": True,
        "mask_l1": 1e-5,
        "mask_self_loop": True,
    },
    "train": {
        "stage1_epochs": 30,
        "stage2_epochs": 60,
        "batch_size": 256,
        # 0 = a full pass over the data. Set it low for smoke runs so a
        # configuration can be checked end-to-end in seconds.
        "max_steps_per_epoch": 0,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "kl_weight": 1e-3,
        # Whether stage 2 also trains the VAE. Named for the whole VAE, not just
        # the encoder: the decoder is frozen with it, and stage 2 runs the VAE in
        # eval mode so dropout is off and the latent is mu rather than a sample.
        #
        # Measured, against the handoff spec's recommendation to fine-tune:
        # letting the encoder move during stage 2 shrinks the latent (||z0|| 8 ->
        # 2.68) and therefore shrinks the flow-matching TARGET itself. fm fell to
        # 0.0019 not because the field fitted well but because z1 - z0 had been
        # made small, and transport under-predicted the true delta by 28 %.
        # Freezing gives fm 1.01 and resid_R2 -0.0003 vs -0.7867, i.e. the
        # additive structure is finally reproduced exactly as it should be.
        # The spec's reasoning is not wrong - a latent optimal for reconstruction
        # need not be one where generators compose well - but allowing the VAE to
        # move opens an easier path than improving that geometry: shrink the
        # latent, and the target shrinks with it. The stage-2 reconstruction term
        # below mitigates the collapse without preventing it, because the scale
        # can fall 3x while reconstruction stays fine. Freezing closes the path.
        # To have both later, pin the latent scale structurally (running-stat
        # normalisation inside encode_z) and fine-tuning becomes safe again.
        # Reuse a finished run's encoder and skip stage 1 entirely. Give it a
        # run directory or a checkpoint.pt path; None trains stage 1 as usual.
        #
        # Legitimate because stage 2 FREEZES the encoder: the same weights would
        # be produced again, and reusing them makes a field experiment differ from
        # its reference in the field alone. The latent standardisation travels
        # with the checkpoint (latent_mean / latent_std are buffers), so it is not
        # refitted either - refitting would resample 8,192 cells and move the
        # coordinate system by a hair for no reason.
        #
        # Stage 2 reseeds regardless of whether stage 1 ran, so a resumed run and
        # a fresh one draw the same minibatches from step one. Runs made before
        # that reseed existed will not reproduce bit-for-bit.
        "init_vae_from": None,
        "finetune_vae_in_stage2": False,
        # Weight on the reconstruction term kept alive during stage 2. Without it
        # the encoder collapses the latent, which is the global optimum of flow
        # matching on its own.
        "stage2_recon_weight": 1.0,
        "single_warmup_epochs": 10,  # singles only before combinations join
        # --- composition residual, the term that puts D into the loss ---
        # Measured on a finished run: ||int(u_a+u_b) - (int u_a + int u_b)|| is
        # 0.9945 of the sum itself, while the DATA is only 12% non-additive
        # (||d_AB|| / ||d_A + d_B|| = 0.879). The model invents eight times the
        # non-additivity it should, and nothing in the loss refers to that
        # quantity, so nothing stops it. This term supervises exactly it.
        #
        # 0 disables. The latent residual it matches is preserved by the encoder
        # (its share of the signal is 0.575 in latent vs 0.393 in gene space), so
        # supervising in latent space is not throwing information away.
        # Distribution-level term: MMD between the one-step endpoint estimate and
        # the coupled targets, over the whole minibatch. scDFM trains with this
        # (gamma=0.5, on gene space) and this model did not - every other term
        # here constrains a MEAN, so nothing asked the predicted population to
        # have the right shape. The weight does NOT transfer from theirs: the
        # kernel runs on the 64-d latent here, not on genes.
        "mmd_weight": 0.0,
        "resid_weight": 0.0,
        # --- endpoint matching, the FIRST-order half of the same idea ---
        # ||Phi_a(z_ctrl) - z_a||^2 over training conditions, singles included.
        # Measured without it (diagnose_transport.py on pcab_strict_commutator):
        # train singles reach 0.646 of their true displacement, train doubles
        # 0.869, test doubles 1.039 - the directly supervised conditions are the
        # worst reproduced. That inversion is a pair of cancelling errors: singles
        # 35 % short, composition non-additivity 8x too large, adding back to land
        # near 1.0 on doubles while pointing the wrong way (gene cosine 0.837).
        #
        # Use it WITH resid_weight. Constraining either half alone lets the model
        # push the error into the other, which is how the cancellation formed.
        #
        # Scale: this loss starts around ||z_a - z_ctrl||^2 / latent_dim, the same
        # order as the flow-matching term, so a weight near 1 already competes.
        "endpoint_weight": 0.0,
        # Integration steps for that term only. The residual is a population-mean
        # quantity, so one point is integrated rather than the batch, and step
        # count was measured not to matter (resid_R2 -0.9577 at 20 steps vs
        # -0.9825 at 100). Small keeps the kernel-launch cost down.
        "resid_steps": 5,
        # --- minibatch OT coupling; never random pairing ---
        "coupling": "uot",  # uot | ot | random (random is a control only)
        "uot_reg": 0.05,
        "uot_reg_marginal": 1.0,
        # 0 = fit the latent standardisation once before stage 2 and keep it.
        #
        # Do not turn this on without a reason. Refitting mid-training moves the
        # latent coordinates, and the velocity field was learned in the old ones:
        # measured at every refit, fm jumped 0.0087 -> 1.146 and each window came
        # back worse than the previous one. The encoder does drift when it is
        # fine-tuned, but the reconstruction term bounds that drift, which is the
        # cheaper of the two problems.
        "latent_renorm_every": 0,
        "n_integration_steps": 20,
        "grad_clip": 1.0,
        "device": "cuda",
        "seed": 0,
        "out_dir": "results/runs",
    },
}


def _coerce(text: str) -> Any:
    """Turn a command-line string into the value it obviously denotes."""
    lowered = text.lower()
    # Only "null" spells None. "none" stays a string, because it is a legitimate
    # value for at least one key (model.interaction) and silently turning it into
    # None made that config crash after the run had already started.
    if lowered == "null":
        return None
    # A bracketed value is JSON. model.hidden is a list, and without this it
    # arrived as the STRING "[2048,1024]" and the first Linear was built from
    # whatever len() of that string returned - a config error that only surfaced
    # as a shape mismatch deep in stage 1.
    if text[:1] in "[{":
        import json
        try:
            return json.loads(text)
        except ValueError:
            return text
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    """Apply `a.b=value` strings onto a copy of `config`.

    Unknown keys raise instead of being silently created, so a typo in a sweep
    script fails loudly rather than running with the default.
    """
    resolved = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        path, raw = item.split("=", 1)
        node = resolved
        keys = path.split(".")
        for key in keys[:-1]:
            if key not in node:
                raise KeyError(f"unknown config section {path!r}")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(f"unknown config key {path!r}")
        node[keys[-1]] = _coerce(raw)
    return resolved


def load(overrides: list[str] | None = None) -> dict:
    return apply_overrides(DEFAULTS, overrides)


def dumps(config: dict) -> str:
    return json.dumps(config, indent=2, sort_keys=True)
