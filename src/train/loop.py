"""Two-stage training: autoencode first, then learn the latent transport.

Stage 1 fits the VAE with the flow disabled, so the latent space exists before
anything tries to move through it.

Stage 2 fits the velocity field by flow matching along the straight interpolant
between OT-coupled latents. The encoder is fine-tuned here by default: freezing it
optimises the latent for reconstruction, whereas what we actually want is a space
in which the generators COMPOSE well - those are not the same objective.

Singles warm up first. Every u_a is identifiable from single perturbations alone,
so letting them settle stops the interaction term from absorbing error that really
belongs to the single-perturbation fields.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from ..data.dataset import ConditionSampler, PerturbationData, condition_genes
from ..models.flow import integrate
from .coupling import sample_pairs
from ..eval.scdfm_metrics import median_sigmas, mmd2_unbiased_multi_sigma


def _to_device(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(array, device=device)


def _aux(x: torch.Tensor, config: dict) -> dict:
    """Extra inputs a head may need during training.

    The hurdle head is the only head in this build and needs none. See
    src/eval/predict.py:_head_aux for the same statement on the evaluation side.
    """
    return {}


@torch.no_grad()
def condition_mean_latents(vae, data: PerturbationData, conditions, device: str,
                           n_cells: int = 256, chunk: int = 128) -> dict:
    """Mean latent per condition, encoded ONCE and in chunks.

    Two mistakes live here if this is written the obvious way, and both were made:

    1. An unchunked encode of n_cells rows is the largest single allocation in the
       run - the same pattern that put the gpu out of memory in fit_latent_scale,
       which is why that function takes a `chunk` (backbones.py). P-CAB attends
       277 tokens against every gene, so the attention tensor alone is
       rows x tokens x genes.
    2. Encoding the same condition once per target set. The endpoint targets and
       the residual targets want overlapping condition sets, so computing them
       separately doubles the pass for no reason and was measured taking over an
       hour before stage 2's first step.

    n_cells is 256 rather than every cell because these are POPULATION MEANS and
    conditions hold 46-1,005 cells; the mean of 256 is already far below the
    sampling noise the metric subtracts.
    """
    means: dict[str, torch.Tensor] = {}
    for condition in conditions:
        if condition in means or condition not in data.rows:
            continue
        rows = data.rows[condition]
        if len(rows) > n_cells:
            rows = rows[:n_cells]
        cells = _to_device(data.x[rows], device)
        mu = torch.cat([vae.encode_z(cells[start:start + chunk])[0]
                        for start in range(0, cells.shape[0], chunk)], dim=0)
        means[condition] = mu.mean(dim=0, keepdim=True)
    return means


@torch.no_grad()
def latent_residual_targets(vae, data: PerturbationData, sampler: ConditionSampler,
                            device: str, n_cells: int = 256,
                            means: dict | None = None) -> dict:
    """z_AB - z_A - z_B + z_ctrl for every TRAINING double, computed once.

    This is the quantity resid_R2 scores, expressed in the latent space. Nothing
    in the stage-2 loss referred to it before, which is why the model was free to
    invent a composition non-additivity 8x larger than the data's.

    Computed once rather than per step: stage 2 freezes the VAE, so these targets
    cannot move between epochs, and re-encoding the same cells every step would
    cost the largest allocation in the loop for a constant.

    A double whose singles are not both present is skipped - its residual is not
    defined - so the returned dict may be smaller than sampler.doubles.
    """
    naming = data.naming
    have = set(data.rows)

    def single_of(gene: str):
        for form in naming.single_forms(gene):
            if form in have:
                return form
        return None

    if means is None:
        wanted = [data.control_condition] + list(sampler.singles) + list(sampler.doubles)
        means = condition_mean_latents(vae, data, wanted, device, n_cells)

    control = means[data.control_condition]
    targets: dict[str, torch.Tensor] = {}
    for condition in sampler.doubles:
        a, b = naming.genes(condition)
        sa, sb = single_of(a), single_of(b)
        if sa is None or sb is None or condition not in means:
            continue
        if sa not in means or sb not in means:
            continue
        targets[condition] = (means[condition] - means[sa] - means[sb] + control)
    return {"targets": targets, "control": control}


@torch.no_grad()
def latent_endpoint_targets(vae, data: PerturbationData, sampler: ConditionSampler,
                            device: str, n_cells: int = 256,
                            means: dict | None = None) -> dict:
    """z_a for every TRAINING condition, and z_ctrl, computed once.

    Where the composition term supervises the SECOND-order structure of the flow
    map, this supervises the first: after integrating from the control mean under
    condition a, the model should land on condition a's mean.

    Nothing in the loss said so before, and the model does not do it. Measured on
    pcab_strict_commutator with diagnose_transport.py:

        train singles   displacement ratio 0.646   (directly supervised!)
        train doubles                      0.869
        test doubles                       1.039

    The order is inverted - the conditions the loss sees most directly are the
    ones it reproduces worst - which is what a pair of cancelling errors looks
    like: singles 35 % short, and a composition non-additivity 8x larger than the
    data's, adding back just enough to land near 1.0 on doubles. The direction is
    wrong even where the length is right (gene-space cosine 0.837).

    That matters more than it sounds. resid_R2's numerator is exactly
    ||m_hat - m_AB||, so the metric is dominated by how well the DOUBLE's mean is
    predicted, and 88 % of that signal is additive - ridge_additive reaches 0.53
    with no interaction term at all. Getting the endpoints right is therefore the
    larger half of the problem, not a detail beneath the interaction claim.

    Same caching argument as latent_residual_targets: stage 2 freezes the VAE, so
    these are constants.
    """
    if means is None:
        wanted = [data.control_condition] + list(sampler.singles) + list(sampler.doubles)
        means = condition_mean_latents(vae, data, wanted, device, n_cells)

    control = means[data.control_condition]
    targets = {c: v for c, v in means.items() if c != data.control_condition}
    # Where each condition's endpoint integration STARTS. Anchored combinations
    # start from the shifted control mean, exactly as predict_cells does; without
    # this the term would supervise a trajectory the model never takes and would
    # pull the field back into reproducing the additive part it no longer owns.
    starts = {}
    anchor = getattr(sampler, "anchor", None) or {}
    if anchor:
        shifted = [c for c in anchor if c in targets]
        if shifted:
            control_cells = data.cells(data.control_condition)
            take = min(n_cells, control_cells.shape[0])
            base = control_cells[:take]
            for condition in shifted:
                shift = anchor[condition].astype(base.dtype, copy=False)
                with torch.no_grad():
                    z, _ = vae.encode_z(torch.as_tensor(base + shift, device=device))
                starts[condition] = z.mean(dim=0, keepdim=True)
    return {"targets": targets, "control": control, "starts": starts}


def composition_residual(field, z0: torch.Tensor, perturbations: list[int],
                         n_steps: int) -> torch.Tensor:
    """Phi_ab(z0) - Phi_a(z0) - Phi_b(z0) + z0, on a single point.

    resid_R2 is a statement about population MEANS, so integrating the mean of z0
    supervises the same quantity as integrating every cell would, at 1/batch_size
    the cost. z0 is expected to be [1, latent_dim].
    """
    a, b = perturbations
    both = integrate(field, z0, [a, b], n_steps)
    only_a = integrate(field, z0, [a], n_steps)
    only_b = integrate(field, z0, [b], n_steps)
    return both - only_a - only_b + z0


def training_rows(data: PerturbationData, conditions) -> np.ndarray:
    """Row indices of the cells a model is allowed to see.

    Stage 1 used to draw from every cell in the cache, which meant the encoder and
    the latent standardisation both read the expression profiles of the held-out
    conditions - 10,643 cells, 12.5% of the dataset. That is transductive
    representation learning, and it invalidates a zero-shot combination claim even
    though no perturbation LABEL leaks.

    Control is added explicitly: it is never held out, every prediction starts from
    it, and it is not part of `training_conditions` for the additive split.
    """
    wanted = set(conditions) | {data.control_condition}
    rows = [data.rows[c] for c in sorted(wanted) if c in data.rows]
    return np.concatenate(rows) if rows else np.arange(data.x.shape[0])


def train_stage1(vae, data: PerturbationData, config: dict, device: str,
                 rng: np.random.Generator, log,
                 rows_allowed: np.ndarray | None = None) -> None:
    train_cfg = config["train"]
    optimiser = torch.optim.AdamW(vae.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    # None keeps the old behaviour (every cell). Callers that care about leakage
    # pass the training rows; scripts/train.py does.
    pool = (np.arange(data.x.shape[0]) if rows_allowed is None
            else np.asarray(rows_allowed))
    n_cells = len(pool)
    batch_size = train_cfg["batch_size"]
    steps = max(n_cells // batch_size, 1)
    if train_cfg["max_steps_per_epoch"]:
        steps = min(steps, train_cfg["max_steps_per_epoch"])

    vae.train()
    for epoch in range(train_cfg["stage1_epochs"]):
        totals: dict[str, float] = {}
        for _ in range(steps):
            rows = pool[rng.choice(n_cells, size=batch_size, replace=False)]
            x = _to_device(data.x[rows], device)
            params, mu, logvar = vae(x)
            recon, parts = vae.loss(params, x, **_aux(x, config))
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + train_cfg["kl_weight"] * kl
            parts = {**parts, "kl": float(kl)}
            # Sparsity on the learned mask correction, so "how far from the
            # annotation" stays a budget rather than a free-for-all. Only pcab
            # has a mask; every other backbone skips this.
            if hasattr(vae, "mask_penalty"):
                penalty = vae.mask_penalty()
                loss = loss + config["model"]["mask_l1"] * penalty
                parts = {**parts, "mask_l1": float(penalty)}
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), train_cfg["grad_clip"])
            optimiser.step()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
        summary = "  ".join(f"{k} {v / steps:.5f}" for k, v in totals.items())
        log(f"  stage1 epoch {epoch + 1:3d}/{train_cfg['stage1_epochs']}  {summary}")


def train_stage2(vae, field, data: PerturbationData, sampler: ConditionSampler,
                 config: dict, device: str, rng: np.random.Generator, log,
                 rows_allowed: np.ndarray | None = None,
                 run_dir: str | None = None) -> None:
    """`run_dir` enables periodic checkpointing; None keeps the old behaviour.

    Stage 2 is the long half - 600 epochs is on the order of a week - and the only
    torch.save used to be the one after evaluation, so anything that ended a run
    early (a kill, an OOM, a reboot) discarded every epoch of it AND the finished
    stage 1 with it. train.stage2_save_every epochs bounds that loss.
    """
    train_cfg = config["train"]
    parameters = list(field.parameters())
    if train_cfg["finetune_vae_in_stage2"]:
        parameters += list(vae.parameters())
    optimiser = torch.optim.AdamW(parameters, lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])

    # Same pool as stage 1: the standardisation statistics are part of the encoder,
    # so reading held-out cells here leaks exactly as much as training on them.
    pool = (np.arange(data.x.shape[0]) if rows_allowed is None
            else np.asarray(rows_allowed))
    scale_rows = pool[rng.choice(len(pool), size=min(8192, len(pool)), replace=False)]
    if train_cfg.get("init_vae_from"):
        # The statistics arrived with the checkpoint. Refitting would resample
        # 8,192 cells and shift the coordinate system slightly, which is exactly
        # what a resumed run must not do - the field is being compared against a
        # reference trained in the coordinates already stored here.
        log(f"  latent standardisation reused from the loaded encoder "
            f"(mean per-dim std {float(vae.latent_std.mean()):.5f})")
    else:
        raw_norm, mean_std = vae.fit_latent_scale(_to_device(data.x[scale_rows], device))
        log(f"  latent standardised: raw ||std|| {raw_norm:.4f} -> unit "
            f"(mean per-dim std was {mean_std:.5f})")

    anchored = bool(getattr(sampler, "anchor", None))
    resid_weight = train_cfg.get("resid_weight", 0.0)
    mmd_weight = train_cfg.get("mmd_weight", 0.0)
    endpoint_weight = train_cfg.get("endpoint_weight", 0.0)

    # ONE encoding pass shared by both target sets. Doing it inside each of them
    # encodes every condition twice, which is minutes of gpu time before stage 2
    # takes its first step - and was an hour when the encode was also unchunked.
    means = None
    if resid_weight > 0 or endpoint_weight > 0:
        wanted = ([data.control_condition] + list(sampler.singles)
                  + list(sampler.doubles))
        started_targets = time.time()
        means = condition_mean_latents(vae, data, wanted, device)
        log(f"  condition mean latents: {len(means)} conditions encoded in "
            f"{time.time() - started_targets:.1f}s")

    if anchored:
        log(f"  anchored: {len(sampler.anchor)} combinations start from a shifted "
            f"control; the field carries the correction only")
        if resid_weight > 0:
            log("  [warn] resid_weight > 0 is ignored under an anchor "
                "(see the composition residual comment in this file)")

    resid = None
    if resid_weight > 0 and not anchored:
        resid = latent_residual_targets(vae, data, sampler, device, means=means)
        sizes = [float(v.norm()) for v in resid["targets"].values()]
        log(f"  composition residual on: {len(sizes)} training doubles, "
            f"target ||r|| median {float(np.median(sizes)):.4f}")

    # The first-order half of the same idea. Both terms supervise the induced flow
    # MAP's population statistics rather than the velocity field pointwise, and
    # they are meant to be used together: constraining only one leaves the model
    # free to move the error into the other, which is how the current cancellation
    # arose in the first place.
    endpoint = None
    if endpoint_weight > 0:
        endpoint = latent_endpoint_targets(vae, data, sampler, device, means=means)
        moves = [float((v - endpoint["control"]).norm())
                 for v in endpoint["targets"].values()]
        log(f"  endpoint matching on: {len(moves)} training conditions, "
            f"target ||z_a - z_ctrl|| median {float(np.median(moves)):.4f}")

    for epoch in range(train_cfg["stage2_epochs"]):
        if train_cfg["latent_renorm_every"] and epoch and                 epoch % train_cfg["latent_renorm_every"] == 0:
            # Refreshing stale statistics, at the cost of moving the coordinate
            # system the field was trained in. Off by default - see config.
            vae.fit_latent_scale(_to_device(data.x[scale_rows], device))
            log(f"  [warn] latent renormalised at epoch {epoch + 1}; the field "
                f"was fitted in the previous coordinates")
        singles_only = epoch < train_cfg["single_warmup_epochs"]
        conditions = sampler.epoch(singles_only=singles_only)
        if train_cfg["max_steps_per_epoch"]:
            conditions = conditions[:train_cfg["max_steps_per_epoch"]]
        vae.train(train_cfg["finetune_vae_in_stage2"])
        field.train()

        total, total_match, count = 0.0, 0.0, 0
        totals_resid: list[float] = []
        totals_end: list[float] = []
        totals_mmd: list[float] = []
        for condition in conditions:
            source, target, _ = sampler.batch(condition)
            perturbations = [data.pert_index[g] for g in condition_genes(condition)]

            x0 = _to_device(source, device)
            x1 = _to_device(target, device)
            if train_cfg["finetune_vae_in_stage2"]:
                z0, _ = vae.encode_z(x0)
                z1, _ = vae.encode_z(x1)
            else:
                with torch.no_grad():
                    z0, _ = vae.encode_z(x0)
                    z1, _ = vae.encode_z(x1)

            z0p, z1p = sample_pairs(z0, z1, train_cfg["coupling"], train_cfg["uot_reg"],
                                    train_cfg["uot_reg_marginal"], rng)
            # One t PER SAMPLE. Drawing a single scalar for the whole batch gives
            # the time axis one sample per step instead of `batch_size`, so [0, 1]
            # is covered sparsely and the gradient is far noisier. Inference then
            # integrates through 20 RK4 times, passing through regions the field
            # barely saw, and the error accumulates as a displacement that is too
            # small - the measured prediction was 72 % of the true delta.
            t = torch.rand(z0p.shape[0], 1, device=device)
            z_t = (1.0 - t) * z0p + t * z1p
            predicted = field(z_t, t.reshape(-1), perturbations)
            matching = torch.nn.functional.mse_loss(predicted, z1p - z0p)
            loss = matching

            # Distribution-level supervision, the term scDFM trains with and this
            # model did not have. Flow matching and the two population terms all
            # constrain MEANS - a velocity at a point, an endpoint, a composition
            # residual - so nothing here asks the predicted population to have the
            # right SHAPE. That is the axis the reported table measures with DS
            # and edist_rel, and the one this model trails on.
            #
            # The endpoint estimate is a single-step extrapolation rather than an
            # RK4 integration, matching scDFM's `x1_hat = x_t + v*(1-t)`. It is
            # cheap enough to run on the whole batch every step, which is what a
            # distribution term needs; integrating the batch properly would cost
            # n_steps*4 field evaluations for a quantity that only has to be
            # approximately right for the kernel comparison.
            #
            # OURS IS IN THE LATENT SPACE and theirs is in gene space. The MMD
            # kernel is not invariant to that, so the weight does not transfer
            # from their gamma=0.5 and the two numbers are not comparable. State
            # the difference rather than implying the setups match.
            if mmd_weight > 0:
                z1_hat = z_t + predicted * (1.0 - t)
                sigmas = median_sigmas(z1p, scales=(0.5, 1.0, 2.0, 4.0))
                mmd = mmd2_unbiased_multi_sigma(z1_hat, z1p, sigmas)
                loss = loss + mmd_weight * mmd
                totals_mmd.append(float(mmd))

            # Endpoint matching, on singles AND doubles: integrating from the
            # control mean under this condition should land on the condition's
            # mean. One integration of one point, so the cost is `resid_steps`
            # RK4 steps rather than a batch-sized pass.
            if endpoint is not None and condition in endpoint["targets"]:
                start = endpoint.get("starts", {}).get(condition, endpoint["control"])
                z_end = integrate(field, start, perturbations,
                                  train_cfg["resid_steps"])
                e_loss = torch.nn.functional.mse_loss(z_end,
                                                      endpoint["targets"][condition])
                loss = loss + endpoint_weight * e_loss
                totals_end.append(float(e_loss))

            # The composition residual. Only doubles have one, and only the ones
            # whose singles both exist in the data - see latent_residual_targets.
            #
            # Skipped under an anchor. Phi_ab - Phi_a - Phi_b + z0 assumes all
            # three integrations share a z0; an anchored Phi_ab starts somewhere
            # else, so the expression no longer denotes the composition residual.
            # It is also redundant there - the anchored field IS the residual -
            # which is why run.sh sets RESID_WEIGHT=0 alongside ANCHOR.
            if resid is not None and not anchored and condition in resid["targets"]:
                r_model = composition_residual(field, z0.mean(dim=0, keepdim=True),
                                               perturbations, train_cfg["resid_steps"])
                r_loss = torch.nn.functional.mse_loss(r_model, resid["targets"][condition])
                loss = loss + resid_weight * r_loss
                totals_resid.append(float(r_loss))

            # Flow matching ALONE has a degenerate optimum when the encoder is
            # trainable: collapse the latent, and z1 - z0 becomes 0 so any field
            # scores perfectly. Measured before this term was added: ||z1 - z0||
            # fell to 0.019 while ||z0|| stayed ~8, and no transport happened at
            # all. Keeping the reconstruction objective on removes that escape.
            if train_cfg["finetune_vae_in_stage2"]:
                params0 = vae.decode_z(z0)
                recon, _ = vae.loss(params0, x0, **_aux(x0, config))
                loss = loss + train_cfg["stage2_recon_weight"] * recon

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, train_cfg["grad_clip"])
            optimiser.step()
            total += float(loss)
            total_match += float(matching)
            count += 1

        phase = "singles" if singles_only else "all"
        # resid is logged separately from fm so the two can be read apart: the
        # whole point of the term is that it moves a quantity fm never saw.
        extra = (f"  mmd {np.mean(totals_mmd):.5f}" if totals_mmd else "")
        extra += (f"  end {np.mean(totals_end):.5f}" if totals_end else "")
        extra += (f"  resid {np.mean(totals_resid):.5f}" if totals_resid else "")
        log(f"  stage2 epoch {epoch + 1:3d}/{train_cfg['stage2_epochs']}  "
            f"[{phase:7s}] loss {total / max(count, 1):.5f}  "
            f"fm {total_match / max(count, 1):.5f}{extra}")

        # Written to a DIFFERENT name than checkpoint.pt on purpose: run.sh treats
        # checkpoint.pt as "this run finished" and refuses to start over one, so a
        # mid-training file under that name would lock the tag out. Evaluate a
        # partial run with --init-vae-from pointing here, or rename it by hand.
        every = train_cfg.get("stage2_save_every", 0)
        if run_dir and every and (epoch + 1) % every == 0:
            torch.save({"vae": vae.state_dict(), "field": field.state_dict(),
                        "config": config, "stage2_epoch": epoch + 1},
                       os.path.join(run_dir, "checkpoint_partial.pt"))
