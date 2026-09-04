"""Stage 1 + stage 2 training, then evaluation against the baseline target line.

    python scripts/train.py
    python scripts/train.py --set model.generator=affine split.method=combinations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.data import splits
from src.data.conventions import ConditionNaming
from src.data.dataset import ConditionSampler, PerturbationData
from src.eval import baselines
from src.eval.predict import evaluate_model
from src.models.flow import PKFMField
from src.models.backbones import build_backbone
from src.train.loop import train_stage1, train_stage2, training_rows


def build_logger(path: str):
    handle = open(path, "a", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        handle.write(message + "\n")
        handle.flush()

    return log, handle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--tag", default=None, help="run directory name")
    args = parser.parse_args()

    config = config_module.load(args.overrides)
    method = config["split"]["method"]
    device = config["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable, falling back to cpu")
        device = config["train"]["device"] = config["eval"]["device"] = "cpu"

    tag = args.tag or (f"koopman_{config['model']['backbone']}"
                       f"_{method}_{time.strftime('%m%d-%H%M')}")
    run_dir = os.path.join(config["train"]["out_dir"], tag)
    os.makedirs(run_dir, exist_ok=True)
    log, handle = build_logger(os.path.join(run_dir, "train.log"))

    torch.manual_seed(config["train"]["seed"])
    rng = np.random.default_rng(config["train"]["seed"])

    log(f"run {tag}")
    log(f"config {json.dumps(config['model'])}")

    naming = ConditionNaming.from_config(config)
    data = PerturbationData(config["data"]["cache_h5ad"], naming=naming)
    fold = splits.folds(config, method)[config["split"]["fold"]]
    log(f"data: {data.x.shape[0]:,} cells x {data.n_genes:,} genes, "
        f"{data.n_perturbations} perturbations")

    stats = baselines.ConditionMeans(data.x, _condition_vector(data), naming)
    train_conditions = baselines.training_conditions(stats, fold, method)
    anchor_kind = config["model"].get("anchor", "none")
    anchor = baselines.anchor_deltas(anchor_kind, stats, train_conditions,
                                     list(stats.mean),
                                     alpha=config["eval"]["ridge_alpha"])
    sampler = ConditionSampler(data, train_conditions, config["train"]["batch_size"],
                               rng, anchor=anchor)
    if anchor_kind != "none":
        # A combination with no shift would train unanchored while the field is in
        # anchored mode - correction only, no additive part - so it would be asked
        # to produce the whole displacement with the term that produces it removed.
        # Under the additive split the count matches; anywhere it does not, the
        # split cannot support anchoring and the run should not start.
        doubles = [c for c in stats.mean if naming.is_double(c)]
        missing = [c for c in doubles if c not in anchor]
        log(f"anchor={anchor_kind}: {len(anchor)} of {len(doubles)} combinations "
            f"have a shift")
        if missing:
            raise SystemExit(
                f"model.anchor={anchor_kind!r} leaves {len(missing)} combinations "
                f"without a shift, e.g. {missing[:3]}. Anchoring needs both singles "
                f"of every combination to be training conditions, which holds in "
                f"the additive split and not in combinations. Use anchor=none.")
    log(f"train conditions: {len(sampler.singles)} singles + {len(sampler.doubles)} doubles")

    # Every cell a model is allowed to see. Stage 1 and the latent standardisation
    # both draw from here, so no held-out condition reaches the encoder.
    allowed = training_rows(data, train_conditions)
    held_out = data.x.shape[0] - len(allowed)
    log(f"cells visible to training: {len(allowed):,} / {data.x.shape[0]:,}  "
        f"({held_out:,} held-out cells excluded)")

    vae = build_backbone(config, data.n_genes, data.gene_names).to(device)
    field = PKFMField(config, data.n_perturbations, vae.latent_dim).to(device)
    log(f"backbone={config['model']['backbone']} head={config['model']['decoder_head']}")
    log(f"params: vae {sum(p.numel() for p in vae.parameters()):,}  "
        f"field {sum(p.numel() for p in field.parameters()):,}")

    started = time.time()
    init_from = config["train"].get("init_vae_from")
    if init_from:
        path = (init_from if init_from.endswith(".pt")
                else os.path.join(init_from, "checkpoint.pt"))
        log("\n=== stage 1: SKIPPED, encoder loaded ===")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        # An encoder is only reusable for the SAME representation problem. Loading
        # one trained on another fold or gene space would not raise - the shapes
        # can match while the gene axis means something else entirely - so the
        # settings that define the encoder are compared explicitly. Everything
        # about the FIELD is free to differ; that is the point of resuming.
        must_match = [("data", "cache_h5ad"), ("data", "n_hvg"),
                      ("data", "hvg_criterion"), ("split", "fold"),
                      ("model", "backbone"), ("model", "latent_dim"),
                      ("model", "latent_readout"), ("model", "d_key"),
                      ("model", "d_value"), ("model", "n_free_tokens"),
                      ("model", "decoder_head"), ("model", "mask_mode"),
                      ("model", "mask_activation"), ("model", "mask_combine")]
        mismatches = [
            f"{section}.{key}: loaded {checkpoint['config'][section].get(key)!r} "
            f"but this run wants {config[section].get(key)!r}"
            for section, key in must_match
            if checkpoint["config"][section].get(key) != config[section].get(key)]
        if mismatches:
            raise ValueError(
                "the checkpoint's encoder does not belong to this run:\n  "
                + "\n  ".join(mismatches)
                + "\nTrain stage 1 here instead (unset train.init_vae_from).")
        vae.load_state_dict(checkpoint["vae"])
        log(f"  from {path}")
        log(f"  latent standardisation came with it: "
            f"mean per-dim std {float(vae.latent_std.mean()):.5f}")
    else:
        log("\n=== stage 1: autoencoding ===")
        train_stage1(vae, data, config, device, rng, log, allowed)
        # Saved HERE, not only at the end. finetune_vae_in_stage2 is False, so
        # the encoder is frozen from this point and these are the weights the
        # final checkpoint would carry. Without it a run killed during stage 2 -
        # which is days long - threw away a finished stage 1 too, and the next
        # run had nothing to hand to --init-vae-from.
        #
        # A separate file rather than checkpoint.pt: run.sh refuses to start
        # when checkpoint.pt exists, and a stage-1-only artifact is not a
        # finished run.
        torch.save({"vae": vae.state_dict(), "config": config},
                   os.path.join(run_dir, "stage1.pt"))
        log(f"  encoder saved -> {os.path.join(run_dir, 'stage1.pt')}")

    # Stage 2 starts from a known state whether or not stage 1 ran. Without this
    # the rng that stage 1 consumed would shift every minibatch, coupling, and
    # time sample in stage 2, and a resumed run could not be compared against a
    # fresh one - which is the whole point of resuming.
    torch.manual_seed(config["train"]["seed"] + 1)
    rng = np.random.default_rng(config["train"]["seed"] + 1)
    sampler.rng = rng

    log("\n=== stage 2: latent flow matching ===")
    train_stage2(vae, field, data, sampler, config, device, rng, log, allowed,
                 run_dir=run_dir)
    log(f"\ntrained in {time.time() - started:.1f}s")

    log("\n=== evaluation (same protocol as the baselines) ===")
    field.anchor_table = anchor
    results = evaluate_model(vae, field, data, stats, [fold], method, config, rng,
                             anchor=anchor)
    for key, value in results.items():
        log(f"  {key:18s} {value}")

    torch.save({"vae": vae.state_dict(), "field": field.state_dict(), "config": config},
               os.path.join(run_dir, "checkpoint.pt"))
    with open(os.path.join(run_dir, "results.json"), "w") as out:
        json.dump({"config": config, "results": results}, out, indent=2)
    log(f"-> {run_dir}")
    handle.close()


def _condition_vector(data: PerturbationData) -> np.ndarray:
    """Rebuild the per-cell condition labels from the row index."""
    labels = np.empty(data.x.shape[0], dtype=object)
    for condition, rows in data.rows.items():
        labels[rows] = condition
    return labels


if __name__ == "__main__":
    main()
