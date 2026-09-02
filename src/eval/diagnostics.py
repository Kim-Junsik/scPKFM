"""The two things results.json cannot separate, measured from a finished checkpoint.

resid_R2 collapses several very different failures onto one number, and two of
them are indistinguishable in the summary table:

  under-transport   the field barely moves the control cells, so the predicted
                    non-additive residual is ~0 by default. The run then scores
                    resid_R2 ~ 0 and reads as "additivity reproduced exactly"
                    when the truth is "the model did nothing". This is not
                    hypothetical - it is what the pre-per-sample-t code did.

Neither is visible downstream, and only the second one has ever been caught here,
by hand. Both are computed from checkpoint.pt alone, so a running sweep is not
disturbed and nothing has to be retrained.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ..data.conventions import ConditionNaming
from ..data.dataset import PerturbationData
from ..data import splits
from ..models.backbones import build_backbone
from ..models.flow import PKFMField, integrate
from .baselines import ConditionMeans, training_conditions
from .predict import _head_aux


# A sweep scores six runs of one dataset, and the load dominates everything else
# here: the h5ad is ~85k x 3k and ConditionMeans walks all of it for ~200
# conditions. Both are functions of the config alone, so they are shared.
_DATASETS: dict[tuple, tuple] = {}
_FOLDS: dict[tuple, list] = {}


def _dataset(config: dict):
    """PerturbationData + ConditionMeans, shared across runs of one dataset.

    Keyed on what actually determines them rather than on the run, so a run
    pointing at a different cache or naming convention still builds its own
    instead of silently reusing the wrong cells.
    """
    data_cfg = config["data"]
    key = (data_cfg["cache_h5ad"], data_cfg.get("control_label", "ctrl"),
           data_cfg.get("condition_separator", "+"))
    if key not in _DATASETS:
        naming = ConditionNaming.from_config(config)
        data = PerturbationData(data_cfg["cache_h5ad"], naming=naming)
        labels = np.empty(data.x.shape[0], dtype=object)
        for condition, rows in data.rows.items():
            labels[rows] = condition
        _DATASETS[key] = (data, ConditionMeans(data.x, labels, naming))
    return _DATASETS[key]


def _folds(config: dict, method: str) -> list:
    """The fold list, shared the same way. Every value in `split` can change it,
    so the whole section is the key."""
    key = (method, tuple(sorted((k, repr(v)) for k, v in config["split"].items())))
    if key not in _FOLDS:
        _FOLDS[key] = splits.folds(config, method)
    return _FOLDS[key]


def load_run(run_dir: str, device: str = "cpu"):
    """Rebuild a finished run from its checkpoint.

    Defaults to cpu: these diagnostics are cheap, and the usual reason to run
    them is that a sweep is still occupying the gpu.

    The returned data and stats are SHARED between calls. Nothing here writes to
    them, but a caller that wants to mutate them must copy first.
    """
    checkpoint = torch.load(os.path.join(run_dir, "checkpoint.pt"),
                            map_location=device, weights_only=False)
    config = checkpoint["config"]
    config["train"]["device"] = config["eval"]["device"] = device

    data, stats = _dataset(config)
    method = config["split"]["method"]
    fold = _folds(config, method)[config["split"]["fold"]]

    vae = build_backbone(config, data.n_genes, data.gene_names).to(device)
    vae.load_state_dict(checkpoint["vae"])
    field = PKFMField(config, data.n_perturbations, vae.latent_dim).to(device)
    field.load_state_dict(checkpoint["field"])
    vae.eval()
    field.eval()
    return config, data, stats, fold, vae, field


def condition_groups(data, stats, fold, method: str) -> dict[str, list[str]]:
    """The three populations worth reporting separately.

    Splitting train from test is the whole point: a field that transports the
    training singles correctly but not the held-out doubles has a generalisation
    problem, while one that fails on both is simply underfitted, and the fix is
    different in each case.
    """
    seen = training_conditions(stats, fold, method)
    naming = data.naming
    test_doubles = [c for c in fold["test"]
                    if naming.is_double(c) and stats.has(c)]
    return {
        "train singles": [c for c in seen if naming.is_single(c) and stats.has(c)],
        "train doubles": [c for c in seen if naming.is_double(c) and stats.has(c)],
        "test doubles": test_doubles,
    }


def subsample(conditions: list[str], limit: int, rng) -> list[str]:
    """Cap a group, deterministically, so a 189-condition sweep stays quick."""
    if limit <= 0 or len(conditions) <= limit:
        return list(conditions)
    pick = rng.choice(len(conditions), size=limit, replace=False)
    return [conditions[i] for i in sorted(pick)]


def _numpy(vector) -> np.ndarray:
    if isinstance(vector, torch.Tensor):
        return vector.detach().cpu().numpy()
    return np.asarray(vector)


def _ratio(predicted, true) -> float:
    predicted, true = _numpy(predicted), _numpy(true)
    denominator = float(np.linalg.norm(true))
    return float(np.linalg.norm(predicted)) / denominator if denominator else float("nan")


def _cosine(predicted, true) -> float:
    predicted, true = _numpy(predicted), _numpy(true)
    scale = float(np.linalg.norm(predicted)) * float(np.linalg.norm(true))
    return float(predicted @ true) / scale if scale else float("nan")


def _control_sample(data, n_cells: int, rng) -> np.ndarray:
    cells = data.cells(data.control_condition)
    pick = rng.choice(cells.shape[0], size=min(n_cells, cells.shape[0]), replace=False)
    return cells[pick]


@torch.no_grad()
def measure_transport(vae, field, data, stats, conditions: list[str], config,
                      rng, device: str, n_cells: int,
                      genes: np.ndarray | None = None) -> list[dict]:
    """How far, and in which direction, the field actually carries the cells.

    Reported as a ratio against the true displacement rather than as an error,
    because the two failures look identical in an error norm: a field that stops
    at 40 % of the way and one that overshoots to 160 % both score badly, and
    only the ratio says which. Cosine separates "too short" from "wrong way".

    Cells are unpaired, so the displacement only exists between population means.
    Both the predicted and the true displacement are taken from the SAME control
    sample, so the control's own sampling noise cancels out of the ratio.
    """
    n_steps = config["train"]["n_integration_steps"]
    rows = []
    for condition in conditions:
        if condition == data.control_condition or not stats.has(condition):
            continue
        x0 = torch.as_tensor(_control_sample(data, n_cells, rng), device=device)
        z0, _ = vae.encode_z(x0)

        perturbations = [data.pert_index[g] for g in data.naming.genes(condition)]
        z1_hat = integrate(field, z0, perturbations, n_steps)

        true_cells = data.cells(condition)
        take = rng.choice(true_cells.shape[0],
                          size=min(n_cells, true_cells.shape[0]), replace=False)
        z1_true, _ = vae.encode_z(torch.as_tensor(true_cells[take], device=device))

        origin = z0.mean(dim=0)
        predicted = vae.reconstruction(vae.decode_z(z1_hat), **_head_aux(vae, x0))
        gene_hat = _numpy(predicted.mean(dim=0))
        gene_true = stats.mean[condition]
        control_mean = stats.control
        if genes is not None:
            # Gene-space numbers only; the latent ones above are unaffected.
            gene_hat, gene_true = gene_hat[genes], gene_true[genes]
            control_mean = control_mean[genes]
        rows.append({
            "condition": condition,
            "n_true": int(true_cells.shape[0]),
            # Latent: the space the field is actually trained in.
            "latent_ratio": _ratio(z1_hat.mean(dim=0) - origin,
                                   z1_true.mean(dim=0) - origin),
            "latent_cos": _cosine(z1_hat.mean(dim=0) - origin,
                                  z1_true.mean(dim=0) - origin),
            # Gene space: what resid_R2 and edist_rel are computed on, so a gap
            # between the two rows localises the loss to the decoder.
            "gene_ratio": _ratio(gene_hat - control_mean, gene_true - control_mean),
            "gene_cos": _cosine(gene_hat - control_mean, gene_true - control_mean),
            # The reported table's L2: ||mu_hat_p - mu_p||_2, taken on the mean
            # vectors themselves rather than on the delta from control.
            "l2": float(np.linalg.norm(gene_hat - gene_true)),
        })
    return rows


@torch.no_grad()
def scdfm_eval_genes(data, fold: dict, n_top: int) -> np.ndarray:
    """The gene subset scDFM reports on, reproduced from its src/data_process/data.py.

    Their pipeline runs scanpy's HVG TWICE: once over the whole dataset to pick
    the modelled space (`n_top_genes`, 5000 in their run.sh) and again over the
    TEST subset alone (`infer_top_gene`, 1000), and the reported table is scored
    on the second. Without this our numbers sit on 3,074 raw-variance genes and
    theirs on 1,000 dispersion-binned ones, which is most of the apparent gap:
    measured on our cells, the same conditions give Control L2 5.69 under raw
    variance and 3.51 under dispersion, against their 3.99.

    scanpy is called rather than reimplemented. The seurat flavour bins genes by
    mean expression and z-scores dispersion WITHIN each bin, so a plain
    variance-to-mean ratio selects a different set.

    Their selection reads the test cells, so the evaluation gene set depends on
    the test distribution. That is their protocol; reproducing it is the point.
    """
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    wanted = set(fold["test"]) | {data.control_condition}
    rows = np.concatenate([data.rows[c] for c in sorted(wanted) if c in data.rows])
    subset = ad.AnnData(X=np.asarray(data.x[rows], dtype=np.float32),
                        var=pd.DataFrame(index=pd.Index(data.gene_names)))
    sc.pp.highly_variable_genes(subset, n_top_genes=min(n_top, subset.n_vars))
    return np.flatnonzero(subset.var["highly_variable"].to_numpy())
