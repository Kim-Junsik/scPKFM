"""Loading and validating the inherited scDFM splits.

These splits are NOT ours. They come from scDFM's src/data_process/data.py so that
"the authors' method won on the authors' split" cannot be raised against us. Two
properties therefore have to hold and are checked here rather than assumed:

  1. the shipped reference is a clean partition per fold (train and test disjoint).
  2. the combinations folds derive deterministically from the additive ones.

There is no cached split artifact. The additive folds ARE the file that ships with
the dataset, and the combinations folds are computed from them at load time - a
copy on disk could drift out of sync with its source and nothing would notice.

The additive folds are five INDEPENDENT random 70/30 draws with different seeds,
not a five-way partition, so test sets overlap between folds. Anything that needs
leak-free cells (e.g. running GRN inference) must intersect across all folds
rather than assume a partition.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

N_TEST_DOUBLES_FOR_COMBINATIONS = 15  # scDFM keeps only the first 15 test doubles
CONTROL_SUFFIX = "ctrl"


def load(path: str) -> list[dict[str, Any]]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def derive_combinations(additive_folds: list[dict], control_suffix: str = CONTROL_SUFFIX) -> list[dict]:
    """Reproduce the combinations split from the additive one.

    scDFM keeps the first 15 test doubles, then moves the singles of every gene
    appearing in those doubles into the test set as well. Those held-out singles
    are the reason a leak-safe cell selection cannot simply keep all singles.
    """
    derived = []
    for fold in additive_folds:
        test_doubles = list(fold["test"][:N_TEST_DOUBLES_FOR_COMBINATIONS])
        held_out_genes = {gene for pair in test_doubles for gene in pair.split("+")}
        held_out_singles = [f"{gene}+{control_suffix}" for gene in held_out_genes]
        derived.append({
            "test_doubles": test_doubles,
            "held_out_genes": held_out_genes,
            "held_out_singles": held_out_singles,
            "test": test_doubles + held_out_singles,
        })
    return derived


def _read_obs(config: dict, needs: tuple[str, ...] = ()):
    """obs from the cache when it can answer, otherwise from the raw file.

    Split validation runs BEFORE the cache is built, so requiring the cache made
    the two steps circular; and a cache built by an earlier run may predate the
    column the split needs, so "exists" is not enough - it has to actually carry
    `needs`. obs is identical in both files apart from the gene axis, so falling
    back to the raw file is always safe.
    """
    import os
    import anndata as ad

    cache = config["data"]["cache_h5ad"]
    if os.path.exists(cache):
        obs = ad.read_h5ad(cache, backed="r").obs
        if all(key in obs.columns for key in needs if key):
            return obs
    return ad.read_h5ad(config["data"]["raw_h5ad"], backed="r").obs


def folds_from_obs(config: dict) -> list[dict[str, Any]]:
    """One fold read out of an obs column, for datasets that ship their split that way.

    combosciplex is the case this exists for: it has no split pickle, but every
    cell carries obs['split'] in {train, test, ood}. Conditions are assigned to
    whichever value holds most of their cells - a condition split across values
    would otherwise land in both train and test.

    There is no k-fold structure here, so the result is a single fold and
    `split.fold` must stay 0.
    """
    split_cfg = config["split"]
    obs = _read_obs(config, (split_cfg["obs_key"],))
    conditions = obs["condition"].astype(str).to_numpy()
    if split_cfg["obs_key"] not in obs.columns:
        raise ValueError(
            f"obs has no column {split_cfg['obs_key']!r}; found {list(obs.columns)[:12]}")
    values = obs[split_cfg["obs_key"]].astype(str).to_numpy()

    assignment: dict[str, str] = {}
    for condition in set(conditions.tolist()):
        mask = conditions == condition
        labels, counts = np.unique(values[mask], return_counts=True)
        assignment[condition] = str(labels[counts.argmax()])

    held = split_cfg["obs_test_value"]
    test = sorted(c for c, v in assignment.items() if v == held)
    train = sorted(c for c, v in assignment.items() if v != held)
    if not test:
        available = sorted(set(assignment.values()))
        available = sorted(set(assignment.values()))
        raise ValueError(
            f"no condition is majority {held!r}. After condition-level "
            f"assignment the available values are {available}. A value that "
            f"exists per CELL but never wins a condition is a within-condition "
            f"cell holdout, not a condition-level split, and this model "
            f"predicts whole conditions - pick one of the values above. "
            f"For combosciplex use split.obs_test_value=ood.")
    return [{"train": train, "test": test, "train_doubles": train}]


def folds_generated(config: dict) -> list[dict[str, Any]]:
    """Make a split for a dataset that ships none.

    Deterministic from `split.generate_seed` and written nowhere: the folds are
    recomputed on every load, so there is no artifact that can drift away from the
    seed that produced it - the same reason the combinations folds are derived
    rather than cached.

    Uses the legacy RandomState rather than default_rng. numpy does not guarantee
    that default_rng's stream is stable across versions, and a split that quietly
    changes when numpy is upgraded would invalidate every number measured against
    it.

    Three schemes:

      doubles       hold out a fraction of the double perturbations. The singles
                    stay available, so the additive baselines remain computable.
      combinations  additionally hold out the singles of every held-out double.
                    Harder, and the additive baselines become uncomputable, which
                    is the point.
      group         hold out whole groups named by an obs column - one cell line,
                    one donor, one batch. A different generalisation axis from the
                    two above, and the one to use for "does this transfer to a
                    cell line the model never saw".
    """
    split_cfg = config["split"]
    naming = _naming(config)
    needs = (split_cfg["group_key"],) if split_cfg["generate_scheme"] == "group" else ()
    obs = _read_obs(config, needs)
    conditions = obs["condition"].astype(str).to_numpy()

    scheme = split_cfg["generate_scheme"]
    n_folds = split_cfg["n_folds"]
    fraction = split_cfg["generate_test_fraction"]
    seed = split_cfg["generate_seed"]

    if scheme == "group":
        key = split_cfg["group_key"]
        if key not in obs.columns:
            raise ValueError(
                f"obs has no column {key!r}; set split.group_key to the column that "
                f"names the cell line / donor / batch")
        groups = obs[key].astype(str).to_numpy()
        unique = sorted(set(groups.tolist()))
        if len(unique) < 2:
            raise ValueError(f"obs[{key!r}] has one group; nothing to hold out")
        generated = []
        for fold in range(n_folds):
            order = list(unique)
            np.random.RandomState(seed + fold).shuffle(order)
            n_test = max(1, int(round(len(unique) * fraction)))
            held_groups = set(order[:n_test])
            test_cells = np.isin(groups, list(held_groups))
            generated.append({
                "train": sorted(set(conditions[~test_cells].tolist())),
                "test": sorted(set(conditions[test_cells].tolist())),
                "held_out_groups": sorted(held_groups),
                "group_key": key,
            })
        return generated

    doubles = sorted({c for c in set(conditions.tolist()) if naming.is_double(c)})
    if not doubles:
        raise ValueError("no double-perturbation conditions found; nothing to hold out")

    generated = []
    for fold in range(n_folds):
        order = list(doubles)
        np.random.RandomState(seed + fold).shuffle(order)
        n_test = max(1, int(round(len(doubles) * fraction)))
        test_doubles = sorted(order[:n_test])
        train_doubles = sorted(order[n_test:])

        entry: dict[str, Any] = {"test_doubles": test_doubles,
                                 "train_doubles": train_doubles}
        if scheme == "combinations":
            genes = {g for pair in test_doubles for g in naming.genes(pair)}
            held_singles = [naming.single(g) for g in sorted(genes)]
            entry["held_out_genes"] = sorted(genes)
            entry["held_out_singles"] = held_singles
            entry["test"] = test_doubles + held_singles
            entry["train"] = train_doubles
        elif scheme == "doubles":
            entry["test"] = test_doubles
            entry["train"] = train_doubles
        else:
            raise ValueError(f"unknown split.generate_scheme {scheme!r}")
        generated.append(entry)
    return generated


def _naming(config: dict):
    from .conventions import ConditionNaming
    return ConditionNaming.from_config(config)


def folds(config: dict, method: str | None = None) -> list[dict[str, Any]]:
    """The folds for `method`.

    Two sources, chosen by `split.source`:

      reference_pkl  the file that ships with the dataset (Norman). additive is
                     that file verbatim; combinations is derived from it.
      obs_column     a column of the cache's obs (combosciplex). One fold, and
                     `method` is ignored because the dataset defines its own.

    additive     : exactly the scDFM reference.
    combinations : derived from it on the spot. Caching this to disk would create
                   a second artifact that can silently disagree with its source.
    """
    method = method or config["split"]["method"]
    source = config["split"].get("source", "reference_pkl")
    if source == "obs_column":
        return folds_from_obs(config)
    if source == "generated":
        return folds_generated(config)
    reference = load(config["split"]["reference_pkl"])
    if method == "additive":
        return reference
    if method == "combinations":
        derived = derive_combinations(reference)
        for fold, source in zip(derived, reference):
            # combinations holds out only the FIRST 15 test doubles, so every
            # other double is trainable - including the ones the additive fold
            # put in test. Deriving train from the additive train instead would
            # silently shrink it from 110 doubles to 88.
            all_doubles = list(source["train"]) + list(source["test"])
            held = set(fold["test"])
            fold["train_doubles"] = [c for c in all_doubles if c not in held]
            # Singles are added by the caller, which knows which ones the data
            # actually contains; see baselines.training_conditions.
            fold["train"] = list(fold["train_doubles"])
        return derived
    raise ValueError(f"unknown split method {method!r}")


def validate(config: dict) -> dict[str, Any]:
    """Sanity-check the split source. Raises on anything inconsistent."""
    source = config["split"].get("source", "reference_pkl")
    if source == "generated":
        generated = folds_generated(config)
        for i, fold in enumerate(generated):
            overlap = set(fold["train"]) & set(fold["test"])
            if overlap:
                raise ValueError(f"generated fold {i} puts {overlap} in both sides")
        return {"n_folds": len(generated), "reference_ok": True,
                "combinations_derived": False, "source": "generated",
                "additive_sizes": [(len(f["train"]), len(f["test"])) for f in generated],
                "combinations_sizes": [],
                "additive_test_pairwise_overlap": (0, 0),
                "doubles_train_in_every_fold": len(
                    set.intersection(*[set(f["train"]) for f in generated]))}

    if source == "obs_column":
        fold = folds_from_obs(config)[0]
        overlap = set(fold["train"]) & set(fold["test"])
        if overlap:
            raise ValueError(f"obs split puts {overlap} in both train and test")
        return {"n_folds": 1, "reference_ok": True, "combinations_derived": False,
                "additive_sizes": [(len(fold["train"]), len(fold["test"]))],
                "combinations_sizes": [], "source": "obs_column",
                "additive_test_pairwise_overlap": (0, 0),
                "doubles_train_in_every_fold": len(fold["train"])}

    reference = load(config["split"]["reference_pkl"])
    report: dict[str, Any] = {"n_folds": len(reference)}

    for i, fold in enumerate(reference):
        if not {"train", "test"} <= set(fold):
            raise ValueError(f"fold {i} is missing train/test")
        if set(fold["train"]) & set(fold["test"]):
            raise ValueError(f"fold {i} has a condition in both train and test")
    report["reference_ok"] = True
    report["additive_sizes"] = [(len(f["train"]), len(f["test"])) for f in reference]

    derived = derive_combinations(reference)
    report["combinations_sizes"] = [
        (len(f["test_doubles"]), len(f["held_out_singles"])) for f in derived
    ]
    for i, fold in enumerate(derived):
        expected = {g for pair in fold["test_doubles"] for g in pair.split("+")}
        if set(fold["held_out_genes"]) != expected:
            raise ValueError(f"fold {i} derivation is inconsistent")
    report["combinations_derived"] = True

    test_sets = [set(f["test"]) for f in reference]
    overlaps = [
        len(test_sets[i] & test_sets[j])
        for i in range(len(test_sets))
        for j in range(i + 1, len(test_sets))
    ]
    report["additive_test_pairwise_overlap"] = (min(overlaps), max(overlaps))
    report["doubles_train_in_every_fold"] = len(
        set.intersection(*[set(f["train"]) for f in reference])
    )
    return report


def held_out_conditions(config: dict) -> set[str]:
    """Every condition held out by ANY fold of ANY split method.

    This is the exclusion set for anything that must not see test information at
    all - GRN inference above all. It is deliberately conservative: one artifact
    that is safe everywhere beats per-split artifacts that can be mixed up.
    """
    excluded: set[str] = set()
    for method in ("additive", "combinations"):
        for fold in folds(config, method):
            excluded |= set(fold["test"])
    return excluded
