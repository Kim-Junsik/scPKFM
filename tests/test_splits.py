"""The split must be bit-for-bit the one that ships with the dataset.

This is a hard requirement, not a preference: the whole point of inheriting
scDFM's folds is that "the authors' method won on the authors' split" cannot be
raised against us. A split that drifts - by a rename, a re-sort, a cached copy, or
a change in the gene space - silently destroys that argument, and nothing in the
training loop would notice.

The gene space is allowed to differ from scDFM's. The split is not.

    python -m pytest tests/test_splits.py -v
"""

from __future__ import annotations

import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.data import splits


@pytest.fixture(scope="module")
def config():
    return config_module.load()


@pytest.fixture(scope="module")
def reference(config):
    with open(config["split"]["reference_pkl"], "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------- identity
def test_additive_is_the_shipped_file_verbatim(config, reference):
    """Not "equivalent", not "same set" - the same lists in the same order."""
    loaded = splits.folds(config, "additive")
    assert len(loaded) == len(reference)
    for i, (ours, theirs) in enumerate(zip(loaded, reference)):
        assert list(ours["train"]) == list(theirs["train"]), f"fold {i} train order differs"
        assert list(ours["test"]) == list(theirs["test"]), f"fold {i} test order differs"


def test_no_cached_split_artifact_exists():
    """A copy on disk can disagree with its source and nothing would notice."""
    for stale in ("assets/splits_additive.pkl", "assets/splits_combinations.pkl"):
        assert not os.path.exists(stale), (
            f"{stale} is back; splits must be read from the shipped reference only")


def test_folds_are_not_mutated_between_calls(config):
    """Callers hold real dicts; a mutation must not leak into the next load."""
    first = splits.folds(config, "additive")
    first[0]["train"].append("SENTINEL+SENTINEL")
    second = splits.folds(config, "additive")
    assert "SENTINEL+SENTINEL" not in second[0]["train"]


# ---------------------------------------------------------------- independence
@pytest.mark.parametrize("n_hvg", [1000, 3000, 5000, None])
def test_gene_space_does_not_touch_the_split(n_hvg, reference):
    """The gene space is a free choice; the split must be invariant to it."""
    altered = config_module.load([f"data.n_hvg={'null' if n_hvg is None else n_hvg}"])
    loaded = splits.folds(altered, "additive")
    for ours, theirs in zip(loaded, reference):
        assert list(ours["train"]) == list(theirs["train"])
        assert list(ours["test"]) == list(theirs["test"])


# ---------------------------------------------------------------- structure
def test_train_and_test_are_disjoint(config):
    for method in ("additive", "combinations"):
        for i, fold in enumerate(splits.folds(config, method)):
            overlap = set(fold["train"]) & set(fold["test"])
            assert not overlap, f"{method} fold {i} shares {overlap}"


def test_combinations_holds_out_the_singles_of_its_test_doubles(config):
    """This is what distinguishes the split, and what a naive baseline leaks through."""
    for i, fold in enumerate(splits.folds(config, "combinations")):
        genes = {g for pair in fold["test_doubles"] for g in pair.split("+")}
        assert set(fold["held_out_genes"]) == genes, f"fold {i}"
        assert set(fold["held_out_singles"]) == {f"{g}+ctrl" for g in genes}, f"fold {i}"
        assert set(fold["held_out_singles"]) <= set(fold["test"]), f"fold {i}"
        assert not set(fold["held_out_singles"]) & set(fold["train"]), (
            f"fold {i} keeps a held-out single in train")


def test_combinations_train_keeps_every_double_it_is_allowed(config, reference):
    """combinations holds out only 15 doubles, so 110 of the 125 stay trainable.

    Deriving this from the ADDITIVE train list instead gives 88 and drops the
    singles too, which lowers the ridge-additive target line from 0.1853 to
    0.0487 - i.e. it makes the bar easier without anything looking wrong. The
    numbers below are the ones the shipped combinations split had.
    """
    all_doubles = set(reference[0]["train"]) | set(reference[0]["test"])
    assert len(all_doubles) == 125

    for i, fold in enumerate(splits.folds(config, "combinations")):
        assert len(fold["test_doubles"]) == 15, f"fold {i}"
        assert len(fold["train_doubles"]) == 110, (
            f"fold {i} has {len(fold['train_doubles'])} train doubles, expected 110")
        assert not set(fold["train_doubles"]) & set(fold["test_doubles"]), f"fold {i}"


def test_combinations_training_conditions_include_the_surviving_singles(config):
    """Held-out singles must go, the rest must stay - 101 - held_out of them."""
    cache = config["data"]["cache_h5ad"]
    if not os.path.exists(cache):
        pytest.skip(f"{cache} not built")
    import anndata as ad
    import numpy as np
    from src.eval import baselines

    adata = ad.read_h5ad(cache)
    conditions = adata.obs["condition"].astype(str).to_numpy()
    stats = baselines.ConditionMeans(np.zeros((len(conditions), 1), dtype=np.float32),
                                     conditions)
    n_singles = sum(1 for c in stats.mean if c.endswith("+ctrl"))

    for i, fold in enumerate(splits.folds(config, "combinations")):
        allowed = baselines.training_conditions(stats, fold, "combinations")
        held = set(fold["held_out_singles"])
        assert not set(allowed) & held, f"fold {i} lets a held-out single into train"
        singles_kept = sum(1 for c in allowed if c.endswith("+ctrl"))
        assert singles_kept == n_singles - len(held), (
            f"fold {i} kept {singles_kept} singles, expected {n_singles - len(held)}")


def test_combinations_derives_from_the_same_reference(config, reference):
    derived = splits.folds(config, "combinations")
    for i, (fold, source) in enumerate(zip(derived, reference)):
        assert list(fold["test_doubles"]) == list(source["test"][:15]), f"fold {i}"


def test_folds_overlap_because_they_are_independent_draws(config):
    """Five 70/30 draws, NOT a five-way partition.

    Code that assumes a partition - leak-free cell selection above all - would be
    wrong, so the property is pinned here rather than left as a comment.
    """
    test_sets = [set(f["test"]) for f in splits.folds(config, "additive")]
    overlaps = [len(test_sets[i] & test_sets[j])
                for i in range(len(test_sets)) for j in range(i + 1, len(test_sets))]
    assert min(overlaps) > 0, "folds look like a partition; the split source changed"


# ---------------------------------------------------------------- generated splits
@pytest.mark.parametrize("scheme", ["doubles", "combinations"])
def test_generated_split_is_deterministic(scheme):
    """Same seed, same folds - it is recomputed on every load, never cached."""
    base = config_module.load(["split.source=generated",
                               f"split.generate_scheme={scheme}"])
    first = splits.folds(base)
    second = splits.folds(base)
    for a, b in zip(first, second):
        assert list(a["test"]) == list(b["test"])
        assert list(a["train"]) == list(b["train"])


def test_generated_split_moves_with_the_seed():
    base = config_module.load(["split.source=generated"])
    if not os.path.exists(base["data"]["cache_h5ad"]):
        pytest.skip("cache not built")
    other = config_module.load(["split.source=generated", "split.generate_seed=7"])
    assert splits.folds(base)[0]["test"] != splits.folds(other)[0]["test"]


def test_generated_combinations_holds_out_the_constituent_singles():
    """The scheme that makes the additive baselines uncomputable must actually do it."""
    config = config_module.load(["split.source=generated",
                                 "split.generate_scheme=combinations"])
    if not os.path.exists(config["data"]["cache_h5ad"]):
        pytest.skip("cache not built")
    for i, fold in enumerate(splits.folds(config)):
        genes = {g for pair in fold["test_doubles"] for g in pair.split("+")}
        assert set(fold["held_out_genes"]) == genes, f"fold {i}"
        assert set(fold["held_out_singles"]) <= set(fold["test"]), f"fold {i}"
        assert not set(fold["held_out_singles"]) & set(fold["train"]), f"fold {i}"


def test_generated_split_never_shares_a_condition():
    for scheme in ("doubles", "combinations"):
        config = config_module.load(["split.source=generated",
                                     f"split.generate_scheme={scheme}"])
        if not os.path.exists(config["data"]["cache_h5ad"]):
            pytest.skip("cache not built")
        for i, fold in enumerate(splits.folds(config)):
            assert not set(fold["train"]) & set(fold["test"]), f"{scheme} fold {i}"


# ---------------------------------------------------------------- data agreement
def test_every_split_condition_exists_in_the_data(config):
    """A condition named by the split but absent from the cache is silently dropped."""
    cache = config["data"]["cache_h5ad"]
    if not os.path.exists(cache):
        pytest.skip(f"{cache} not built")
    import anndata as ad
    conditions = set(ad.read_h5ad(cache, backed="r").obs["condition"].astype(str))
    for method in ("additive", "combinations"):
        for i, fold in enumerate(splits.folds(config, method)):
            for key in ("train", "test"):
                missing = set(fold[key]) - conditions
                assert not missing, f"{method} fold {i} {key} names absent conditions: {missing}"
