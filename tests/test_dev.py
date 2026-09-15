"""Development tooling: the coupling's fallback accounting, the experiment planner
and the decision rule.

    python -m pytest tests/test_dev.py -v
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import dev_queue  # noqa: E402
import dev_score  # noqa: E402
from src.train.coupling import sample_pairs  # noqa: E402


# ---------------------------------------------------------------- coupling fallback
def test_a_healthy_plan_is_counted_without_a_fallback():
    torch.manual_seed(0)
    source, target = torch.randn(16, 4), torch.randn(16, 4)
    stats: dict = {}
    sample_pairs(source, target, "uot", 0.05, 1.0, np.random.default_rng(0), stats=stats)
    assert stats == {"plans": 1, "fallbacks": 0}


def test_a_degenerate_plan_is_counted_as_a_fallback():
    """Equal costs at a vanishing entropy underflow Sinkhorn to no mass at all."""
    torch.manual_seed(0)
    source = torch.zeros(16, 4)
    target = torch.ones(16, 4)
    stats: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z0p, z1p = sample_pairs(source, target, "uot", 1e-12, 1.0,
                                np.random.default_rng(0), stats=stats)
    assert stats == {"plans": 1, "fallbacks": 1}
    assert z0p.shape == source.shape and z1p.shape == target.shape


def test_sample_pairs_still_works_without_stats():
    torch.manual_seed(0)
    z0p, _ = sample_pairs(torch.randn(8, 3), torch.randn(8, 3), "uot", 0.05, 1.0,
                          np.random.default_rng(0))
    assert z0p.shape == (8, 3)


# ---------------------------------------------------------------- planner
ARMS = {"base": "", "ot": "--coupling ot", "sb": "--generator shared_basis --private-rank 8"}


def plan(**overrides):
    kwargs = dict(name="t", datasets=["norman", "combosciplex"], arms=ARMS,
                  seeds=[0, 1, 10], gpus=[0, 1])
    kwargs.update(overrides)
    return dev_queue.plan_jobs(**kwargs)


def test_tags_are_unique_and_never_prefix_another_runs_directory():
    tags = [job["tag"] for job in plan()]
    assert len(tags) == len(set(tags))
    for a in tags:
        for b in tags:
            if a != b:
                assert not b.startswith(a + "_"), (a, b)


def test_one_encoder_per_validation_set_and_every_arm_waits_for_its_own():
    jobs = plan()
    encoders = {job["tag"]: job for job in jobs if job["kind"] == "encoder"}
    assert sorted(job["group"] for job in encoders.values()) == ["cv0", "cv1", "cv2", "nval"]
    for job in (j for j in jobs if j["kind"] == "arm"):
        assert job["encoder"] in encoders
        assert encoders[job["encoder"]]["group"] == job["group"]


def test_encoders_train_stage1_only():
    for job in (j for j in plan() if j["kind"] == "encoder"):
        flags = job["flags"]
        i = flags.index("--init-vae-from")
        assert flags[i + 1] == ""
        assert flags[flags.index("--stage2") + 1] == "0"
        assert "--lr-cosine" not in flags


def test_extra_flags_cannot_override_the_encoder_stage2():
    for job in (j for j in plan(extra=["--stage2", "3"]) if j["kind"] == "encoder"):
        flags = job["flags"]
        assert flags[len(flags) - 1 - flags[::-1].index("--stage2") + 1] == "0"


def test_non_affine_arms_drop_generator_rank():
    for job in plan():
        if job["arm"] == "sb":
            assert "--generator-rank" not in job["flags"]
        elif job["kind"] == "arm":
            assert "--generator-rank" in job["flags"]


def test_split_flags_select_the_development_sets():
    for job in plan():
        flags = job["flags"]
        assert "--validation" in flags
        if job["dataset"] == "norman":
            assert flags[flags.index("--method") + 1] == "additive"
            assert flags[flags.index("--fold") + 1] == "0"
        else:
            assert flags[flags.index("--val-fold") + 1] == job["group"][2:]


def test_reusing_encoders_plans_no_new_encoder():
    jobs = plan(name="c1", encoder_name="p0")
    assert not [job for job in jobs if job["kind"] == "encoder"]
    assert all(job["encoder"].startswith("p0_enc_") for job in jobs)


def test_names_with_underscores_are_refused():
    with pytest.raises(ValueError):
        plan(arms={"bad_arm": ""})


# ---------------------------------------------------------------- decision rule
def stats(mean, se, n=6):
    return {"n": n, "mean": mean, "sd": se * np.sqrt(n), "se": se}


def test_rule_adopts_a_two_se_gain_that_hurts_nowhere():
    assert dev_score.decide({"norman": stats(-0.05, 0.02),
                             "combosciplex": stats(0.01, 0.02)}) == "ADOPT"


def test_rule_rejects_a_gain_below_two_se():
    assert dev_score.decide({"norman": stats(-0.03, 0.02),
                             "combosciplex": stats(0.0, 0.02)}).startswith("reject")


def test_rule_rejects_a_gain_that_hurts_the_other_dataset():
    assert dev_score.decide({"norman": stats(-0.10, 0.02),
                             "combosciplex": stats(0.03, 0.02)}) == "reject (hurts a dataset)"


def test_rule_waits_for_pairs():
    assert dev_score.decide({"norman": stats(-0.1, float("nan"), n=1)}) == "insufficient pairs"
