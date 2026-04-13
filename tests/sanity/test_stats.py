"""Unit tests for the §9.6 stats module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent_swarms.sanity.stats import (  # noqa: E402
    bootstrap_g_w,
    entropy_mm,
    kl_smoothed,
    tv_distance,
)


def test_entropy_mm_uniform_on_k():
    counts = np.array([100, 100, 100, 100])
    h = entropy_mm(counts)
    # plugin uniform entropy on 4 modes = log 4
    assert abs(h - (np.log(4) + 3 / 800)) < 1e-12


def test_entropy_mm_point_mass():
    counts = np.array([500, 0, 0, 0])
    h = entropy_mm(counts)
    assert h == 0.0  # single nonzero bin, plugin 0, correction (1-1)/2N = 0


def test_tv_distance_identity():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert tv_distance(p, p) == 0.0
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert abs(tv_distance(p, q) - 0.75) < 1e-12


def test_kl_smoothed_nonnegative():
    p = np.array([50, 30, 10, 10])
    q = np.array([10, 20, 30, 40])
    kl = kl_smoothed(p, q)
    assert kl > 0


def test_bootstrap_pass_on_strong_effect():
    rng = np.random.default_rng(0)
    modes = ["a", "b", "c", "d"]
    lab_a = rng.choice(modes, size=400, p=[0.25] * 4).tolist()
    lab_b = rng.choice(modes, size=400, p=[0.85, 0.05, 0.05, 0.05]).tolist()
    res = bootstrap_g_w(lab_a, lab_b, modes, n_resamples=500, seed=0)
    assert res.verdict == "PASS"
    assert res.G_W_point > 0
    assert res.G_W_ci_low > 0


def test_bootstrap_fail_on_null_effect():
    rng = np.random.default_rng(1)
    modes = ["a", "b", "c", "d"]
    lab_a = rng.choice(modes, size=400, p=[0.25] * 4).tolist()
    lab_b = rng.choice(modes, size=400, p=[0.25] * 4).tolist()
    res = bootstrap_g_w(lab_a, lab_b, modes, n_resamples=500, seed=1)
    assert res.verdict == "FAIL"


def test_bootstrap_negative_g_w_also_passes():
    rng = np.random.default_rng(2)
    modes = ["a", "b", "c", "d"]
    lab_a = rng.choice(modes, size=400, p=[0.85, 0.05, 0.05, 0.05]).tolist()
    lab_b = rng.choice(modes, size=400, p=[0.25] * 4).tolist()
    res = bootstrap_g_w(lab_a, lab_b, modes, n_resamples=500, seed=2)
    assert res.verdict == "PASS"
    assert res.G_W_point < 0
    assert res.G_W_ci_high < 0
