#!/usr/bin/env python3
"""Offline dry-run for the §9.6 sanity experiment.

Does NOT call the Anthropic API. Instead:
  1. Runs select_w() against the real source run (writes W_t payload).
  2. Renders side-by-side prompts and prints them.
  3. Synthesizes two mock label sequences drawn from hand-specified
     mode distributions, runs the bootstrap_g_w estimator, and prints
     the rendered report. This verifies the estimator and the
     PASS/MARGIN/FAIL decision logic on known inputs before any
     real API calls are made.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_swarms.sanity.select_w import select_w                  # noqa: E402
from agent_swarms.sanity.run_with_without import render_side_by_side  # noqa: E402
from agent_swarms.sanity.stats import bootstrap_g_w, render_report  # noqa: E402


TAXONOMY_COARSE = [
    "optimizer", "lr_schedule", "architecture",
    "regularization", "data_pipeline", "init", "other",
]


def sample_labels(dist: dict[str, float], n: int, rng: np.random.Generator) -> list[str]:
    modes = list(dist.keys())
    probs = np.array([dist[m] for m in modes], dtype=float)
    probs = probs / probs.sum()
    idx = rng.choice(len(modes), size=n, p=probs)
    return [modes[i] for i in idx]


def main() -> int:
    print("=" * 72)
    print("1. Selecting W_t from source swarm run")
    print("=" * 72)
    source_run = REPO_ROOT / "runs" / "experiment_exp_20260406_024115"
    out_dir = REPO_ROOT / "runs" / "sanity" / "dry_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = select_w(source_run=source_run, out_dir=out_dir, repo_root=REPO_ROOT)
    print(f"source_run         : {sel.source_run}")
    print(f"cut_index          : {sel.cut_index}")
    print(f"cut_timestamp      : {sel.cut_timestamp}")
    print(f"n_entries_in_slice : {sel.n_entries_in_slice}")
    print(f"n_results_in_slice : {sel.n_results_in_slice}")
    print(f"payload_bytes      : {sel.payload_bytes}")
    print(f"payload_path       : {sel.payload_path}")
    w_payload = sel.payload_path.read_text(encoding="utf-8")
    print()
    print("--- first 800 chars of W_t payload ---")
    print(w_payload[:800])
    print("--- (truncated) ---")
    print()

    print("=" * 72)
    print("2. Side-by-side prompts that will be sent when --execute is used")
    print("=" * 72)
    print(render_side_by_side(w_payload))

    print("=" * 72)
    print("3a. Mock-data sanity case: clear concentration effect (PASS expected)")
    print("=" * 72)
    rng = np.random.default_rng(42)
    # Without W: near-uniform
    dist_a = {m: 1.0 / 7 for m in TAXONOMY_COARSE}
    # With W: concentrated on 'optimizer' and 'lr_schedule'
    dist_b = {
        "optimizer": 0.45, "lr_schedule": 0.35, "architecture": 0.05,
        "regularization": 0.05, "data_pipeline": 0.05, "init": 0.03, "other": 0.02,
    }
    labels_a = sample_labels(dist_a, 300, rng)
    labels_b = sample_labels(dist_b, 300, rng)
    res = bootstrap_g_w(labels_a, labels_b, TAXONOMY_COARSE, n_resamples=2000, seed=1)
    print(render_report(res))
    assert res.verdict == "PASS", f"expected PASS, got {res.verdict}"
    print()

    print("=" * 72)
    print("3b. Mock-data sanity case: no effect (FAIL expected)")
    print("=" * 72)
    labels_a2 = sample_labels(dist_a, 300, rng)
    labels_b2 = sample_labels(dist_a, 300, rng)
    res2 = bootstrap_g_w(labels_a2, labels_b2, TAXONOMY_COARSE, n_resamples=2000, seed=2)
    print(render_report(res2))
    assert res2.verdict == "FAIL", f"expected FAIL, got {res2.verdict}"
    print()

    print("=" * 72)
    print("3c. Mock-data sanity case: negative G^W (brainstorm expander, PASS expected)")
    print("=" * 72)
    # Without W: concentrated
    dist_c = {
        "optimizer": 0.7, "lr_schedule": 0.15, "architecture": 0.05,
        "regularization": 0.04, "data_pipeline": 0.03, "init": 0.02, "other": 0.01,
    }
    # With W: near-uniform (W disperses the prior)
    labels_a3 = sample_labels(dist_c, 300, rng)
    labels_b3 = sample_labels(dist_a, 300, rng)
    res3 = bootstrap_g_w(labels_a3, labels_b3, TAXONOMY_COARSE, n_resamples=2000, seed=3)
    print(render_report(res3))
    assert res3.G_W_point < 0, "expected negative G^W"
    assert res3.verdict == "PASS", f"expected PASS, got {res3.verdict}"
    print()

    print("=" * 72)
    print("3d. Mock-data sanity case: MARGIN (small but resolvable effect)")
    print("=" * 72)
    dist_d = {
        "optimizer": 0.18, "lr_schedule": 0.17, "architecture": 0.14,
        "regularization": 0.14, "data_pipeline": 0.14, "init": 0.12, "other": 0.11,
    }
    labels_a4 = sample_labels(dist_a, 2000, rng)
    labels_b4 = sample_labels(dist_d, 2000, rng)
    res4 = bootstrap_g_w(labels_a4, labels_b4, TAXONOMY_COARSE, n_resamples=2000, seed=4)
    print(render_report(res4))
    print(f"(verdict={res4.verdict} — demonstrates MARGIN/PASS boundary behavior)")
    print()

    print("All dry-run checks passed. No API calls were made.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
