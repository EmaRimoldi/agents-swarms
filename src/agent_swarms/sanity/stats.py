"""Estimators for the §9.6 sanity experiment.

Primary observable: G^W := H(p_A) - H(p_B) with Miller-Madow
bias correction, where p_A is the mode distribution elicited with
W absent and p_B the one elicited with W present. Sign is
retained.

Secondary diagnostics: total variation TV(p_A, p_B) and KL
divergence KL(p_A || p_B) with Laplace smoothing.

All estimators operate on per-proposal label lists so bootstrap
resampling is statistically clean (no within-call correlation,
per the Q6 correction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ── Entropy estimator ────────────────────────────────────────────

def _counts_from_labels(labels: Sequence[str], modes: Sequence[str]) -> np.ndarray:
    idx = {m: i for i, m in enumerate(modes)}
    c = np.zeros(len(modes), dtype=np.int64)
    for l in labels:
        c[idx[l if l in idx else modes[-1]]] += 1  # assume last mode is 'other'
    return c


def entropy_mm(counts: np.ndarray) -> float:
    """Shannon entropy in nats with Miller-Madow bias correction.

    H_MM = H_plugin + (K_nonzero - 1) / (2 N).
    """
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    nz = p > 0
    h_plugin = float(-(p[nz] * np.log(p[nz])).sum())
    k_nz = int(nz.sum())
    return h_plugin + (k_nz - 1) / (2.0 * n)


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def kl_smoothed(p_counts: np.ndarray, q_counts: np.ndarray,
                alpha: float | None = None) -> float:
    """KL(p || q) in nats, Laplace-smoothed on both sides."""
    k = len(p_counts)
    if alpha is None:
        alpha = 1.0 / max(p_counts.sum() + q_counts.sum(), 1)
    p = (p_counts + alpha) / (p_counts.sum() + alpha * k)
    q = (q_counts + alpha) / (q_counts.sum() + alpha * k)
    return float((p * (np.log(p) - np.log(q))).sum())


# ── Bootstrap on G^W ─────────────────────────────────────────────

@dataclass(frozen=True)
class SanityResult:
    modes: tuple[str, ...]
    n_without: int
    n_with: int
    p_without: tuple[float, ...]
    p_with: tuple[float, ...]
    H_without: float
    H_with: float
    G_W_point: float
    G_W_ci_low: float
    G_W_ci_high: float
    TV_point: float
    KL_point: float
    bootstrap_n: int
    verdict: str          # PASS / MARGIN / FAIL


def bootstrap_g_w(
    labels_without: Sequence[str],
    labels_with: Sequence[str],
    modes: Sequence[str],
    n_resamples: int = 2000,
    margin_threshold_nats: float = 0.1,
    seed: int = 20260408,
) -> SanityResult:
    """Compute G^W with bootstrap 95% CI and render pass/margin/fail.

    Sign is retained. PASS if CI excludes zero and
    |lower_bound|>threshold; MARGIN if CI excludes zero but
    magnitude smaller; FAIL if CI includes zero.
    """
    rng = np.random.default_rng(seed)
    modes = tuple(modes)
    lab_a = list(labels_without)
    lab_b = list(labels_with)
    n_a = len(lab_a)
    n_b = len(lab_b)

    c_a = _counts_from_labels(lab_a, modes)
    c_b = _counts_from_labels(lab_b, modes)
    h_a = entropy_mm(c_a)
    h_b = entropy_mm(c_b)
    g_point = h_a - h_b

    p_a = c_a / max(n_a, 1)
    p_b = c_b / max(n_b, 1)
    tv = tv_distance(p_a, p_b)
    kl = kl_smoothed(c_a, c_b)

    idx_a = np.arange(n_a)
    idx_b = np.arange(n_b)
    lab_a_arr = np.array(lab_a)
    lab_b_arr = np.array(lab_b)
    draws = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        sa = rng.choice(idx_a, size=n_a, replace=True)
        sb = rng.choice(idx_b, size=n_b, replace=True)
        ca = _counts_from_labels(lab_a_arr[sa].tolist(), modes)
        cb = _counts_from_labels(lab_b_arr[sb].tolist(), modes)
        draws[b] = entropy_mm(ca) - entropy_mm(cb)

    lo = float(np.quantile(draws, 0.025))
    hi = float(np.quantile(draws, 0.975))

    excludes_zero = (lo > 0) or (hi < 0)
    magnitude_lb = min(abs(lo), abs(hi)) if excludes_zero else 0.0
    if excludes_zero and magnitude_lb > margin_threshold_nats:
        verdict = "PASS"
    elif excludes_zero:
        verdict = "MARGIN"
    else:
        verdict = "FAIL"

    return SanityResult(
        modes=modes,
        n_without=n_a,
        n_with=n_b,
        p_without=tuple(float(x) for x in p_a),
        p_with=tuple(float(x) for x in p_b),
        H_without=h_a,
        H_with=h_b,
        G_W_point=g_point,
        G_W_ci_low=lo,
        G_W_ci_high=hi,
        TV_point=tv,
        KL_point=kl,
        bootstrap_n=n_resamples,
        verdict=verdict,
    )


def render_report(res: SanityResult) -> str:
    lines = [
        "# §9.6 Sanity experiment — result",
        "",
        f"Modes ({len(res.modes)}): {', '.join(res.modes)}",
        f"N without W: {res.n_without}   N with W: {res.n_with}",
        "",
        "## Primary observable",
        f"G^W = H(p_without) - H(p_with) = **{res.G_W_point:+.4f} nats**",
        f"95% bootstrap CI: [{res.G_W_ci_low:+.4f}, {res.G_W_ci_high:+.4f}]  "
        f"(B = {res.bootstrap_n})",
        f"Verdict: **{res.verdict}**",
        "",
        "## Entropies",
        f"H(p_without) = {res.H_without:.4f} nats   H(p_with) = {res.H_with:.4f} nats",
        "",
        "## Secondary diagnostics",
        f"TV(p_without, p_with) = {res.TV_point:.4f}",
        f"KL(p_without || p_with) [Laplace-smoothed] = {res.KL_point:.4f} nats",
        "",
        "## Per-mode frequencies",
        "| mode | p_without | p_with | Δ |",
        "|---|---|---|---|",
    ]
    for m, pa, pb in zip(res.modes, res.p_without, res.p_with):
        lines.append(f"| {m} | {pa:.3f} | {pb:.3f} | {pb - pa:+.3f} |")
    return "\n".join(lines)
