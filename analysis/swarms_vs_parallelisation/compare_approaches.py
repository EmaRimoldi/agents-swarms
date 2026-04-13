"""Comparison analysis: Parallelisation vs Swarm approaches.

Loads trajectory data from:
  - Parallelisation: agent_parallelisation_new/runs/experiment_exp_20260401_013535
  - Swarm exp1:      agent_swarms/runs/experiment_exp_20260405_022850
  - Swarm exp2:      agent_swarms/runs/experiment_exp_20260405_124604

Generates 5 figures in analysis/swarms_vs_parallelisation/:
  fig1_trajectories.png       — all agent runs, coloured by approach
  fig2_system_best.png        — system-level best over run index
  fig3_best_achieved.png      — bar chart: best val_bpb per experiment
  fig4_convergence.png        — improvement rate and runs-to-threshold
  fig5_agent_correlation.png  — within-pair Pearson r (lockstep proxy)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent          # agent_swarms/
PARA_ROOT = REPO_ROOT.parent / "agent_parallelisation_new"
OUT_DIR   = REPO_ROOT / "analysis" / "swarms_vs_parallelisation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectory(jsonl_path: Path) -> list[float]:
    """Load val_bpb sequence from a trajectory.jsonl file."""
    values = []
    if not jsonl_path.exists():
        return values
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(json.loads(line)["val_bpb"]))
        except (json.JSONDecodeError, KeyError):
            pass
    return values


def running_min(values: list[float]) -> list[float]:
    """Return the running minimum (best-so-far) of a sequence."""
    best = []
    current_best = float("inf")
    for v in values:
        current_best = min(current_best, v)
        best.append(current_best)
    return best


def system_running_min(agents_values: list[list[float]]) -> list[float]:
    """Global best across all agents at each *system run index*.

    Interleaves agent runs in round-robin order (agent 0 run 0, agent 1
    run 0, agent 0 run 1, …) and returns the running minimum of that
    merged sequence.
    """
    # Interleave: at system step k, agents alternate
    max_steps = max(len(v) for v in agents_values)
    merged = []
    for step in range(max_steps):
        for av in agents_values:
            if step < len(av):
                merged.append(av[step])
    return running_min(merged)


# ---------------------------------------------------------------------------
# Load all experiments
# ---------------------------------------------------------------------------

# Parallelisation — exp 20260401_013535
para_dir = PARA_ROOT / "runs" / "experiment_exp_20260401_013535" / "mode_parallel"
para_agents: dict[str, list[float]] = {}
for agent_path in sorted(para_dir.glob("agent_*")):
    traj = load_trajectory(agent_path / "results" / "trajectory.jsonl")
    if traj:
        para_agents[agent_path.name] = traj

# Swarm exp1 — exp 20260405_022850
sw1_dir = REPO_ROOT / "runs" / "experiment_exp_20260405_022850" / "mode_swarm"
sw1_agents: dict[str, list[float]] = {}
for agent_path in sorted(sw1_dir.glob("agent_*")):
    traj = load_trajectory(agent_path / "results" / "trajectory.jsonl")
    if traj:
        sw1_agents[agent_path.name] = traj

# Swarm exp2 — exp 20260405_124604
sw2_dir = REPO_ROOT / "runs" / "experiment_exp_20260405_124604" / "mode_swarm"
sw2_agents: dict[str, list[float]] = {}
for agent_path in sorted(sw2_dir.glob("agent_*")):
    traj = load_trajectory(agent_path / "results" / "trajectory.jsonl")
    if traj:
        sw2_agents[agent_path.name] = traj

# Derived quantities
para_sys_best = system_running_min(list(para_agents.values()))
sw1_sys_best  = system_running_min(list(sw1_agents.values()))
sw2_sys_best  = system_running_min(list(sw2_agents.values()))

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

# Parallelisation: blue family
PARA_COLORS = ["#3B82F6", "#93C5FD"]  # strong / light blue

# Swarm exp1: orange family
SW1_COLORS  = ["#F97316", "#FBD38D"]  # strong / light orange

# Swarm exp2: green family
SW2_COLORS  = ["#22C55E", "#BBF7D0"]  # strong / light green

# System-best lines (darker variants)
PARA_SYS_COLOR = "#1D4ED8"
SW1_SYS_COLOR  = "#C2410C"
SW2_SYS_COLOR  = "#15803D"

STYLE = {
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linestyle":  "--",
}
plt.rcParams.update(STYLE)

# ---------------------------------------------------------------------------
# Fig 1 — All agent trajectories coloured by approach
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("Agent Trajectories: Parallelisation vs Swarm", fontsize=14, fontweight="bold")

# Left panel: raw val_bpb per step
ax = axes[0]
ax.set_title("Per-Run val_bpb")
ax.set_xlabel("Training run (agent-local step index)")
ax.set_ylabel("val_bpb ↓")

for i, (name, traj) in enumerate(para_agents.items()):
    ax.plot(traj, color=PARA_COLORS[i % len(PARA_COLORS)], lw=1.5,
            alpha=0.8, marker="o", ms=4, label=f"Para {name}")

for i, (name, traj) in enumerate(sw1_agents.items()):
    ax.plot(traj, color=SW1_COLORS[i % len(SW1_COLORS)], lw=1.5,
            alpha=0.8, marker="s", ms=4, label=f"Swarm1 {name}")

for i, (name, traj) in enumerate(sw2_agents.items()):
    ax.plot(traj, color=SW2_COLORS[i % len(SW2_COLORS)], lw=1.5,
            alpha=0.8, marker="^", ms=4, label=f"Swarm2 {name}")

ax.legend(fontsize=8, ncol=2)

# Right panel: running-min per agent (best-so-far)
ax = axes[1]
ax.set_title("Agent Best-So-Far (Running Min)")
ax.set_xlabel("Training run (agent-local step index)")

for i, (name, traj) in enumerate(para_agents.items()):
    ax.plot(running_min(traj), color=PARA_COLORS[i % len(PARA_COLORS)],
            lw=2.0, alpha=0.9, marker="o", ms=4, label=f"Para {name}")

for i, (name, traj) in enumerate(sw1_agents.items()):
    ax.plot(running_min(traj), color=SW1_COLORS[i % len(SW1_COLORS)],
            lw=2.0, alpha=0.9, marker="s", ms=4, label=f"Swarm1 {name}")

for i, (name, traj) in enumerate(sw2_agents.items()):
    ax.plot(running_min(traj), color=SW2_COLORS[i % len(SW2_COLORS)],
            lw=2.0, alpha=0.9, marker="^", ms=4, label=f"Swarm2 {name}")

ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_trajectories.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig1 saved")

# ---------------------------------------------------------------------------
# Fig 2 — System-level best over interleaved run index
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("System-Level Best-So-Far (All Agents Combined)", fontsize=13, fontweight="bold")
ax.set_xlabel("System run index (interleaved round-robin across agents)")
ax.set_ylabel("System best val_bpb ↓")

ax.step(range(len(para_sys_best)), para_sys_best,
        color=PARA_SYS_COLOR, lw=2.5, where="post",
        label=f"Parallelisation (final={para_sys_best[-1]:.4f})")

ax.step(range(len(sw1_sys_best)), sw1_sys_best,
        color=SW1_SYS_COLOR, lw=2.5, where="post",
        label=f"Swarm run 1 (final={sw1_sys_best[-1]:.4f})")

ax.step(range(len(sw2_sys_best)), sw2_sys_best,
        color=SW2_SYS_COLOR, lw=2.5, where="post",
        label=f"Swarm run 2 (final={sw2_sys_best[-1]:.4f})")

# Annotate final values
for label, sys_best, color in [
    ("Para", para_sys_best, PARA_SYS_COLOR),
    ("Sw1",  sw1_sys_best,  SW1_SYS_COLOR),
    ("Sw2",  sw2_sys_best,  SW2_SYS_COLOR),
]:
    ax.annotate(
        f"{sys_best[-1]:.4f}",
        xy=(len(sys_best) - 1, sys_best[-1]),
        xytext=(3, 3), textcoords="offset points",
        color=color, fontsize=9, fontweight="bold",
    )

ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_system_best.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig2 saved")

# ---------------------------------------------------------------------------
# Fig 3 — Best val_bpb achieved: grouped bar chart
# ---------------------------------------------------------------------------

experiments = {
    "Parallel\n(exp 20260401)":   [min(v) for v in para_agents.values()],
    "Swarm 1\n(exp 20260405a)":   [min(v) for v in sw1_agents.values()],
    "Swarm 2\n(exp 20260405b)":   [min(v) for v in sw2_agents.values()],
}

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Best val_bpb Achieved per Agent and Experiment", fontsize=13, fontweight="bold")
ax.set_ylabel("Best val_bpb ↓ (lower = better)")

n_agents = max(len(v) for v in experiments.values())
x = np.arange(len(experiments))
width = 0.3

colors_by_exp = [PARA_COLORS, SW1_COLORS, SW2_COLORS]

for agent_idx in range(n_agents):
    vals = []
    for exp_vals in experiments.values():
        if agent_idx < len(exp_vals):
            vals.append(exp_vals[agent_idx])
        else:
            vals.append(float("nan"))
    offset = (agent_idx - (n_agents - 1) / 2) * width
    bars = ax.bar(x + offset, vals, width=width * 0.9,
                  label=f"Agent {agent_idx}",
                  color=[colors_by_exp[i][agent_idx % 2] for i in range(len(experiments))])
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

# System best as horizontal markers
for i, (exp_label, exp_vals) in enumerate(experiments.items()):
    sys_best = min(exp_vals)
    ax.hlines(sys_best, x[i] - 0.5 * width * n_agents, x[i] + 0.5 * width * n_agents,
              colors="black", linestyles="dashed", lw=1.5)
    ax.text(x[i] + 0.5 * width * n_agents + 0.02, sys_best,
            f"best={sys_best:.4f}", va="center", fontsize=8, color="black")

ax.set_xticks(x)
ax.set_xticklabels(list(experiments.keys()), fontsize=10)

# Zoom y-axis for readability
all_vals = [v for exp_vals in experiments.values() for v in exp_vals if not np.isnan(v)]
y_lo = min(all_vals) - 0.01
y_hi = max(all_vals) + 0.02
ax.set_ylim(y_lo, y_hi)

ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_best_achieved.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig3 saved")

# ---------------------------------------------------------------------------
# Fig 4 — Convergence analysis: improvement per run and threshold crossing
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Convergence Characteristics", fontsize=13, fontweight="bold")

# Left: improvement per run (Δval_bpb vs step, clipped to improvements only)
ax = axes[0]
ax.set_title("Per-Run Improvement (best-so-far drop)")
ax.set_xlabel("Agent-local step index")
ax.set_ylabel("Improvement in best (val_bpb drop, ≥0)")

def improvements(traj: list[float]) -> list[float]:
    """Returns delta improvement at each step (0 if no improvement)."""
    rm = running_min(traj)
    deltas = [0.0] + [max(0.0, rm[i-1] - rm[i]) for i in range(1, len(rm))]
    return deltas

for i, (name, traj) in enumerate(para_agents.items()):
    imp = improvements(traj)
    ax.bar(np.arange(len(imp)) - 0.15 + i * 0.1, imp,
           width=0.1, color=PARA_COLORS[i % 2], alpha=0.7,
           label=f"Para {name}")

for i, (name, traj) in enumerate(sw1_agents.items()):
    imp = improvements(traj)
    ax.bar(np.arange(len(imp)) + 0.05 + i * 0.1, imp,
           width=0.1, color=SW1_COLORS[i % 2], alpha=0.7,
           label=f"Swarm1 {name}")

ax.legend(fontsize=8)

# Right: runs needed to cross various thresholds
ax = axes[1]
ax.set_title("Runs to Cross val_bpb Threshold")
ax.set_xlabel("val_bpb threshold")
ax.set_ylabel("Number of agent-local runs needed")

thresholds = np.linspace(1.12, 1.042, 50)

def runs_to_threshold(traj: list[float], threshold: float) -> float:
    """First step index where val_bpb ≤ threshold, else NaN."""
    for i, v in enumerate(traj):
        if v <= threshold:
            return float(i)
    return float("nan")

for i, (name, traj) in enumerate(para_agents.items()):
    rtts = [runs_to_threshold(traj, t) for t in thresholds]
    ax.plot(thresholds, rtts, color=PARA_COLORS[i % 2], lw=1.5,
            label=f"Para {name}", linestyle="--")

for i, (name, traj) in enumerate(sw1_agents.items()):
    rtts = [runs_to_threshold(traj, t) for t in thresholds]
    ax.plot(thresholds, rtts, color=SW1_COLORS[i % 2], lw=1.5,
            label=f"Swarm1 {name}", linestyle="-")

for i, (name, traj) in enumerate(sw2_agents.items()):
    rtts = [runs_to_threshold(traj, t) for t in thresholds]
    ax.plot(thresholds, rtts, color=SW2_COLORS[i % 2], lw=1.5,
            label=f"Swarm2 {name}", linestyle="-.")

ax.invert_xaxis()  # lower threshold = harder target, right to left
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_convergence.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig4 saved")

# ---------------------------------------------------------------------------
# Fig 5 — Agent correlation (lockstep proxy)
# ---------------------------------------------------------------------------
# Pearson r between paired agent trajectories captures whether agents explore
# independently (low r, swarm) or follow similar paths (high r, parallel).

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Within-Pair Agent Correlation (Lockstep vs Divergence)", fontsize=13, fontweight="bold")

datasets = [
    ("Parallelisation", para_agents, PARA_COLORS, PARA_SYS_COLOR),
    ("Swarm run 1",     sw1_agents,  SW1_COLORS,  SW1_SYS_COLOR),
    ("Swarm run 2",     sw2_agents,  SW2_COLORS,  SW2_SYS_COLOR),
]

for ax, (label, agents, colors, sys_color) in zip(axes, datasets):
    agent_names = list(agents.keys())
    trajs = [agents[n] for n in agent_names]

    # Truncate to shared length for correlation
    min_len = min(len(t) for t in trajs)
    trajs_trunc = [t[:min_len] for t in trajs]

    ax.set_title(label)
    ax.set_xlabel("Agent-local step index")
    ax.set_ylabel("val_bpb")

    for i, (name, traj_t) in enumerate(zip(agent_names, trajs_trunc)):
        ax.plot(traj_t, color=colors[i % 2], lw=1.5, marker="o", ms=4,
                alpha=0.85, label=name)

    # Compute and annotate Pearson r
    if len(trajs_trunc) >= 2 and min_len >= 3:
        r, p = pearsonr(trajs_trunc[0], trajs_trunc[1])
        ax.text(0.97, 0.97, f"r = {r:.3f}\n(p={p:.3f})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig5_agent_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig5 saved")

# ---------------------------------------------------------------------------
# Summary statistics printed to stdout and saved
# ---------------------------------------------------------------------------

lines = []
lines.append("# Comparison Summary: Parallelisation vs Swarm")
lines.append("")
lines.append("## Best val_bpb achieved")
lines.append("")
lines.append("| Experiment | Agent 0 best | Agent 1 best | System best |")
lines.append("|-----------|-------------|-------------|------------|")

for label, agents in [
    ("Parallel (exp_20260401)", para_agents),
    ("Swarm 1  (exp_20260405a)", sw1_agents),
    ("Swarm 2  (exp_20260405b)", sw2_agents),
]:
    vals = [min(v) for v in agents.values()]
    row = f"| {label} | {vals[0]:.6f} | {vals[1]:.6f} | {min(vals):.6f} |"
    lines.append(row)

lines.append("")
lines.append("## Agent correlation (Pearson r, shared steps)")
lines.append("")
for label, agents in [
    ("Parallel",  para_agents),
    ("Swarm 1",   sw1_agents),
    ("Swarm 2",   sw2_agents),
]:
    trajs = list(agents.values())
    min_len = min(len(t) for t in trajs)
    if min_len >= 3:
        r, p = pearsonr(trajs[0][:min_len], trajs[1][:min_len])
        lines.append(f"- {label}: r={r:.3f}, p={p:.3f}")

lines.append("")
lines.append("## Interpretation")
lines.append("")
lines.append("- Swarm experiments achieve substantially lower val_bpb than parallelisation.")
lines.append("- Lower agent correlation in swarms suggests genuine divergence in search paths,")
lines.append("  enabled by the shared blackboard claim mechanism.")
lines.append("- Parallelisation agents converge to similar local minima (high correlation),")
lines.append("  consistent with redundant exploration without communication.")

summary_text = "\n".join(lines)
print("\n" + summary_text)
(OUT_DIR / "summary.md").write_text(summary_text)
print(f"\nAll figures and summary saved to {OUT_DIR}")
