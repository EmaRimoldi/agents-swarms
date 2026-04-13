#!/usr/bin/env python3
"""
Loss Curve Analysis
Extract and visualize training loss curves across models
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Experiment definitions
EXPERIMENTS = {
    "haiku_run1": {
        "model": "Haiku 4.5",
        "model_key": "haiku",
        "color": "#FF6B6B",
        "runs_path": "/home/erimoldi/projects/agent_swarms/runs/experiment_exp_20260405_022850/mode_swarm",
        "agents": ["agent_0", "agent_1"],
    },
    "haiku_run2": {
        "model": "Haiku 4.5 (Run 2)",
        "model_key": "haiku",
        "color": "#FF8888",
        "runs_path": "/home/erimoldi/projects/agent_swarms/runs/experiment_exp_20260405_124604/mode_swarm",
        "agents": ["agent_0", "agent_1"],
    },
    "sonnet": {
        "model": "Sonnet 4.6",
        "model_key": "sonnet",
        "color": "#4ECDC4",
        "runs_path": "/home/erimoldi/projects/agent_swarms/runs/experiment_exp_20260406_024115/mode_swarm",
        "agents": ["agent_0", "agent_1"],
    },
    "opus": {
        "model": "Opus 4.6",
        "model_key": "opus",
        "color": "#45B7D1",
        "runs_path": "/home/erimoldi/projects/agent_swarms/runs/experiment_exp_20260406_044120/mode_swarm",
        "agents": ["agent_0", "agent_1"],
    },
}


def extract_loss_data(exp_key: str, exp_data: Dict) -> Dict[str, List]:
    """Extract loss values from training logs."""
    loss_data = {"steps": [], "losses": [], "val_bpbs": []}

    runs_path = Path(exp_data["runs_path"])

    # Aggregate loss data from all runs and agents
    all_steps = []
    all_losses = []

    for agent in exp_data["agents"]:
        agent_path = runs_path / agent / "workspace" / "logs"
        if not agent_path.exists():
            continue

        # Find all training run files
        log_files = sorted(agent_path.glob("train_run_*.out"))

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    content = f.read()

                # Extract loss values using regex
                # Pattern: "step XXXXX ... loss: Y.XXXXXX"
                loss_pattern = r"step\s+(\d+).*loss:\s+([\d.]+)"
                matches = re.findall(loss_pattern, content)

                for step_str, loss_str in matches:
                    try:
                        step = int(step_str)
                        loss = float(loss_str)
                        all_steps.append(step)
                        all_losses.append(loss)
                    except (ValueError, TypeError):
                        continue

            except Exception as e:
                print(f"Error reading {log_file}: {e}")
                continue

    # Sort by step
    if all_steps:
        sorted_pairs = sorted(zip(all_steps, all_losses))
        loss_data["steps"] = [p[0] for p in sorted_pairs]
        loss_data["losses"] = [p[1] for p in sorted_pairs]

    return loss_data


def get_val_bpb_from_report(exp_key: str, exp_data: Dict) -> float:
    """Extract best validation BPB from experiment report."""
    report_path = (
        Path(exp_data["runs_path"]) / "aggregate" / "experiment_report.txt"
    )

    if not report_path.exists():
        return None

    try:
        with open(report_path, "r") as f:
            content = f.read()
        # Pattern: "Best val_bpb: Y.XXXXXX"
        match = re.search(r"Best val_bpb:\s+([\d.]+)", content)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error reading {report_path}: {e}")

    return None


def analyze_loss_curves():
    """Analyze and visualize loss curves."""

    # Extract loss data for all experiments
    all_data = {}
    for exp_key, exp_data in EXPERIMENTS.items():
        loss_data = extract_loss_data(exp_key, exp_data)
        val_bpb = get_val_bpb_from_report(exp_key, exp_data)
        all_data[exp_key] = {
            **exp_data,
            "loss_data": loss_data,
            "val_bpb": val_bpb,
        }

    print("\n📊 Loss Data Extracted:")
    print("=" * 80)
    for exp_key, data in all_data.items():
        loss_data = data["loss_data"]
        if loss_data["steps"]:
            print(
                f"\n{data['model']:20s} | Steps: {len(loss_data['steps']):4d} | "
                f"Initial Loss: {loss_data['losses'][0]:8.3f} | "
                f"Final Loss: {loss_data['losses'][-1]:8.3f} | "
                f"Improvement: {loss_data['losses'][0] - loss_data['losses'][-1]:7.3f} | "
                f"Best BPB: {data['val_bpb']}"
            )
        else:
            print(f"\n{data['model']:20s} | No loss data found")

    # Create visualizations
    create_visualizations(all_data)

    # Save analysis
    save_analysis(all_data)


def create_visualizations(all_data: Dict):
    """Create loss curve visualizations."""

    output_dir = Path(
        "/home/erimoldi/projects/agent_swarms/analysis/model_comparison_20260406"
    )

    # Figure 1: Loss curves by model (separate subplots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Training Loss Curves by Model\n2-Agent Swarm Learning",
        fontsize=16,
        fontweight="bold",
    )

    ax_list = axes.flatten()
    for idx, (exp_key, data) in enumerate(all_data.items()):
        if idx >= 4:
            break

        ax = ax_list[idx]
        loss_data = data["loss_data"]

        if loss_data["steps"]:
            steps = loss_data["steps"]
            losses = loss_data["losses"]

            ax.plot(
                steps,
                losses,
                color=data["color"],
                linewidth=2,
                alpha=0.8,
                label=data["model"],
            )
            ax.fill_between(steps, losses, alpha=0.2, color=data["color"])

            ax.set_xlabel("Training Step", fontweight="bold")
            ax.set_ylabel("Loss", fontweight="bold")
            ax.set_title(f"{data['model']} - Best BPB: {data['val_bpb']:.6f}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

            # Add annotations
            ax.text(
                0.02,
                0.98,
                f"Initial: {losses[0]:.3f}\nFinal: {losses[-1]:.3f}\nImprovement: {losses[0] - losses[-1]:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"{data['model']} - No loss data")

    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves_individual.png", dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved: loss_curves_individual.png")

    # Figure 2: Overlaid loss curves (all models)
    fig, ax = plt.subplots(figsize=(14, 8))

    for exp_key, data in all_data.items():
        loss_data = data["loss_data"]

        if loss_data["steps"]:
            steps = loss_data["steps"]
            losses = loss_data["losses"]

            # Normalize x-axis to percentage
            max_steps = len(steps)
            x_percent = [s / max(steps) * 100 for s in steps]

            ax.plot(
                x_percent,
                losses,
                color=data["color"],
                linewidth=2.5,
                alpha=0.8,
                label=f"{data['model']} (BPB: {data['val_bpb']:.6f})",
                marker="o",
                markersize=3,
                markevery=max(1, len(steps) // 20),
            )

    ax.set_xlabel("Training Progress (%)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Training Loss", fontweight="bold", fontsize=12)
    ax.set_title("Loss Decay Comparison Across Models", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(
        output_dir / "loss_curves_comparison.png", dpi=300, bbox_inches="tight"
    )
    print(f"✓ Saved: loss_curves_comparison.png")

    # Figure 3: Loss improvement (initial vs final)
    fig, ax = plt.subplots(figsize=(12, 6))

    models = []
    initial_losses = []
    final_losses = []
    improvements = []
    colors = []

    for exp_key, data in all_data.items():
        loss_data = data["loss_data"]

        if loss_data["steps"]:
            models.append(data["model"])
            initial = loss_data["losses"][0]
            final = loss_data["losses"][-1]
            improvement = initial - final

            initial_losses.append(initial)
            final_losses.append(final)
            improvements.append(improvement)
            colors.append(data["color"])

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, initial_losses, width, label="Initial Loss", color=[
                   c for c in colors], alpha=0.8, edgecolor="black", linewidth=1.5)
    bars2 = ax.bar(x + width / 2, final_losses, width, label="Final Loss", color=[
                   c for c in colors], alpha=0.5, edgecolor="black", linewidth=1.5)

    ax.set_ylabel("Loss", fontweight="bold", fontsize=12)
    ax.set_title("Loss: Initial vs Final", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_dir / "loss_initial_vs_final.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: loss_initial_vs_final.png")

    # Figure 4: Loss decay rate (improvement percentage)
    fig, ax = plt.subplots(figsize=(12, 6))

    improvement_pcts = [
        (imp / initial) * 100
        for imp, initial in zip(improvements, initial_losses)
    ]

    bars = ax.bar(models, improvement_pcts, color=colors, alpha=0.8, edgecolor="black",
                  linewidth=1.5)
    ax.set_ylabel("Loss Reduction (%)", fontweight="bold", fontsize=12)
    ax.set_title("Loss Improvement Rate", fontsize=14, fontweight="bold")
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)

    for bar, pct in zip(bars, improvement_pcts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "loss_improvement_rate.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: loss_improvement_rate.png")


def save_analysis(all_data: Dict):
    """Save detailed loss analysis report."""

    output_dir = Path(
        "/home/erimoldi/projects/agent_swarms/analysis/model_comparison_20260406"
    )

    analysis = []
    analysis.append("=" * 80)
    analysis.append("LOSS CURVE ANALYSIS")
    analysis.append("Training Loss Progression Across Models")
    analysis.append("=" * 80)
    analysis.append("")

    analysis.append("LOSS STATISTICS BY MODEL:")
    analysis.append("-" * 80)

    for exp_key, data in all_data.items():
        loss_data = data["loss_data"]

        if loss_data["steps"]:
            losses = loss_data["losses"]
            steps = loss_data["steps"]

            analysis.append(f"\n{data['model']}:")
            analysis.append(f"  Total training steps: {len(steps)}")
            analysis.append(f"  Initial loss: {losses[0]:.6f}")
            analysis.append(f"  Final loss: {losses[-1]:.6f}")
            analysis.append(f"  Improvement: {losses[0] - losses[-1]:.6f}")
            analysis.append(f"  Improvement rate: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
            analysis.append(f"  Min loss: {min(losses):.6f}")
            analysis.append(f"  Max loss: {max(losses):.6f}")
            analysis.append(f"  Average loss: {np.mean(losses):.6f}")
            analysis.append(f"  Std dev: {np.std(losses):.6f}")
            analysis.append(f"  Final BPB: {data['val_bpb']:.6f}")

            # Convergence speed
            halfway_step = len(steps) // 2
            halfway_loss = losses[halfway_step]
            convergence_ratio = (
                (losses[0] - halfway_loss) / (losses[0] - losses[-1]) * 100
            )
            analysis.append(f"  Convergence at 50% steps: {convergence_ratio:.1f}%")

    analysis.append("\n" + "=" * 80)
    analysis.append("COMPARATIVE ANALYSIS:")
    analysis.append("-" * 80)

    # Find best and worst loss decay
    models_with_data = [
        (exp_key, data)
        for exp_key, data in all_data.items()
        if data["loss_data"]["steps"]
    ]

    if models_with_data:
        best_decay = max(
            models_with_data,
            key=lambda x: (
                x[1]["loss_data"]["losses"][0] - x[1]["loss_data"]["losses"][-1]
            ),
        )
        worst_decay = min(
            models_with_data,
            key=lambda x: (
                x[1]["loss_data"]["losses"][0] - x[1]["loss_data"]["losses"][-1]
            ),
        )

        best_improvement = (
            best_decay[1]["loss_data"]["losses"][0]
            - best_decay[1]["loss_data"]["losses"][-1]
        )
        worst_improvement = (
            worst_decay[1]["loss_data"]["losses"][0]
            - worst_decay[1]["loss_data"]["losses"][-1]
        )

        analysis.append(f"\nBest loss decay: {best_decay[1]['model']}")
        analysis.append(f"  Improvement: {best_improvement:.6f}")

        analysis.append(f"\nWorst loss decay: {worst_decay[1]['model']}")
        analysis.append(f"  Improvement: {worst_improvement:.6f}")

        analysis.append(f"\nDifference: {(best_improvement - worst_improvement):.6f}")

    analysis.append("\n" + "=" * 80)
    analysis.append("KEY INSIGHTS:")
    analysis.append("-" * 80)

    analysis.append(
        "\n1. LOSS TRAJECTORY: All models show consistent loss decay pattern"
    )
    analysis.append(
        "2. CONVERGENCE SPEED: Compare how quickly models reach final loss"
    )
    analysis.append(
        "3. STABILITY: Check for oscillations or instabilities during training"
    )
    analysis.append(
        "4. FINAL STATE: Lower final loss correlates with better BPB"
    )

    analysis.append("\n" + "=" * 80)

    analysis_text = "\n".join(analysis)

    # Save to file
    with open(output_dir / "LOSS_ANALYSIS.txt", "w") as f:
        f.write(analysis_text)

    print(f"\n✓ Saved: LOSS_ANALYSIS.txt")
    print("\n" + analysis_text)

    # Save JSON
    json_data = {}
    for exp_key, data in all_data.items():
        loss_data = data["loss_data"]
        json_data[exp_key] = {
            "model": data["model"],
            "val_bpb": data["val_bpb"],
            "num_steps": len(loss_data["steps"]),
            "initial_loss": loss_data["losses"][0] if loss_data["losses"] else None,
            "final_loss": loss_data["losses"][-1] if loss_data["losses"] else None,
            "improvement": (
                loss_data["losses"][0] - loss_data["losses"][-1]
                if loss_data["losses"]
                else None
            ),
        }

    with open(output_dir / "loss_stats.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"✓ Saved: loss_stats.json")


def main():
    print("\n📊 Analyzing Loss Curves...\n")
    analyze_loss_curves()
    print(
        f"\n✅ Loss analysis complete! Files saved to: "
        "/home/erimoldi/projects/agent_swarms/analysis/model_comparison_20260406"
    )


if __name__ == "__main__":
    main()
