#!/usr/bin/env python3
"""
Model Performance Comparison Analysis
Comparing Claude models (Haiku, Sonnet, Opus) in 2-agent swarm setting
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Experiment data
EXPERIMENTS = {
    "haiku_run1": {
        "id": "exp_20260405_022850",
        "model": "Haiku 4.5",
        "model_key": "haiku",
        "start": datetime(2026, 4, 5, 2, 28, 50),
        "end": datetime(2026, 4, 5, 4, 28, 0),
        "best_bpb": 1.041477,
        "total_runs": 27,
        "agent0_best": 1.047929,
        "agent0_runs": 14,
        "agent1_best": 1.041477,
        "agent1_runs": 13,
        "improvement": -0.07552,
    },
    "haiku_run2": {
        "id": "exp_20260405_124604",
        "model": "Haiku 4.5",
        "model_key": "haiku",
        "start": datetime(2026, 4, 5, 12, 46, 3),
        "end": datetime(2026, 4, 5, 14, 46, 0),
        "best_bpb": 1.044341,
        "total_runs": 28,
        "agent0_best": 1.044341,
        "agent0_runs": 14,
        "agent1_best": 1.050358,
        "agent1_runs": 14,
        "improvement": -0.059369,
    },
    "sonnet": {
        "id": "exp_20260406_024115",
        "model": "Sonnet 4.6",
        "model_key": "sonnet",
        "start": datetime(2026, 4, 6, 2, 41, 15),
        "end": datetime(2026, 4, 6, 4, 46, 0),
        "best_bpb": 1.044216,
        "total_runs": 29,
        "agent0_best": 1.044662,
        "agent0_runs": 13,
        "agent1_best": 1.044216,
        "agent1_runs": 16,
        "improvement": -0.077424,
    },
    "opus": {
        "id": "exp_20260406_044120",
        "model": "Opus 4.6",
        "model_key": "opus",
        "start": datetime(2026, 4, 6, 4, 41, 20),
        "end": datetime(2026, 4, 6, 6, 41, 0),
        "best_bpb": 1.044304,
        "total_runs": 22,
        "agent0_best": 1.044304,
        "agent0_runs": 11,
        "agent1_best": 1.047083,
        "agent1_runs": 11,
        "improvement": -0.074044,
    },
}

BASELINE = 1.1020746984708296


def calculate_metrics() -> Dict:
    """Calculate comprehensive metrics for each experiment."""
    results = {}

    for exp_key, exp_data in EXPERIMENTS.items():
        duration_minutes = (exp_data["end"] - exp_data["start"]).total_seconds() / 60

        results[exp_key] = {
            **exp_data,
            "duration_minutes": duration_minutes,
            "avg_time_per_run_seconds": (duration_minutes * 60) / exp_data["total_runs"],
            "improvement_pct": abs(exp_data["improvement"]) / 1.128 * 100,  # ~1.128 is typical initial BPB
            "beats_baseline_pct": (BASELINE - exp_data["best_bpb"]) / BASELINE * 100,
            "efficiency_score": exp_data["total_runs"] / duration_minutes,  # runs per minute
        }

    return results


def create_summary_table(results: Dict) -> pd.DataFrame:
    """Create summary comparison table."""
    summary_data = []

    for exp_key, metrics in results.items():
        summary_data.append({
            "Experiment": exp_key,
            "Model": metrics["model"],
            "Best BPB": f"{metrics['best_bpb']:.6f}",
            "Beats Baseline": f"{metrics['beats_baseline_pct']:.2f}%",
            "Total Runs": metrics["total_runs"],
            "Duration (min)": f"{metrics['duration_minutes']:.1f}",
            "Avg Run Time (s)": f"{metrics['avg_time_per_run_seconds']:.1f}",
            "Runs/Min": f"{metrics['efficiency_score']:.2f}",
        })

    df = pd.DataFrame(summary_data)
    return df


def create_detailed_analysis(results: Dict) -> str:
    """Create detailed text analysis."""
    analysis = []

    analysis.append("=" * 80)
    analysis.append("CLAUDE MODEL PERFORMANCE COMPARISON ANALYSIS")
    analysis.append("2-Agent Swarm Learning Experiment")
    analysis.append("=" * 80)
    analysis.append("")

    # Aggregate by model
    by_model = {}
    for exp_key, metrics in results.items():
        model_key = metrics["model_key"]
        if model_key not in by_model:
            by_model[model_key] = []
        by_model[model_key].append(metrics)

    # Overall rankings
    analysis.append("OVERALL RANKINGS (by best BPB):")
    analysis.append("-" * 80)
    sorted_runs = sorted(results.items(), key=lambda x: x[1]["best_bpb"])
    for rank, (exp_key, metrics) in enumerate(sorted_runs, 1):
        analysis.append(
            f"{rank}. {metrics['model']:12s} - Best BPB: {metrics['best_bpb']:.6f} "
            f"({metrics['total_runs']} runs in {metrics['duration_minutes']:.0f} min)"
        )
    analysis.append("")

    # By Model Comparison
    analysis.append("COMPARISON BY MODEL:")
    analysis.append("-" * 80)
    for model_key in ["haiku", "sonnet", "opus"]:
        if model_key not in by_model:
            continue

        runs = by_model[model_key]
        model_name = runs[0]["model"]

        if len(runs) == 1:
            run = runs[0]
            analysis.append(f"\n{model_name}:")
            analysis.append(f"  Experiments: 1")
            analysis.append(f"  Best BPB: {run['best_bpb']:.6f}")
            analysis.append(f"  Beats Baseline: {run['beats_baseline_pct']:.2f}%")
            analysis.append(f"  Total Runs: {run['total_runs']}")
            analysis.append(f"  Duration: {run['duration_minutes']:.0f} minutes")
            analysis.append(f"  Avg Run Time: {run['avg_time_per_run_seconds']:.1f}s")
            analysis.append(f"  Efficiency: {run['efficiency_score']:.2f} runs/min")
        else:
            best_bpb_values = [r["best_bpb"] for r in runs]
            avg_best_bpb = np.mean(best_bpb_values)
            std_best_bpb = np.std(best_bpb_values)

            analysis.append(f"\n{model_name}: (2 runs)")
            analysis.append(f"  Best BPB (run 1): {runs[0]['best_bpb']:.6f}")
            analysis.append(f"  Best BPB (run 2): {runs[1]['best_bpb']:.6f}")
            analysis.append(f"  Average Best BPB: {avg_best_bpb:.6f} ± {std_best_bpb:.6f}")
            analysis.append(f"  Avg Beats Baseline: {np.mean([r['beats_baseline_pct'] for r in runs]):.2f}%")
            analysis.append(f"  Total Runs (both): {sum(r['total_runs'] for r in runs)}")
            analysis.append(f"  Avg Duration: {np.mean([r['duration_minutes'] for r in runs]):.0f} min")
            analysis.append(f"  Avg Run Time: {np.mean([r['avg_time_per_run_seconds'] for r in runs]):.1f}s")

    analysis.append("\n" + "=" * 80)
    analysis.append("KEY FINDINGS:")
    analysis.append("-" * 80)

    # Best overall
    best_run = sorted_runs[0]
    worst_run = sorted_runs[-1]
    analysis.append(f"\n✓ BEST OVERALL: {best_run[1]['model']} ({best_run[1]['id']})")
    analysis.append(f"  BPB: {best_run[1]['best_bpb']:.6f}")
    analysis.append(f"  Outperforms baseline by {best_run[1]['beats_baseline_pct']:.2f}%")

    analysis.append(f"\n✗ WORST OVERALL: {worst_run[1]['model']} ({worst_run[1]['id']})")
    analysis.append(f"  BPB: {worst_run[1]['best_bpb']:.6f}")
    analysis.append(f"  Difference from best: {worst_run[1]['best_bpb'] - best_run[1]['best_bpb']:.6f}")

    # Efficiency
    most_efficient = max(results.items(), key=lambda x: x[1]["efficiency_score"])
    analysis.append(f"\n⚡ MOST EFFICIENT: {most_efficient[1]['model']}")
    analysis.append(f"  {most_efficient[1]['efficiency_score']:.2f} runs/minute")
    analysis.append(f"  Completed {most_efficient[1]['total_runs']} runs in {most_efficient[1]['duration_minutes']:.0f} min")

    # Quality vs Efficiency trade-off
    analysis.append(f"\n📊 QUALITY vs EFFICIENCY TRADE-OFF:")
    analysis.append("")
    for exp_key in ["haiku_run1", "sonnet", "opus"]:
        if exp_key in results:
            m = results[exp_key]
            analysis.append(
                f"  {m['model']:12s}: BPB={m['best_bpb']:.6f}, "
                f"Efficiency={m['efficiency_score']:.2f} runs/min, "
                f"Quality/speed={m['best_bpb']/m['efficiency_score']:.4f}"
            )

    analysis.append("")
    analysis.append("=" * 80)

    return "\n".join(analysis)


def create_visualizations(results: Dict, output_dir: Path):
    """Create comparison visualizations."""

    # Prepare data
    models = []
    bpbs = []
    runs = []
    durations = []
    efficiency = []
    colors = []
    color_map = {"haiku": "#FF6B6B", "sonnet": "#4ECDC4", "opus": "#45B7D1"}

    for exp_key, metrics in results.items():
        models.append(f"{metrics['model']}\n({metrics['id'][-8:]})")
        bpbs.append(metrics["best_bpb"])
        runs.append(metrics["total_runs"])
        durations.append(metrics["duration_minutes"])
        efficiency.append(metrics["efficiency_score"])
        colors.append(color_map[metrics["model_key"]])

    # Figure 1: Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Claude Model Performance Comparison\n2-Agent Swarm Learning", fontsize=16, fontweight="bold")

    # BPB comparison
    ax = axes[0, 0]
    bars = ax.bar(models, bpbs, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.axhline(BASELINE, color="red", linestyle="--", linewidth=2, label=f"Baseline ({BASELINE:.4f})")
    ax.set_ylabel("Validation BPB (Lower is Better)", fontweight="bold")
    ax.set_title("Best Validation BPB")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, bpb in zip(bars, bpbs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{bpb:.4f}', ha='center', va='bottom', fontweight="bold")

    # Number of runs
    ax = axes[0, 1]
    bars = ax.bar(models, runs, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Total Successful Runs", fontweight="bold")
    ax.set_title("Total Runs Completed")
    ax.grid(axis="y", alpha=0.3)
    for bar, run_count in zip(bars, runs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(run_count)}', ha='center', va='bottom', fontweight="bold")

    # Duration
    ax = axes[1, 0]
    bars = ax.bar(models, durations, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Duration (minutes)", fontweight="bold")
    ax.set_title("Experiment Duration")
    ax.grid(axis="y", alpha=0.3)
    for bar, dur in zip(bars, durations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{dur:.0f}', ha='center', va='bottom', fontweight="bold")

    # Efficiency (runs per minute)
    ax = axes[1, 1]
    bars = ax.bar(models, efficiency, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Runs per Minute", fontweight="bold")
    ax.set_title("Efficiency (Runs/Minute)")
    ax.grid(axis="y", alpha=0.3)
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.2f}', ha='center', va='bottom', fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_metrics.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: comparison_metrics.png")

    # Figure 2: BPB improvement from baseline
    fig, ax = plt.subplots(figsize=(10, 6))
    improvements = [(BASELINE - bpb) / BASELINE * 100 for bpb in bpbs]
    bars = ax.barh(models, improvements, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_xlabel("Improvement from Baseline (%)", fontweight="bold")
    ax.set_title("Performance Improvement Over Baseline", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, imp in zip(bars, improvements):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {imp:.2f}%', ha='left', va='center', fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_improvement.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: baseline_improvement.png")

    # Figure 3: Quality vs Efficiency scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model, bpb, eff, color) in enumerate(zip(models, bpbs, efficiency, colors)):
        ax.scatter(eff, bpb, s=300, alpha=0.7, color=color, edgecolors="black", linewidth=2)
        ax.annotate(model, (eff, bpb), xytext=(10, 5), textcoords="offset points", fontweight="bold")

    ax.set_xlabel("Efficiency (Runs/Minute) →", fontweight="bold")
    ax.set_ylabel("← Best BPB (Lower is Better)", fontweight="bold")
    ax.set_title("Quality vs Efficiency Trade-off", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add quadrant lines
    ax.axhline(np.mean(bpbs), color="gray", linestyle=":", alpha=0.5)
    ax.axvline(np.mean(efficiency), color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "quality_vs_efficiency.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: quality_vs_efficiency.png")


def main():
    output_dir = Path("/home/erimoldi/projects/agent_swarms/analysis/model_comparison_20260406")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 Analyzing Model Performance Comparison...\n")

    # Calculate metrics
    results = calculate_metrics()

    # Summary table
    summary_df = create_summary_table(results)
    summary_csv = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved: summary_table.csv")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

    # Detailed analysis
    analysis_text = create_detailed_analysis(results)
    analysis_file = output_dir / "ANALYSIS.txt"
    with open(analysis_file, "w") as f:
        f.write(analysis_text)
    print(f"\n✓ Saved: ANALYSIS.txt")
    print("\n" + analysis_text)

    # Visualizations
    print("\n📈 Creating Visualizations...\n")
    create_visualizations(results, output_dir)

    # Save raw data as JSON
    json_file = output_dir / "raw_data.json"
    with open(json_file, "w") as f:
        # Convert datetime objects for JSON serialization
        json_results = {}
        for k, v in results.items():
            v_copy = v.copy()
            v_copy["start"] = v_copy["start"].isoformat()
            v_copy["end"] = v_copy["end"].isoformat()
            json_results[k] = v_copy
        json.dump(json_results, f, indent=2)
    print(f"✓ Saved: raw_data.json")

    print(f"\n✅ Analysis complete! All files saved to: {output_dir}")


if __name__ == "__main__":
    main()
