"""Write final_report.md and comparison tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from agent_swarms.outputs.evaluator import (
    compare_parallel_vs_single,
    evaluate_experiment,
    KNOWN_BEST_VAL_BPB,
)
from agent_swarms.outputs.schema import ExperimentSummary


def write_experiment_report(
    summary: ExperimentSummary,
    output_dir: Path,
) -> Path:
    """Write experiment_report.txt to output_dir/aggregate/."""
    eval_data = evaluate_experiment(summary)
    lines = []
    lines.append(f"Experiment: {summary.experiment_id}")
    lines.append(f"Mode:       {summary.mode}")
    lines.append(f"Best val_bpb: {eval_data['best_val_bpb']}")
    lines.append(
        f"Beats known best ({KNOWN_BEST_VAL_BPB}): {eval_data['beats_known_best']}"
    )
    lines.append(f"Best agent: {eval_data['best_agent_id']}")
    lines.append(f"Total successful runs: {eval_data['total_successful_runs']}")
    lines.append("")
    lines.append("Per-agent summary:")
    for a in eval_data["agents"]:
        lines.append(
            f"  {a['agent_id']}: best={a['best_val_bpb']}  first={a['first_val_bpb']}"
            f"  improvement={a['improvement']}  runs={a['total_training_runs']}"
            f"  failed={a['failed']}"
        )

    report = "\n".join(lines) + "\n"
    out = output_dir / "aggregate" / "experiment_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return out


def write_final_comparison(
    parallel_summary: ExperimentSummary,
    single_summary: ExperimentSummary,
    output_dir: Path,
) -> Path:
    """Write parallel_vs_single.csv and final_report.md to output_dir/final_comparison/."""
    final_dir = output_dir / "final_comparison"
    final_dir.mkdir(parents=True, exist_ok=True)

    comparison = compare_parallel_vs_single(parallel_summary, single_summary)

    # CSV
    csv_path = final_dir / "parallel_vs_single.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison.keys()))
        writer.writeheader()
        writer.writerow(comparison)

    # Markdown report
    md_lines = [
        "# Final Comparison: Parallel vs Single-Agent-Longer",
        "",
        f"| Metric | Parallel | Single Long |",
        f"|--------|----------|-------------|",
        f"| Best val_bpb | {comparison['parallel_best_val_bpb']} | {comparison['single_best_val_bpb']} |",
        f"| Total successful runs | {comparison['parallel_total_runs']} | {comparison['single_total_runs']} |",
        f"| Parallel wins | {comparison['parallel_wins']} | — |",
        f"| Delta val_bpb | {comparison['delta_val_bpb']} | — |",
        "",
        f"Known best baseline: {KNOWN_BEST_VAL_BPB}",
        "",
        "## Parallel agent details",
    ]
    for r in parallel_summary.agent_results:
        md_lines.append(
            f"- {r.agent_id}: best={r.best_val_bpb}, runs={r.total_training_runs}, failed={r.failed}"
        )
    md_lines.append("")
    md_lines.append("## Single-agent-longer details")
    for r in single_summary.agent_results:
        md_lines.append(
            f"- {r.agent_id}: best={r.best_val_bpb}, runs={r.total_training_runs}, failed={r.failed}"
        )

    md_path = final_dir / "final_report.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    return md_path
