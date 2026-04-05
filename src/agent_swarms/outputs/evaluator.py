"""Compute evaluation metrics across agent results."""

from __future__ import annotations

from typing import Optional

from agent_swarms.outputs.schema import AgentResult, ExperimentSummary

KNOWN_BEST_VAL_BPB = 1.1020746984708296  # best result from original system


def evaluate_agent(result: AgentResult) -> dict:
    """Per-agent evaluation metrics."""
    return {
        "agent_id": result.agent_id,
        "failed": result.failed,
        "best_val_bpb": result.best_val_bpb,
        "first_val_bpb": result.first_val_bpb,
        "improvement": result.improvement(),
        "beats_known_best": (
            result.best_val_bpb < KNOWN_BEST_VAL_BPB
            if result.best_val_bpb is not None
            else False
        ),
        "total_training_runs": result.total_training_runs,
        "successful_training_runs": result.successful_training_runs,
    }


def evaluate_experiment(summary: ExperimentSummary) -> dict:
    """Experiment-level evaluation."""
    agent_evals = [evaluate_agent(r) for r in summary.agent_results]
    best_bpb = summary.best_val_bpb()

    return {
        "experiment_id": summary.experiment_id,
        "mode": summary.mode,
        "best_val_bpb": best_bpb,
        "beats_known_best": (
            best_bpb < KNOWN_BEST_VAL_BPB if best_bpb is not None else False
        ),
        "best_agent_id": (
            summary.best_agent().agent_id if summary.best_agent() else None
        ),
        "total_successful_runs": sum(
            r.successful_training_runs for r in summary.agent_results
        ),
        "agents": agent_evals,
    }


def compare_parallel_vs_single(
    parallel_summary: ExperimentSummary,
    single_summary: ExperimentSummary,
) -> dict:
    """Compare parallel-agent mode vs single-agent-longer mode."""
    parallel_best = parallel_summary.best_val_bpb()
    single_best = single_summary.best_val_bpb()

    if parallel_best is not None and single_best is not None:
        parallel_wins = parallel_best < single_best
        delta = single_best - parallel_best
    else:
        parallel_wins = None
        delta = None

    return {
        "parallel_best_val_bpb": parallel_best,
        "single_best_val_bpb": single_best,
        "parallel_wins": parallel_wins,
        "delta_val_bpb": delta,
        "parallel_total_runs": sum(
            r.successful_training_runs for r in parallel_summary.agent_results
        ),
        "single_total_runs": sum(
            r.successful_training_runs for r in single_summary.agent_results
        ),
    }
