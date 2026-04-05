"""Read per-agent output files after all agents finish."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from agent_swarms.outputs.schema import AgentResult, ExperimentSummary, TrajectoryEntry
from agent_swarms.snapshotting import SnapshotManager
from agent_swarms.reasoning_trace import ReasoningTracer


def collect_agent_result(
    agent_dir: Path,
    agent_id: str,
    experiment_id: str,
    mode: str,
) -> AgentResult:
    """Read all output files for one agent and build an AgentResult.

    Does not crash if agent produced no output (failed agents → failed=True).
    """
    result = AgentResult(
        agent_id=agent_id,
        experiment_id=experiment_id,
        mode=mode,
        workspace_path=str(agent_dir / "workspace"),
        results_path=str(agent_dir / "results"),
    )

    # Read metadata.json if present
    metadata_path = agent_dir / "results" / "metadata.json"
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            result.start_time = meta.get("start_time")
            result.end_time = meta.get("end_time")
            result.budget_seconds = meta.get("budget_seconds", 0)
            result.total_turns = meta.get("total_turns", 0)
        except Exception:
            pass

    # Read trajectory.jsonl (authoritative)
    traj_path = agent_dir / "results" / "trajectory.jsonl"
    if traj_path.exists():
        try:
            entries = []
            for line in traj_path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        entries.append(TrajectoryEntry.from_dict(json.loads(line)))
                    except Exception:
                        pass
            result.trajectory = entries
        except Exception:
            pass

    result.compute_derived()

    if not result.trajectory:
        result.failed = True
        result.failure_reason = "no trajectory entries found"

    return result


def collect_agent_snapshots(agent_dir: Path) -> list[dict]:
    """Return all snapshot metadata dicts for one agent."""
    snap_mgr = SnapshotManager(agent_dir / "snapshots")
    return [s.to_dict() for s in snap_mgr.list_snapshots()]


def collect_agent_reasoning(agent_dir: Path, agent_id: str) -> list[dict]:
    """Return all reasoning trace entries for one agent."""
    tracer = ReasoningTracer(agent_dir / "reasoning", agent_id)
    return [e.to_dict() for e in tracer.read_all()]


def collect_experiment(
    experiment_dir: Path,
    experiment_id: str,
    mode: str,
    agent_ids: list[str],
) -> ExperimentSummary:
    """Collect results for all agents in a mode directory.

    Never crashes — missing agents show up as failed rows.
    """
    summary = ExperimentSummary(experiment_id=experiment_id, mode=mode)

    mode_dir = experiment_dir / f"mode_{mode}"

    for agent_id in agent_ids:
        agent_dir = mode_dir / agent_id
        result = collect_agent_result(agent_dir, agent_id, experiment_id, mode)
        summary.agent_results.append(result)

    # Write aggregate outputs
    agg_dir = mode_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    combined_path = agg_dir / "combined_summary.json"
    combined_path.write_text(json.dumps(summary.to_dict(), indent=2))

    comparison_path = agg_dir / "comparison_table.csv"
    _write_comparison_csv(comparison_path, summary)

    # Collect snapshots and reasoning traces for all agents
    collected_candidates: dict = {"agents": {}}
    for agent_id in agent_ids:
        agent_dir = mode_dir / agent_id
        snapshots = collect_agent_snapshots(agent_dir)
        reasoning = collect_agent_reasoning(agent_dir, agent_id)
        collected_candidates["agents"][agent_id] = {
            "snapshots": snapshots,
            "reasoning_trace": reasoning,
            "snapshot_count": len(snapshots),
            "reasoning_steps": len(reasoning),
        }
    (agg_dir / "collected_candidates.json").write_text(
        json.dumps(collected_candidates, indent=2)
    )

    return summary


def collect_results(mode_dir: str | Path) -> dict:
    """Collect all agent results from a mode directory.

    Scans <mode_dir>/agent_*/results/ for trajectory.jsonl and metadata.json.
    Returns a plain dict suitable for JSON serialisation.

    Example:
        collect_results('runs/smoke_test/mode_parallel')
    """
    mode_path = Path(mode_dir).expanduser().resolve()
    agents = {}
    for agent_dir in sorted(mode_path.glob("agent_*")):
        if not agent_dir.is_dir():
            continue
        agent_id = agent_dir.name
        results_dir = agent_dir / "results"

        entry: dict = {
            "agent_id": agent_id,
            "results_path": str(results_dir),
            "trajectory": [],
            "metadata": {},
            "best_val_bpb": None,
            "runs_completed": 0,
        }

        # Read trajectory.jsonl
        traj_path = results_dir / "trajectory.jsonl"
        if traj_path.exists():
            for line in traj_path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        entry["trajectory"].append(json.loads(line))
                    except Exception:
                        pass

        if entry["trajectory"]:
            bpbs = [r.get("val_bpb") for r in entry["trajectory"] if r.get("val_bpb") is not None]
            if bpbs:
                entry["best_val_bpb"] = min(bpbs)
            entry["runs_completed"] = len(entry["trajectory"])

        # Read metadata.json
        meta_path = results_dir / "metadata.json"
        if meta_path.exists():
            try:
                entry["metadata"] = json.loads(meta_path.read_text())
            except Exception:
                pass

        agents[agent_id] = entry

    return {
        "mode_dir": str(mode_path),
        "agents": agents,
        "total_agents": len(agents),
        "best_val_bpb": min(
            (v["best_val_bpb"] for v in agents.values() if v["best_val_bpb"] is not None),
            default=None,
        ),
    }


def _write_comparison_csv(path: Path, summary: ExperimentSummary) -> None:
    rows = []
    for r in summary.agent_results:
        rows.append({
            "agent_id": r.agent_id,
            "best_val_bpb": r.best_val_bpb if r.best_val_bpb is not None else "",
            "first_val_bpb": r.first_val_bpb if r.first_val_bpb is not None else "",
            "improvement": r.improvement() if r.improvement() is not None else "",
            "total_runs": r.total_training_runs,
            "successful_runs": r.successful_training_runs,
            "failed": r.failed,
            "failure_reason": r.failure_reason,
            "total_turns": r.total_turns,
        })

    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
