"""Map original repo contracts to the new system.

The original repo writes results to:
  results/trajectories/<RUN_ID>/<agent_id>.jsonl
  results/snapshots/<RUN_ID>/<agent_id>/iter*.py
  results/diffs/<RUN_ID>/agent_N_changes.diff
  results/weights/<RUN_ID>/<agent_id>/model.pt
  results.tsv  (tab-separated: commit, val_bpb, memory_gb, status, description)

This adapter reads those formats and maps them to new-system structures.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from agent_swarms.outputs.schema import AgentResult, TrajectoryEntry


RESULTS_TSV_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"

# Best known result from original system
ORIGINAL_BEST = {
    "val_bpb": 1.1020746984708296,
    "run_id": "exp_smart_20260330_063836",
    "agent_id": "agent_0",
    "snapshot": "iter0003_s350_bpb1.1021.py",
    "hyperparameters": {
        "EMBEDDING_LR": 0.8,
        "UNEMBEDDING_LR": 0.005,
        "MATRIX_LR": 0.06,
        "WEIGHT_DECAY": 0.1,
        "WARMDOWN_RATIO": 0.4,
    },
}


def read_original_trajectory(
    original_results_root: Path,
    run_id: str,
    agent_id: str,
) -> list[TrajectoryEntry]:
    """Read trajectory from original repo format: results/trajectories/<run_id>/<agent_id>.jsonl"""
    traj_path = original_results_root / "trajectories" / run_id / f"{agent_id}.jsonl"
    if not traj_path.exists():
        return []
    entries = []
    for line in traj_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                d = json.loads(line)
                entries.append(TrajectoryEntry.from_dict(d))
            except Exception:
                pass
    return entries


def read_all_original_trajectories(
    original_results_root: Path,
) -> dict[tuple[str, str], list[TrajectoryEntry]]:
    """Read all trajectory files from original repo. Returns {(run_id, agent_id): entries}."""
    traj_root = original_results_root / "trajectories"
    if not traj_root.exists():
        return {}
    results = {}
    for run_dir in sorted(traj_root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        for jsonl_file in sorted(run_dir.glob("*.jsonl")):
            agent_id = jsonl_file.stem
            entries = []
            for line in jsonl_file.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        entries.append(TrajectoryEntry.from_dict(d))
                    except Exception:
                        pass
            results[(run_id, agent_id)] = entries
    return results


def find_best_original_result(
    original_results_root: Path,
) -> Optional[tuple[str, str, float]]:
    """Scan all trajectory files and return (run_id, agent_id, best_val_bpb)."""
    all_traj = read_all_original_trajectories(original_results_root)
    best = None
    best_key = None
    for key, entries in all_traj.items():
        for e in entries:
            if best is None or e.val_bpb < best:
                best = e.val_bpb
                best_key = key
    if best_key is None:
        return None
    return (best_key[0], best_key[1], best)


def generate_diff(
    workspace: Path,
    run_id: str,
    agent_id: str,
    diffs_dir: Path,
) -> Optional[Path]:
    """Generate diff between train.py.baseline and current train.py."""
    baseline = workspace / "train.py.baseline"
    current = workspace / "train.py"
    if not baseline.exists() or not current.exists():
        return None

    diffs_dir.mkdir(parents=True, exist_ok=True)
    diff_path = diffs_dir / run_id / f"{agent_id}_changes.diff"
    diff_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["diff", "-u", str(baseline), str(current)],
        capture_output=True,
        text=True,
    )
    diff_path.write_text(result.stdout)
    return diff_path


def write_results_tsv_row(
    tsv_path: Path,
    commit: str,
    val_bpb: float,
    memory_gb: float,
    status: str,
    description: str,
) -> None:
    """Append one row to results.tsv, preserving original schema."""
    if not tsv_path.exists():
        tsv_path.write_text(RESULTS_TSV_HEADER + "\n")
    row = f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"
    with open(tsv_path, "a") as f:
        f.write(row)
