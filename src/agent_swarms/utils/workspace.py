"""Git worktree creation and teardown for isolated agent workspaces."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from agent_swarms.compatibility.training_harness import (
    generate_check_training_sh,
    generate_run_on_worker_sh,
    generate_run_training_sh,
    generate_slurm_check_training_sh,
    generate_snapshot_helpers,
    generate_start_gpu_worker_sh,
    generate_stop_gpu_worker_sh,
    generate_submit_training_sh,
)


class WorkspaceError(Exception):
    pass


def create_workspace(
    autoresearch_dir: Path,
    workspace_path: Path,
    branch_name: str,
    train_budget_seconds: int,
    run_id: str,
    agent_id: str,
    results_root: Path,
    slurm_partition: str = "pi_tpoggio",
    slurm_gres: str = "gpu:1",
    slurm_time: str = "00:08:00",
    use_slurm: bool = True,
    persistent_worker: bool = True,
    agent_time_budget_minutes: int = 60,
    swarm_memory_path: Optional[Path] = None,
) -> Path:
    """Create an isolated git worktree for one agent.

    Steps:
    1. Create branch in autoresearch if not exists
    2. Create git worktree at workspace_path
    3. Copy train.py.baseline
    4. Symlink .venv and data/
    5. Generate training scripts (SLURM: submit_training.sh + check_training.sh;
       local: run_training.sh + check_training.sh)
    6. Create results directory

    Returns workspace_path.
    """
    autoresearch_dir = autoresearch_dir.resolve()
    workspace_path = workspace_path.resolve()
    results_root = results_root.resolve()

    _ensure_branch(autoresearch_dir, branch_name)
    _create_worktree(autoresearch_dir, workspace_path, branch_name)
    _save_baseline(workspace_path)
    _symlink_shared(autoresearch_dir, workspace_path)
    _install_swarm_tools(workspace_path, swarm_memory_path)

    if use_slurm:
        if persistent_worker:
            # Convert agent budget to HH:MM:SS, adding 10-minute safety margin
            total_minutes = agent_time_budget_minutes + 10
            worker_time = f"{total_minutes // 60:02d}:{total_minutes % 60:02d}:00"
            generate_start_gpu_worker_sh(
                workspace_path,
                agent_id=agent_id,
                results_root=results_root,
                slurm_partition=slurm_partition,
                slurm_gres=slurm_gres,
                worker_time=worker_time,
            )
            generate_run_on_worker_sh(workspace_path, train_budget_seconds)
            generate_stop_gpu_worker_sh(workspace_path)
        else:
            # Legacy: one sbatch per train.py run
            generate_submit_training_sh(
                workspace_path,
                agent_id=agent_id,
                results_root=results_root,
                slurm_partition=slurm_partition,
                slurm_gres=slurm_gres,
                slurm_time=slurm_time,
            )
            generate_slurm_check_training_sh(workspace_path)
    else:
        generate_run_training_sh(workspace_path, train_budget_seconds)
        generate_check_training_sh(workspace_path)

    # Generate snapshot helper scripts (save_snapshot.py, update_snapshot.py)
    generate_snapshot_helpers(
        workspace=workspace_path,
        agent_id=agent_id,
        results_root=results_root,
    )

    # Set up per-agent output directories
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root.parent / "snapshots").mkdir(parents=True, exist_ok=True)
    (results_root.parent / "reasoning").mkdir(parents=True, exist_ok=True)

    return workspace_path


def destroy_workspace(autoresearch_dir: Path, workspace_path: Path) -> None:
    """Remove a git worktree and its directory."""
    autoresearch_dir = autoresearch_dir.resolve()
    workspace_path = workspace_path.resolve()
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(workspace_path)],
            cwd=autoresearch_dir,
            check=False,
            capture_output=True,
        )
    except Exception:
        pass
    if workspace_path.exists():
        shutil.rmtree(workspace_path, ignore_errors=True)


def _ensure_branch(autoresearch_dir: Path, branch_name: str) -> None:
    result = subprocess.run(
        ["git", "show-ref", "--quiet", f"refs/heads/{branch_name}"],
        cwd=autoresearch_dir,
        capture_output=True,
    )
    if result.returncode != 0:
        subprocess.run(
            ["git", "branch", branch_name, "HEAD"],
            cwd=autoresearch_dir,
            check=True,
            capture_output=True,
        )


def _create_worktree(
    autoresearch_dir: Path, workspace_path: Path, branch_name: str
) -> None:
    if workspace_path.exists():
        return
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "worktree", "add", str(workspace_path), branch_name],
        cwd=autoresearch_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise WorkspaceError(
            f"Failed to create worktree at {workspace_path}: {result.stderr}"
        )


def _install_swarm_tools(
    workspace_path: Path,
    swarm_memory_path: Optional[Path] = None,
) -> None:
    """Copy coordinator.py, collab.md, and program.md into the workspace.

    coordinator.py is the local shim implementing the autoresearch-at-home
    Coordinator Python class API backed by our JSONL blackboard.

    collab.md and program.md are taken verbatim from the cloned
    autoresearch-at-home repo so agents follow the upstream protocol exactly.
    """
    pkg_dir = Path(__file__).parent.parent
    project_root = pkg_dir.parent.parent  # agent_swarms/
    autoresearch_dir = project_root / "autoresearch"

    # coordinator.py — local shim (implements Ensue-compatible Python API)
    coordinator_src = pkg_dir / "coordinator_local.py"
    coordinator_dst = workspace_path / "coordinator.py"
    if coordinator_src.exists():
        shutil.copy2(coordinator_src, coordinator_dst)

    # collab.md — verbatim from autoresearch-at-home clone
    collab_src = autoresearch_dir / "collab.md"
    collab_dst = workspace_path / "collab.md"
    if collab_src.exists():
        shutil.copy2(collab_src, collab_dst)

    # program.md — verbatim from autoresearch-at-home clone
    # (overrides the one from the git worktree, which is for solo mode)
    program_src = autoresearch_dir / "program.md"
    program_dst = workspace_path / "program.md"
    if program_src.exists():
        shutil.copy2(program_src, program_dst)

    # Write a .swarm_env file so coordinator.py can find the blackboard
    # even without the orchestrator-injected environment variable.
    if swarm_memory_path is not None:
        env_file = workspace_path / ".swarm_env"
        env_file.write_text(f"SWARM_MEMORY_PATH={swarm_memory_path}\n")


def _save_baseline(workspace_path: Path) -> None:
    train_py = workspace_path / "train.py"
    baseline = workspace_path / "train.py.baseline"
    if train_py.exists() and not baseline.exists():
        shutil.copy2(train_py, baseline)


def _symlink_shared(autoresearch_dir: Path, workspace_path: Path) -> None:
    venv_src = autoresearch_dir / ".venv"
    venv_dst = workspace_path / ".venv"
    if venv_src.exists() and not venv_dst.exists():
        venv_dst.symlink_to(venv_src)

    data_src = autoresearch_dir / "data"
    data_dst = workspace_path / "data"
    if data_src.exists() and not data_dst.exists():
        data_dst.symlink_to(data_src)
