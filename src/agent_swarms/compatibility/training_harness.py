"""Generate training wrapper scripts for each agent workspace.

Provides two SLURM strategies:
  - Persistent worker (default): one long-lived GPU allocation per agent,
    trigger-file protocol for each train.py run.  No re-queuing overhead.
      start_gpu_worker.sh  — allocate GPU for full agent budget
      run_on_worker.sh     — drop trigger, block until result
      stop_gpu_worker.sh   — signal worker to exit, cancel job
  - One-shot (legacy): one sbatch per train.py run.
      submit_training.sh + check_training.sh

Local (non-SLURM) scripts remain available for development without a cluster:
  run_training.sh + check_training.sh

Scripts are called by the Claude Code sub-agent via its Bash tool.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


# ---------------------------------------------------------------------------
# SLURM-based scripts (production)
# ---------------------------------------------------------------------------

def generate_submit_training_sh(
    workspace: Path,
    agent_id: str,
    results_root: Path,
    slurm_partition: str = "pi_tpoggio",
    slurm_gres: str = "gpu:1",
    slurm_time: str = "00:08:00",
) -> Path:
    """Write submit_training.sh into workspace.

    The script:
    1. Cancels any previous SLURM job named train_<agent_id> owned by $USER
    2. Submits a new job with --parsable and prints only the job ID
    Returns path to the script.
    """
    path_additions = _path_additions()
    uv_bin = _find_bin("uv")

    script = f"""#!/bin/bash
# submit_training.sh — submit one SLURM training job and print its job ID
export PATH="{path_additions}:$PATH"
AGENT_ID="{agent_id}"
WORKSPACE_PATH="{workspace}"
RESULTS_ROOT="{results_root}"

# Cancel any previous job with the same name owned by this user
squeue -u "$USER" -n "train_${{AGENT_ID}}" -h -o "%i" 2>/dev/null | xargs -r scancel

mkdir -p "$WORKSPACE_PATH/logs"

JOB_ID=$(sbatch \\
  --parsable \\
  --job-name="train_${{AGENT_ID}}" \\
  --partition={slurm_partition} \\
  --gres={slurm_gres} \\
  --time={slurm_time} \\
  --nodes=1 \\
  --ntasks=1 \\
  --cpus-per-task=4 \\
  --mem=32G \\
  --output="$WORKSPACE_PATH/logs/train_%j.out" \\
  --error="$WORKSPACE_PATH/logs/train_%j.err" \\
  --export=ALL,RUN_ID="${{RUN_ID:-smoke_test}}",AGENT_ID="${{AGENT_ID}}",RESULTS_ROOT="${{RESULTS_ROOT}}" \\
  --wrap="export PATH=\\"{path_additions}:\\$PATH\\" && cd \\"$WORKSPACE_PATH\\" && {uv_bin} run train.py")

echo "$JOB_ID"
"""
    out = workspace / "submit_training.sh"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out


def generate_slurm_check_training_sh(workspace: Path) -> Path:
    """Write check_training.sh (SLURM-aware) into workspace.

    Usage: bash check_training.sh <JOB_ID>
    Prints one of:
      TRAINING RUNNING: <status>
      TRAINING FAILED: <reason>
      TRAINING DONE
    Returns path to the script.
    """
    script = f"""#!/bin/bash
# check_training.sh — poll a SLURM job for completion
WORKSPACE_PATH="{workspace}"
JOB_ID="${{1:?Usage: $0 <JOB_ID>}}"

STATUS=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null | tr -d '[:space:]')

if [ -z "$STATUS" ]; then
  # Job no longer in queue — check sacct for final state
  SACCT_STATE=$(sacct -j "$JOB_ID" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
  LOG="$WORKSPACE_PATH/logs/train_${{JOB_ID}}.out"
  if grep -q "val_bpb:" "$LOG" 2>/dev/null; then
    echo "TRAINING DONE"
    grep "val_bpb:" "$LOG"
    grep "peak_vram_mb:" "$LOG" 2>/dev/null || true
  else
    echo "TRAINING FAILED: no val_bpb in output (sacct state: ${{SACCT_STATE:-unknown}})"
    echo "--- last 20 lines of $LOG ---"
    tail -20 "$LOG" 2>/dev/null || echo "(log not found)"
  fi
elif [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "CANCELLED" ] || [ "$STATUS" = "TIMEOUT" ]; then
  echo "TRAINING FAILED: $STATUS"
else
  echo "TRAINING RUNNING: $STATUS"
fi
"""
    out = workspace / "check_training.sh"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out


# ---------------------------------------------------------------------------
# SLURM persistent-worker scripts (preferred — one GPU allocation per agent)
# ---------------------------------------------------------------------------

def generate_worker_loop_sh(
    workspace: Path,
    uv_bin: str,
    path_additions: str,
) -> Path:
    """Write worker_loop.sh — the actual SLURM job body.

    Keeping this in a separate file avoids all --wrap double-quote expansion
    issues: every $VAR and $(cmd) here is a plain shell variable, evaluated
    on the compute node where the file is executed, not on the login node.
    """
    script = f"""#!/bin/bash
# worker_loop.sh — persistent GPU worker job body.
# Submitted by start_gpu_worker.sh; runs entirely on the compute node.
set -uo pipefail
export PATH="{path_additions}:$PATH"
cd "{workspace}"
mkdir -p logs

# Detect CUDA stubs dir so Triton can compile its driver helper (libcuda.so).
for _cuda_stubs in /usr/local/cuda/lib64/stubs /usr/local/cuda-*/lib64/stubs /cm/shared/apps/cuda/current/lib64/stubs; do
  if [ -f "$_cuda_stubs/libcuda.so" ]; then
    export TRITON_LIBCUDA_PATH="$_cuda_stubs"
    break
  fi
done

RUN_COUNT=0
while [ ! -f stop_worker ]; do
  if [ -f run.trigger ]; then
    rm -f run.trigger run.result
    RUN_COUNT=$((RUN_COUNT + 1))
    RUN_LOG=$(printf 'logs/train_run_%03d.out' "$RUN_COUNT")
    {uv_bin} run train.py > "$RUN_LOG" 2>&1
    cp "$RUN_LOG" logs/train_current.out
    VAL=$(grep 'val_bpb:' logs/train_current.out 2>/dev/null | head -1 | awk '{{print $2}}') || VAL=""
    VRAM=$(grep 'peak_vram_mb:' logs/train_current.out 2>/dev/null | head -1 | awk '{{print $2}}') || VRAM=""
    if [ -n "$VAL" ]; then
      {{ echo TRAINING DONE; echo val_bpb: "$VAL"; [ -n "$VRAM" ] && echo peak_vram_mb: "$VRAM"; }} > run.result
    else
      {{ echo TRAINING FAILED; tail -30 logs/train_current.out; }} > run.result
    fi
  fi
  sleep 2
done
"""
    out = workspace / "worker_loop.sh"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out


def generate_start_gpu_worker_sh(
    workspace: Path,
    agent_id: str,
    results_root: Path,
    slurm_partition: str = "pi_tpoggio",
    slurm_gres: str = "gpu:1",
    worker_time: str = "01:00:00",
) -> Path:
    """Write start_gpu_worker.sh into workspace.

    Submits a single long-lived SLURM job (worker_loop.sh) that holds the GPU
    for the full agent budget.  The job runs a 2-second poll loop:
      - waits for run.trigger to appear (written by run_on_worker.sh)
      - runs train.py, writes val_bpb / error to run.result
      - removes trigger, waits for next iteration
      - exits cleanly when stop_worker sentinel appears

    Returns the script path.  Agent should call once at startup:
        WORKER_JOB=$(bash start_gpu_worker.sh)
    """
    path_additions = _path_additions()
    uv_bin = _find_bin("uv")

    # Write the worker loop as a separate file so all shell variables are
    # evaluated on the compute node, not expanded by the login shell.
    generate_worker_loop_sh(workspace, uv_bin, path_additions)

    script = f"""#!/bin/bash
# start_gpu_worker.sh — allocate a GPU for the full agent budget and start
# the persistent training worker loop.  Prints the SLURM job ID on stdout.
export PATH="{path_additions}:$PATH"
AGENT_ID="{agent_id}"
WORKSPACE_PATH="{workspace}"
RESULTS_ROOT="{results_root}"

# If a worker is already running for THIS workspace, reuse it.
EXISTING_JOB=$(squeue -u "$USER" -n "worker_${{AGENT_ID}}" -h -o "%i %T %Z" 2>/dev/null \
  | awk -v ws="$WORKSPACE_PATH" '$2=="RUNNING" && $3==ws {{print $1}}' | head -1)
if [ -n "$EXISTING_JOB" ]; then
  echo "Worker already running (job $EXISTING_JOB). Reusing." >&2
  [ -f "$WORKSPACE_PATH/gpu_allocated_at" ] || date -Iseconds > "$WORKSPACE_PATH/gpu_allocated_at"
  echo "$EXISTING_JOB"
  exit 0
fi

# Cancel any previous pending job with the same name (prevents deadlock on re-submission)
squeue -u "$USER" -n "worker_${{AGENT_ID}}" -h -o "%i" 2>/dev/null | xargs -r scancel
sleep 2

# Clean up any stale sentinel / trigger files from a previous run
rm -f "$WORKSPACE_PATH/stop_worker" "$WORKSPACE_PATH/run.trigger" "$WORKSPACE_PATH/run.result"
mkdir -p "$WORKSPACE_PATH/logs"

JOB_ID=$(sbatch \\
  --parsable \\
  --job-name="worker_${{AGENT_ID}}" \\
  --partition={slurm_partition} \\
  --gres={slurm_gres} \\
  --time={worker_time} \\
  --nodes=1 \\
  --ntasks=1 \\
  --cpus-per-task=4 \\
  --mem=32G \\
  --output="$WORKSPACE_PATH/logs/worker_%j.out" \\
  --error="$WORKSPACE_PATH/logs/worker_%j.err" \\
  --export=ALL,RUN_ID="${{RUN_ID:-run}}",AGENT_ID="${{AGENT_ID}}",RESULTS_ROOT="${{RESULTS_ROOT}}" \\
  "$WORKSPACE_PATH/worker_loop.sh")

if [ -z "$JOB_ID" ]; then
  echo "ERROR: sbatch returned empty job ID" >&2
  exit 1
fi

# Wait for the job to leave the queue and actually start running.
# This blocks until a GPU is allocated — agents start staggered as GPUs free up.
echo "Worker job $JOB_ID submitted. Waiting for GPU allocation..." >&2
while true; do
  STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null | tr -d '[:space:]')
  if [ "$STATE" = "RUNNING" ]; then
    echo "GPU allocated for $AGENT_ID (job $JOB_ID)." >&2
    date -Iseconds > "$WORKSPACE_PATH/gpu_allocated_at"
    sleep 3   # give the worker loop a moment to start polling
    date -Iseconds > "$WORKSPACE_PATH/worker_ready_at"
    break
  elif [ -z "$STATE" ] || [ "$STATE" = "FAILED" ] || [ "$STATE" = "CANCELLED" ] || [ "$STATE" = "TIMEOUT" ]; then
    echo "ERROR: worker job $JOB_ID failed to start (state: ${{STATE:-gone}})" >&2
    exit 1
  fi
  sleep 10
done

echo "$JOB_ID"
"""
    out = workspace / "start_gpu_worker.sh"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out


def generate_run_on_worker_sh(
    workspace: Path,
    train_budget_seconds: int = 600,
) -> Path:
    """Write run_on_worker.sh into workspace.

    Drops run.trigger, then blocks until run.result appears (or timeout).
    Prints the content of run.result (TRAINING DONE / FAILED + val_bpb).

    Agent calls this once per iteration after modifying train.py:
        bash run_on_worker.sh
    """
    # Give a 20 % margin over the training budget before declaring timeout
    timeout = int(train_budget_seconds * 1.2) + 30

    script = f"""#!/bin/bash
# run_on_worker.sh — trigger one training run on the persistent GPU worker
# and block until the result is available.
WORKSPACE_PATH="{workspace}"
TIMEOUT={timeout}

# Fail fast if no worker job is running
if ! squeue -u "$USER" -n "worker_*" -h -o "%i" 2>/dev/null | grep -q .; then
  echo "TRAINING FAILED: no GPU worker running — call start_gpu_worker.sh first"
  exit 1
fi

rm -f "$WORKSPACE_PATH/run.result"
touch "$WORKSPACE_PATH/run.trigger"

elapsed=0
while [ $elapsed -lt $TIMEOUT ]; do
  if [ -f "$WORKSPACE_PATH/run.result" ]; then
    # Worker may report FAILED even when val_bpb is present (NFS grep issue on compute node).
    # Re-verify from login node which has correct NFS view.
    VAL=$(grep 'val_bpb:' "$WORKSPACE_PATH/logs/train_current.out" 2>/dev/null | head -1 | awk '{{print $2}}')
    VRAM=$(grep 'peak_vram_mb:' "$WORKSPACE_PATH/logs/train_current.out" 2>/dev/null | head -1 | awk '{{print $2}}')
    if [ -n "$VAL" ]; then
      echo "TRAINING DONE"
      echo "val_bpb: $VAL"
      [ -n "$VRAM" ] && echo "peak_vram_mb: $VRAM"
    else
      cat "$WORKSPACE_PATH/run.result"
    fi
    exit 0
  fi
  sleep 3
  elapsed=$((elapsed + 3))
done

echo "TRAINING FAILED: timeout after ${{TIMEOUT}}s waiting for worker result"
exit 1
"""
    out = workspace / "run_on_worker.sh"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out


def generate_stop_gpu_worker_sh(workspace: Path) -> Path:
    """Write stop_gpu_worker.sh into workspace.

    Signals the worker loop to exit cleanly, then cancels the SLURM job.
    Agent calls this at the very end of its loop:
        bash stop_gpu_worker.sh $WORKER_JOB_ID
    """
    script = f"""#!/bin/bash
# stop_gpu_worker.sh — gracefully shut down the persistent GPU worker
WORKSPACE_PATH="{workspace}"
WORKER_JOB_ID="${{1:-}}"

touch "$WORKSPACE_PATH/stop_worker"
sleep 3

if [ -n "$WORKER_JOB_ID" ]; then
  scancel "$WORKER_JOB_ID" 2>/dev/null || true
  echo "Worker job $WORKER_JOB_ID cancelled."
fi
"""
    out = workspace / "stop_gpu_worker.sh"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out


# ---------------------------------------------------------------------------
# Local scripts (kept for non-SLURM development)
# ---------------------------------------------------------------------------

def generate_run_training_sh(workspace: Path, train_budget_seconds: int) -> Path:
    """Write run_training.sh (local execution) into workspace."""
    uv_bin = _find_bin("uv")
    path_additions = _path_additions()

    script = f"""#!/bin/bash
export PATH="{path_additions}:$PATH"
cd "{workspace}"
# Kill any previous training still running
pkill -f "uv run train.py" 2>/dev/null || true
sleep 1
# Start training in background — agent polls run.log for results
nohup {uv_bin} run train.py > run.log 2>&1 &
echo "Training started (PID=$!, budget={train_budget_seconds}s). Poll with: ./check_training.sh"
"""
    out = workspace / "run_training.sh"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out


def generate_check_training_sh(workspace: Path) -> Path:
    """Write check_training.sh (local execution) into workspace."""
    script = f"""#!/bin/bash
cd "{workspace}"
if pgrep -f "uv run train.py" > /dev/null 2>&1; then
  echo "TRAINING RUNNING"
  tail -c 300 run.log 2>/dev/null | grep -oP 'step \\d+.*' | tail -1
else
  echo "TRAINING DONE"
  grep "^val_bpb:\\|^peak_vram_mb:" run.log 2>/dev/null || echo "No results (check run.log for errors)"
fi
"""
    out = workspace / "check_training.sh"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out


def generate_snapshot_helpers(
    workspace: Path,
    agent_id: str,
    results_root: Path,
) -> None:
    """Generate save_snapshot.py and update_snapshot.py into workspace.

    These are called by the sub-agent after each train.py modification and
    after each training evaluation to persist snapshots and reasoning traces.
    """
    from agent_swarms.snapshotting import (
        generate_save_snapshot_py,
        generate_update_snapshot_py,
    )
    generate_save_snapshot_py(workspace, agent_id, results_root)
    generate_update_snapshot_py(workspace, results_root)


def _find_bin(name: str) -> str:
    """Find a binary, falling back to ~/.local/bin/<name>."""
    import shutil
    found = shutil.which(name)
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / name
    if fallback.exists():
        return str(fallback)
    return name


def _path_additions() -> str:
    """Return extra PATH entries needed for uv/python3."""
    additions = [
        str(Path.home() / ".local" / "bin"),
        str(Path.home() / "miniforge3" / "bin"),
    ]
    return ":".join(additions)
