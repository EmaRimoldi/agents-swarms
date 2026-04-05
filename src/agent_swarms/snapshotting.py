"""Snapshot management for train.py changes.

Every time an agent modifies train.py, a snapshot is saved with metadata:
- the train.py content at that point
- the agent's hypothesis and expected effect
- the evaluation result (val_bpb before/after, accepted/rejected)
- git commit hash and message

Directory layout per agent:
    {agent_dir}/snapshots/
        step_000/
            train.py
            metadata.json
        step_001/
            train.py
            metadata.json
        ...

The snapshot helper script (save_snapshot.py) is generated into each agent's
workspace so that the sub-agent can call it directly from bash.
"""

from __future__ import annotations

import json
import shutil
import stat
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class SnapshotMetadata:
    """Complete metadata for one train.py snapshot."""
    step_index: int
    timestamp: str
    agent_id: str

    # Git context
    git_commit: str = ""
    git_message: str = ""

    # Reasoning context
    hypothesis: str = ""
    expected_effect: str = ""
    evidence: str = ""

    # Evaluation results
    val_bpb_before: Optional[float] = None
    val_bpb_after: Optional[float] = None
    accepted: Optional[bool] = None
    reason: str = ""

    # Change metadata
    changed_files: list = field(default_factory=lambda: ["train.py"])
    evaluation_command: str = "uv run train.py"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SnapshotMetadata":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SnapshotManager:
    """Manages train.py snapshots for one agent.

    Usage:
        manager = SnapshotManager(agent_dir / "snapshots")
        step = manager.next_step_index()
        meta = SnapshotMetadata(step_index=step, timestamp=..., agent_id="agent_0",
                                hypothesis="reduce LR", ...)
        manager.save(workspace / "train.py", meta)
        # ... after training ...
        manager.update(step, val_bpb_after=1.23, accepted=True, reason="improved")
    """

    def __init__(self, snapshots_dir: Path):
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def next_step_index(self) -> int:
        """Return the next step index (0-based)."""
        existing = sorted(self.snapshots_dir.glob("step_*"))
        if not existing:
            return 0
        try:
            last_name = existing[-1].name  # e.g. "step_007"
            return int(last_name.split("_")[1]) + 1
        except (IndexError, ValueError):
            return len(existing)

    def save(self, train_py_path: Path, metadata: SnapshotMetadata) -> Path:
        """Copy train.py and write metadata.json. Returns the snapshot dir."""
        snap_dir = self.snapshots_dir / f"step_{metadata.step_index:03d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_py_path, snap_dir / "train.py")
        (snap_dir / "metadata.json").write_text(
            json.dumps(metadata.to_dict(), indent=2)
        )
        return snap_dir

    def update(self, step_index: int, **kwargs) -> None:
        """Merge kwargs into an existing snapshot's metadata.json."""
        meta_path = self.snapshots_dir / f"step_{step_index:03d}" / "metadata.json"
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text())
        meta.update(kwargs)
        meta_path.write_text(json.dumps(meta, indent=2))

    def list_snapshots(self) -> list[SnapshotMetadata]:
        """Return all snapshots sorted by step index."""
        result = []
        for snap_dir in sorted(self.snapshots_dir.glob("step_*")):
            meta_path = snap_dir / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                result.append(SnapshotMetadata.from_dict(json.loads(meta_path.read_text())))
            except Exception:
                pass
        return result

    def get_snapshot_dir(self, step_index: int) -> Optional[Path]:
        d = self.snapshots_dir / f"step_{step_index:03d}"
        return d if d.exists() else None

    def best_snapshot(self) -> Optional[SnapshotMetadata]:
        """Return the snapshot with the lowest val_bpb_after (None if none exist)."""
        candidates = [s for s in self.list_snapshots() if s.val_bpb_after is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.val_bpb_after)  # type: ignore[arg-type]

    def accepted_snapshots(self) -> list[SnapshotMetadata]:
        return [s for s in self.list_snapshots() if s.accepted is True]

    def informative_snapshots(self, top_k: int = 5) -> list[SnapshotMetadata]:
        """Return top_k snapshots most informative for merge phase.

        Selects:
        - The best-performing snapshot
        - Snapshots around metric jumps (large val_bpb improvements)
        - The final snapshot
        """
        snaps = self.list_snapshots()
        if not snaps:
            return []

        chosen: dict[int, SnapshotMetadata] = {}

        # Always include last
        chosen[snaps[-1].step_index] = snaps[-1]

        # Best by val_bpb_after
        best = self.best_snapshot()
        if best:
            chosen[best.step_index] = best

        # Large single-step improvements
        prev_bpb: Optional[float] = None
        for snap in snaps:
            if snap.val_bpb_after is not None:
                if prev_bpb is not None and (prev_bpb - snap.val_bpb_after) > 0.002:
                    chosen[snap.step_index] = snap
                prev_bpb = snap.val_bpb_after

        # Fill up to top_k with accepted snapshots
        if len(chosen) < top_k:
            for snap in self.accepted_snapshots():
                if snap.step_index not in chosen:
                    chosen[snap.step_index] = snap
                if len(chosen) >= top_k:
                    break

        return sorted(chosen.values(), key=lambda s: s.step_index)


# ---------------------------------------------------------------------------
# Workspace helper scripts (written into agent workspace by training_harness)
# ---------------------------------------------------------------------------

def generate_save_snapshot_py(
    workspace: Path,
    agent_id: str,
    results_root: Path,
) -> Path:
    """Write save_snapshot.py into workspace.

    Usage (from within workspace):
        python save_snapshot.py <step> <hypothesis> <expected_effect> [val_bpb_before]

    Saves train.py snapshot and logs a reasoning trace entry.
    """
    snapshots_dir = results_root.parent / "snapshots"
    reasoning_dir = results_root.parent / "reasoning"

    script = f'''#!/usr/bin/env python3
"""Save a train.py snapshot and reasoning trace entry.

Usage:
    python save_snapshot.py <step_index> <hypothesis> <expected_effect> [val_bpb_before]

Called by the sub-agent before each training run, after modifying train.py.
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).parent.resolve()
SNAPSHOTS_DIR = Path("{snapshots_dir}")
REASONING_DIR = Path("{reasoning_dir}")
AGENT_ID = os.environ.get("AGENT_ID", "{agent_id}")

def main():
    args = sys.argv[1:]
    if len(args) < 3:
        print("Usage: python save_snapshot.py <step> <hypothesis> <expected_effect> [val_bpb_before]")
        sys.exit(1)

    step = int(args[0])
    hypothesis = args[1]
    expected_effect = args[2]
    val_bpb_before = float(args[3]) if len(args) > 3 else None

    # --- Snapshot ---
    snap_dir = SNAPSHOTS_DIR / f"step_{{step:03d}}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WORKSPACE / "train.py", snap_dir / "train.py")

    # Git info
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=WORKSPACE
    ).stdout.strip() or "uncommitted"
    git_message = subprocess.run(
        ["git", "log", "-1", "--format=%s"], capture_output=True, text=True, cwd=WORKSPACE
    ).stdout.strip() or ""

    metadata = {{
        "step_index": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": AGENT_ID,
        "git_commit": git_commit,
        "git_message": git_message,
        "hypothesis": hypothesis,
        "expected_effect": expected_effect,
        "evidence": "",
        "val_bpb_before": val_bpb_before,
        "val_bpb_after": None,
        "accepted": None,
        "reason": "",
        "changed_files": ["train.py"],
        "evaluation_command": "uv run train.py",
    }}
    (snap_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # --- Reasoning trace ---
    REASONING_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = REASONING_DIR / "trace.jsonl"
    entry = {{
        "step_index": step,
        "timestamp": metadata["timestamp"],
        "agent_id": AGENT_ID,
        "hypothesis": hypothesis,
        "expected_effect": expected_effect,
        "val_bpb_before": val_bpb_before,
        "val_bpb_after": None,
        "confirmed": None,
        "next_step": "",
    }}
    with open(trace_path, "a") as f:
        f.write(json.dumps(entry) + "\\n")

    print(f"Snapshot saved: step_{{step:03d}} — {{hypothesis[:60]}}")

if __name__ == "__main__":
    main()
'''
    out = workspace / "save_snapshot.py"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out


def generate_update_snapshot_py(
    workspace: Path,
    results_root: Path,
) -> Path:
    """Write update_snapshot.py into workspace.

    Usage:
        python update_snapshot.py <step> <val_bpb_after> <accepted> <reason> [next_step]

    Updates both the snapshot metadata and the reasoning trace after training.
    """
    snapshots_dir = results_root.parent / "snapshots"
    reasoning_dir = results_root.parent / "reasoning"

    script = f'''#!/usr/bin/env python3
"""Update snapshot metadata and reasoning trace after a training run.

Usage:
    python update_snapshot.py <step_index> <val_bpb_after> <accepted> <reason> [next_step]

    accepted: "true" or "false"
    val_bpb_after: float or "null" (for crashes)

Called after ./check_training.sh returns a result.
"""
import json
import sys
from pathlib import Path

SNAPSHOTS_DIR = Path("{snapshots_dir}")
REASONING_DIR = Path("{reasoning_dir}")

def main():
    args = sys.argv[1:]
    if len(args) < 4:
        print("Usage: python update_snapshot.py <step> <val_bpb_after> <accepted> <reason> [next_step]")
        sys.exit(1)

    step = int(args[0])
    val_bpb_after = float(args[1]) if args[1] not in ("null", "None", "") else None
    accepted = args[2].lower() in ("true", "yes", "keep", "1")
    reason = args[3]
    next_step_str = args[4] if len(args) > 4 else ""

    # Update snapshot metadata
    meta_path = SNAPSHOTS_DIR / f"step_{{step:03d}}" / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["val_bpb_after"] = val_bpb_after
        meta["accepted"] = accepted
        meta["reason"] = reason
        meta_path.write_text(json.dumps(meta, indent=2))

    # Update reasoning trace
    trace_path = REASONING_DIR / "trace.jsonl"
    if trace_path.exists():
        lines = trace_path.read_text().splitlines()
        new_lines = []
        updated = False
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                new_lines.insert(0, line)
                continue
            try:
                entry = json.loads(stripped)
                if entry.get("step_index") == step and not updated:
                    entry["val_bpb_after"] = val_bpb_after
                    entry["accepted"] = accepted
                    entry["reason"] = reason
                    entry["next_step"] = next_step_str
                    # Determine confirmation status
                    if val_bpb_after is None:
                        entry["confirmed"] = "crash"
                    elif accepted:
                        entry["confirmed"] = "confirmed"
                    else:
                        entry["confirmed"] = "falsified"
                    new_lines.insert(0, json.dumps(entry))
                    updated = True
                else:
                    new_lines.insert(0, line)
            except Exception:
                new_lines.insert(0, line)
        trace_path.write_text("\\n".join(new_lines) + "\\n")

    status = "kept" if accepted else "reverted"
    bpb_str = f"{{val_bpb_after:.6f}}" if val_bpb_after is not None else "crash"
    print(f"Step {{step:03d}} updated: val_bpb={{bpb_str}} status={{status}}")

if __name__ == "__main__":
    main()
'''
    out = workspace / "update_snapshot.py"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out
