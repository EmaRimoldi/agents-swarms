"""Select a representative W_t slice from a completed swarm run.

Procedure:
1. Read the source run's shared_memory.jsonl.
2. Locate the median 'result' entry (by ordinal) and record its
   timestamp t*.
3. Truncate the JSONL at t* (inclusive) and write it to
   <out_dir>/W_t.jsonl.
4. Capture the *byte-identical* W payload the swarm agent would
   have seen by invoking the production `coordinator.py think`
   command against the truncated JSONL, and save its stdout to
   <out_dir>/W_t_payload.txt.

The payload captured in (4) is the single source of truth fed to
the LLM during the with-W condition of the sanity experiment.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WSelection:
    source_run: Path
    source_jsonl: Path
    cut_index: int           # ordinal of the median 'result' entry
    cut_timestamp: str
    truncated_jsonl: Path
    payload_path: Path
    payload_bytes: int
    n_results_in_slice: int
    n_entries_in_slice: int


def _load_entries(jsonl: Path) -> list[dict]:
    out: list[dict] = []
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _median_result_timestamp(entries: list[dict]) -> tuple[int, str]:
    results = [(i, e) for i, e in enumerate(entries)
               if e.get("entry_type") == "result"]
    if not results:
        raise ValueError("no 'result' entries in shared_memory.jsonl")
    mid_ordinal = len(results) // 2  # median by ordinal among results
    line_idx, entry = results[mid_ordinal]
    return line_idx, entry["timestamp"]


def _write_truncated(entries: list[dict], cut_line_idx: int,
                     out_path: Path) -> int:
    kept = entries[: cut_line_idx + 1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for e in kept:
            f.write(json.dumps(e) + "\n")
    return len(kept)


def _capture_think_stdout(coordinator_py: Path, truncated_jsonl: Path,
                          agent_id: str = "sanity_observer") -> str:
    """Invoke `coordinator.py think` against the truncated JSONL and
    return its exact stdout. This is the byte-identical payload the
    production swarm agent would have seen at timestamp t*.
    """
    env = os.environ.copy()
    env["SWARM_MEMORY_PATH"] = str(truncated_jsonl.resolve())
    env["AGENT_ID"] = agent_id
    proc = subprocess.run(
        [sys.executable, str(coordinator_py), "think"],
        cwd=coordinator_py.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"coordinator.py think failed: rc={proc.returncode}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def select_w(
    source_run: Path,
    out_dir: Path,
    repo_root: Path,
) -> WSelection:
    source_run = Path(source_run)
    out_dir = Path(out_dir)
    source_jsonl = source_run / "mode_swarm" / "shared_memory.jsonl"
    if not source_jsonl.exists():
        raise FileNotFoundError(source_jsonl)

    entries = _load_entries(source_jsonl)
    cut_line_idx, cut_ts = _median_result_timestamp(entries)
    truncated = out_dir / "W_t.jsonl"
    n_kept = _write_truncated(entries, cut_line_idx, truncated)

    # The production coordinator.py is copied into agent workspaces,
    # but the module under src/agent_swarms/coordinator.py is the
    # authoritative source. Invoke it directly.
    coordinator_py = repo_root / "src" / "agent_swarms" / "coordinator.py"
    if not coordinator_py.exists():
        raise FileNotFoundError(coordinator_py)

    # coordinator.py imports sibling modules; copy it into a temp
    # dir alongside shared_memory.py so the import chain resolves
    # without touching sys.path.
    sandbox = out_dir / "_coordinator_sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    for fname in ("coordinator.py", "shared_memory.py"):
        src = repo_root / "src" / "agent_swarms" / fname
        shutil.copy2(src, sandbox / fname)
    coord_copy = sandbox / "coordinator.py"

    payload = _capture_think_stdout(coord_copy, truncated)

    payload_path = out_dir / "W_t_payload.txt"
    payload_path.write_text(payload, encoding="utf-8")

    n_results_in_slice = sum(
        1 for e in entries[: cut_line_idx + 1]
        if e.get("entry_type") == "result"
    )

    sel = WSelection(
        source_run=source_run,
        source_jsonl=source_jsonl,
        cut_index=cut_line_idx,
        cut_timestamp=cut_ts,
        truncated_jsonl=truncated,
        payload_path=payload_path,
        payload_bytes=len(payload.encode("utf-8")),
        n_results_in_slice=n_results_in_slice,
        n_entries_in_slice=n_kept,
    )
    (out_dir / "W_t_selection.json").write_text(
        json.dumps(
            {
                "source_run": str(sel.source_run),
                "source_jsonl": str(sel.source_jsonl),
                "cut_index": sel.cut_index,
                "cut_timestamp": sel.cut_timestamp,
                "truncated_jsonl": str(sel.truncated_jsonl),
                "payload_path": str(sel.payload_path),
                "payload_bytes": sel.payload_bytes,
                "n_results_in_slice": sel.n_results_in_slice,
                "n_entries_in_slice": sel.n_entries_in_slice,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return sel
