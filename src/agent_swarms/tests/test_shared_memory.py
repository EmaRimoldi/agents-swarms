"""Unit tests for the SharedMemory blackboard.

Tests cover:
- Basic write / read cycle
- Filtering by agent_id (read_others)
- Concurrent writes from multiple processes (correctness under flock)
- max_context_entries truncation
- Malformed JSON lines are skipped gracefully
- format_for_context renders each entry_type correctly
"""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import time
from pathlib import Path

import pytest

from agent_swarms.shared_memory import (
    ENTRY_HYPOTHESIS,
    ENTRY_INSIGHT,
    ENTRY_RESULT,
    ENTRY_STATUS,
    SharedMemory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sm(tmp_path: Path) -> SharedMemory:
    return SharedMemory(tmp_path / "shared_memory.jsonl", max_context_entries=20)


# ---------------------------------------------------------------------------
# Basic write / read
# ---------------------------------------------------------------------------


def test_write_creates_file(sm: SharedMemory) -> None:
    sm.write("agent_0", ENTRY_STATUS, {"message": "hello"})
    assert sm.path.exists()
    lines = [l for l in sm.path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["agent_id"] == "agent_0"
    assert entry["entry_type"] == ENTRY_STATUS
    assert entry["content"]["message"] == "hello"


def test_multiple_writes_append(sm: SharedMemory) -> None:
    for i in range(5):
        sm.write("agent_0", ENTRY_RESULT, {"step": i, "val_bpb": 1.1 - i * 0.01})
    lines = [l for l in sm.path.read_text().splitlines() if l.strip()]
    assert len(lines) == 5


def test_read_all_returns_all_entries(sm: SharedMemory) -> None:
    sm.write("agent_0", ENTRY_RESULT, {"step": 1, "val_bpb": 1.1})
    sm.write("agent_1", ENTRY_RESULT, {"step": 1, "val_bpb": 1.09})
    entries = sm.read_all()
    assert len(entries) == 2
    agent_ids = {e["agent_id"] for e in entries}
    assert agent_ids == {"agent_0", "agent_1"}


# ---------------------------------------------------------------------------
# read_others filtering
# ---------------------------------------------------------------------------


def test_read_others_excludes_own_entries(sm: SharedMemory) -> None:
    sm.write("agent_0", ENTRY_RESULT, {"step": 1, "val_bpb": 1.1})
    sm.write("agent_1", ENTRY_RESULT, {"step": 1, "val_bpb": 1.09})
    sm.write("agent_0", ENTRY_RESULT, {"step": 2, "val_bpb": 1.08})

    others_for_0 = sm.read_others("agent_0")
    assert len(others_for_0) == 1
    assert others_for_0[0]["agent_id"] == "agent_1"

    others_for_1 = sm.read_others("agent_1")
    assert len(others_for_1) == 2
    assert all(e["agent_id"] == "agent_0" for e in others_for_1)


def test_read_others_empty_when_only_own(sm: SharedMemory) -> None:
    sm.write("agent_0", ENTRY_STATUS, {"message": "started"})
    assert sm.read_others("agent_0") == []


# ---------------------------------------------------------------------------
# max_context_entries truncation
# ---------------------------------------------------------------------------


def test_max_context_entries_truncates(tmp_path: Path) -> None:
    sm = SharedMemory(tmp_path / "sm.jsonl", max_context_entries=3)
    for i in range(10):
        sm.write("agent_1", ENTRY_RESULT, {"step": i, "val_bpb": 1.0 + i * 0.01})
    entries = sm.read_others("agent_0")
    # Should return the 3 most recent entries
    assert len(entries) == 3
    steps = [e["content"]["step"] for e in entries]
    assert steps == [7, 8, 9]


# ---------------------------------------------------------------------------
# Malformed JSON resilience
# ---------------------------------------------------------------------------


def test_malformed_lines_skipped(sm: SharedMemory) -> None:
    sm.write("agent_0", ENTRY_RESULT, {"step": 1, "val_bpb": 1.1})
    # Inject a corrupt line directly
    with open(sm.path, "a") as f:
        f.write("this is not json\n")
    sm.write("agent_1", ENTRY_RESULT, {"step": 1, "val_bpb": 1.09})

    entries = sm.read_all()
    # 2 valid entries; corrupt line silently skipped
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# Concurrent writes from multiple processes
# ---------------------------------------------------------------------------


def _writer_process(path_str: str, agent_id: str, n_writes: int) -> None:
    """Worker function: write n_writes entries to the shared memory."""
    sm = SharedMemory(Path(path_str), max_context_entries=1000)
    for i in range(n_writes):
        sm.write(agent_id, ENTRY_RESULT, {"step": i, "val_bpb": 1.0 + i * 0.001})
        time.sleep(0.001)  # tiny jitter to interleave writes


def test_concurrent_writes_no_corruption(tmp_path: Path) -> None:
    """N processes writing simultaneously must not corrupt the JSONL file."""
    path = tmp_path / "sm_concurrent.jsonl"
    n_agents = 4
    n_writes_each = 20

    processes = [
        multiprocessing.Process(
            target=_writer_process,
            args=(str(path), f"agent_{i}", n_writes_each),
        )
        for i in range(n_agents)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=30)

    sm = SharedMemory(path)
    entries = sm.read_all()
    assert len(entries) == n_agents * n_writes_each, (
        f"Expected {n_agents * n_writes_each} entries, got {len(entries)}. "
        "Likely a lost write due to missing flock."
    )
    # Every entry must be valid JSON (no partial writes)
    for entry in entries:
        assert "agent_id" in entry
        assert "timestamp" in entry
        assert "entry_type" in entry
        assert "content" in entry


# ---------------------------------------------------------------------------
# format_for_context
# ---------------------------------------------------------------------------


def test_format_empty_returns_empty_string(sm: SharedMemory) -> None:
    assert sm.format_for_context([]) == ""


def test_format_result_entry(sm: SharedMemory) -> None:
    entries = [
        {
            "agent_id": "agent_1",
            "entry_type": ENTRY_RESULT,
            "timestamp": "2026-04-03T14:22:05+00:00",
            "content": {
                "step": 3,
                "hypothesis": "increase LR to 3e-4",
                "val_bpb": 1.0845,
                "accepted": True,
                "reason": "improved by 0.012",
            },
        }
    ]
    text = sm.format_for_context(entries)
    assert "SWARM UPDATE" in text
    assert "agent_1" in text
    assert "ACCEPTED" in text
    assert "1.0845" in text
    assert "increase LR to 3e-4" in text


def test_format_rejected_result(sm: SharedMemory) -> None:
    entries = [
        {
            "agent_id": "agent_1",
            "entry_type": ENTRY_RESULT,
            "timestamp": "2026-04-03T14:22:05+00:00",
            "content": {
                "step": 4,
                "hypothesis": "increase LR to 2e-3",
                "val_bpb": 1.1100,
                "accepted": False,
                "reason": "val_bpb got worse",
            },
        }
    ]
    text = sm.format_for_context(entries)
    assert "REJECTED" in text


def test_format_hypothesis_entry(sm: SharedMemory) -> None:
    entries = [
        {
            "agent_id": "agent_1",
            "entry_type": ENTRY_HYPOTHESIS,
            "timestamp": "2026-04-03T14:22:05+00:00",
            "content": {"step": 5, "hypothesis": "try cosine annealing"},
        }
    ]
    text = sm.format_for_context(entries)
    assert "Planning Step 5" in text
    assert "cosine annealing" in text


def test_format_insight_entry(sm: SharedMemory) -> None:
    entries = [
        {
            "agent_id": "agent_1",
            "entry_type": ENTRY_INSIGHT,
            "timestamp": "2026-04-03T14:22:05+00:00",
            "content": {"insight": "LR > 5e-4 diverges with batch_size=32"},
        }
    ]
    text = sm.format_for_context(entries)
    assert "Insight" in text
    assert "LR > 5e-4" in text


def test_format_ends_with_guidance(sm: SharedMemory) -> None:
    entries = [
        {
            "agent_id": "agent_1",
            "entry_type": ENTRY_STATUS,
            "timestamp": "2026-04-03T14:22:05+00:00",
            "content": {"message": "started"},
        }
    ]
    text = sm.format_for_context(entries)
    assert "avoid redundant experiments" in text


# ---------------------------------------------------------------------------
# File creation on init
# ---------------------------------------------------------------------------


def test_init_creates_file_if_missing(tmp_path: Path) -> None:
    path = tmp_path / "subdir" / "sm.jsonl"
    assert not path.exists()
    sm = SharedMemory(path)
    assert path.exists()


def test_init_does_not_overwrite_existing(tmp_path: Path) -> None:
    path = tmp_path / "sm.jsonl"
    sm = SharedMemory(path)
    sm.write("agent_0", ENTRY_STATUS, {"message": "first"})
    # Re-init should not wipe the file
    sm2 = SharedMemory(path)
    assert len(sm2.read_all()) == 1
