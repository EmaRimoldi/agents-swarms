"""Structured reasoning trace for agent decision dynamics.

For each agent step the trace records:
  - the hypothesis / intended improvement
  - evidence from prior runs that motivated the change
  - the files changed
  - the expected metric impact
  - the observed effect after evaluation
  - whether the hypothesis was confirmed, partially confirmed, or falsified
  - the next intended direction

Stored as JSONL at {agent_dir}/reasoning/trace.jsonl (one JSON object per line).

The merge orchestrator reads these traces to understand each agent's search
trajectory and to ground the merge decision in deliberative evidence rather
than just endpoint metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ReasoningEntry:
    """One step in an agent's reasoning trace."""
    step_index: int
    timestamp: str
    agent_id: str

    # Decision context
    objective: str = ""
    current_best_val_bpb: Optional[float] = None
    hypothesis: str = ""
    evidence: str = ""                # what prior data motivated this change
    proposed_change: str = ""         # description of the code change

    # Prediction vs outcome
    expected_effect: str = ""
    observed_effect: str = ""
    val_bpb_before: Optional[float] = None
    val_bpb_after: Optional[float] = None

    # Epistemic update
    confirmed: Optional[str] = None   # "confirmed" | "partial" | "falsified" | "crash"
    confidence_update: str = ""
    next_step: str = ""

    # Bookkeeping
    accepted: Optional[bool] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReasoningEntry":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class ReasoningTracer:
    """Reads and writes structured reasoning traces for one agent.

    Trace file: {reasoning_dir}/trace.jsonl
    One JSON object per line, one entry per agent step.
    """

    def __init__(self, reasoning_dir: Path, agent_id: str):
        self.reasoning_dir = Path(reasoning_dir)
        self.agent_id = agent_id
        self.reasoning_dir.mkdir(parents=True, exist_ok=True)
        self._trace_path = self.reasoning_dir / "trace.jsonl"

    def append(self, entry: ReasoningEntry) -> None:
        """Append a new reasoning entry."""
        with open(self._trace_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def update_step(self, step_index: int, **kwargs) -> None:
        """Update the most recent entry with matching step_index."""
        if not self._trace_path.exists():
            return
        lines = self._trace_path.read_text().splitlines()
        new_lines: list[str] = []
        updated = False
        # Walk in reverse so we update the last matching entry
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                new_lines.insert(0, line)
                continue
            try:
                entry = json.loads(stripped)
                if entry.get("step_index") == step_index and not updated:
                    entry.update(kwargs)
                    new_lines.insert(0, json.dumps(entry))
                    updated = True
                else:
                    new_lines.insert(0, line)
            except Exception:
                new_lines.insert(0, line)
        self._trace_path.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else "")
        )

    def read_all(self) -> list[ReasoningEntry]:
        """Return all entries in chronological order."""
        if not self._trace_path.exists():
            return []
        result = []
        for line in self._trace_path.read_text().splitlines():
            stripped = line.strip()
            if stripped:
                try:
                    result.append(ReasoningEntry.from_dict(json.loads(stripped)))
                except Exception:
                    pass
        return result

    def confirmed_steps(self) -> list[ReasoningEntry]:
        return [e for e in self.read_all() if e.confirmed == "confirmed"]

    def falsified_steps(self) -> list[ReasoningEntry]:
        return [e for e in self.read_all() if e.confirmed == "falsified"]

    def summarize(self) -> dict:
        """Return a high-level summary of this agent's reasoning dynamics."""
        entries = self.read_all()
        if not entries:
            return {"agent_id": self.agent_id, "total_steps": 0}

        confirmed = [e for e in entries if e.confirmed == "confirmed"]
        falsified = [e for e in entries if e.confirmed == "falsified"]
        crashes = [e for e in entries if e.confirmed == "crash"]

        bpbs = [e.val_bpb_after for e in entries if e.val_bpb_after is not None]
        best_bpb = min(bpbs) if bpbs else None
        first_bpb = bpbs[0] if bpbs else None

        return {
            "agent_id": self.agent_id,
            "total_steps": len(entries),
            "confirmed_hypotheses": len(confirmed),
            "falsified_hypotheses": len(falsified),
            "crashes": len(crashes),
            "best_val_bpb": best_bpb,
            "first_val_bpb": first_bpb,
            "improvement": (first_bpb - best_bpb) if (first_bpb and best_bpb) else None,
            "confirmed_hypotheses_text": [e.hypothesis for e in confirmed],
            "falsified_hypotheses_text": [e.hypothesis for e in falsified],
        }


# ---------------------------------------------------------------------------
# Helpers for reading across all agents
# ---------------------------------------------------------------------------

def collect_all_traces(experiment_dir: Path, mode: str) -> dict[str, list[ReasoningEntry]]:
    """Load reasoning traces for all agents in a mode directory."""
    mode_dir = experiment_dir / f"mode_{mode}"
    result: dict[str, list[ReasoningEntry]] = {}
    for agent_dir in sorted(mode_dir.glob("agent_*")):
        agent_id = agent_dir.name
        trace_path = agent_dir / "reasoning" / "trace.jsonl"
        if not trace_path.exists():
            result[agent_id] = []
            continue
        entries = []
        for line in trace_path.read_text().splitlines():
            stripped = line.strip()
            if stripped:
                try:
                    entries.append(ReasoningEntry.from_dict(json.loads(stripped)))
                except Exception:
                    pass
        result[agent_id] = entries
    return result


def summarize_all_traces(
    traces: dict[str, list[ReasoningEntry]],
) -> dict:
    """Produce a cross-agent reasoning summary for the merge orchestrator."""
    agent_summaries = []
    all_confirmed_hypotheses: list[str] = []
    all_falsified_hypotheses: list[str] = []

    for agent_id, entries in traces.items():
        confirmed = [e for e in entries if e.confirmed == "confirmed"]
        falsified = [e for e in entries if e.confirmed == "falsified"]
        all_confirmed_hypotheses.extend(e.hypothesis for e in confirmed)
        all_falsified_hypotheses.extend(e.hypothesis for e in falsified)

        bpbs = [e.val_bpb_after for e in entries if e.val_bpb_after is not None]
        agent_summaries.append({
            "agent_id": agent_id,
            "total_steps": len(entries),
            "confirmed": len(confirmed),
            "falsified": len(falsified),
            "best_val_bpb": min(bpbs) if bpbs else None,
        })

    # Find hypotheses confirmed by multiple agents
    from collections import Counter
    confirmed_counts = Counter(all_confirmed_hypotheses)
    independently_confirmed = [
        {"hypothesis": h, "count": c}
        for h, c in confirmed_counts.items()
        if c >= 2
    ]

    return {
        "agents": agent_summaries,
        "independently_confirmed_hypotheses": independently_confirmed,
        "all_confirmed_hypotheses": all_confirmed_hypotheses,
        "all_falsified_hypotheses": all_falsified_hypotheses,
    }
