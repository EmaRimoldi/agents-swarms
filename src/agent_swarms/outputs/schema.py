"""Output data structures: TrajectoryEntry, AgentResult, ExperimentSummary."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TrajectoryEntry:
    """One completed training run."""
    step: int
    val_bpb: float

    def to_json(self) -> str:
        return json.dumps({"step": self.step, "val_bpb": self.val_bpb})

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryEntry":
        return cls(step=int(d["step"]), val_bpb=float(d["val_bpb"]))


@dataclass
class AgentResult:
    """Aggregated results for one agent after it finishes."""
    agent_id: str
    experiment_id: str
    mode: str

    # Run metadata
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    budget_seconds: int = 0
    total_turns: int = 0

    # Training results
    trajectory: list[TrajectoryEntry] = field(default_factory=list)
    best_val_bpb: Optional[float] = None
    first_val_bpb: Optional[float] = None
    total_training_runs: int = 0
    successful_training_runs: int = 0

    # Paths
    workspace_path: str = ""
    results_path: str = ""

    # Status
    failed: bool = False
    failure_reason: str = ""

    def compute_derived(self) -> None:
        if self.trajectory:
            self.best_val_bpb = min(e.val_bpb for e in self.trajectory)
            self.first_val_bpb = self.trajectory[0].val_bpb
            self.total_training_runs = len(self.trajectory)
            self.successful_training_runs = len(self.trajectory)

    def improvement(self) -> Optional[float]:
        """Improvement from first to best run (negative = better)."""
        if self.first_val_bpb is not None and self.best_val_bpb is not None:
            return self.best_val_bpb - self.first_val_bpb
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trajectory"] = [{"step": e.step, "val_bpb": e.val_bpb} for e in self.trajectory]
        return d


@dataclass
class ExperimentSummary:
    """Summary of one full experiment (parallel or single-long)."""
    experiment_id: str
    mode: str
    agent_results: list[AgentResult] = field(default_factory=list)

    def best_agent(self) -> Optional[AgentResult]:
        successful = [r for r in self.agent_results if not r.failed and r.best_val_bpb is not None]
        if not successful:
            return None
        return min(successful, key=lambda r: r.best_val_bpb)

    def best_val_bpb(self) -> Optional[float]:
        best = self.best_agent()
        return best.best_val_bpb if best else None

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "mode": self.mode,
            "best_val_bpb": self.best_val_bpb(),
            "best_agent_id": (self.best_agent().agent_id if self.best_agent() else None),
            "agent_results": [r.to_dict() for r in self.agent_results],
        }
