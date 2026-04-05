"""Wall-clock and iteration budget tracking.

Budget accounting:

  Parallel mode:
    agent_0: budget = T minutes
    agent_1: budget = T minutes
    total compute = 2T minutes (running on 2 GPUs simultaneously)
    wall-clock time ≈ T minutes

  Single-agent-longer mode:
    agent_0: budget = 2T minutes
    total compute = 2T minutes
    wall-clock time ≈ 2T minutes

  Budget matching:
    Both modes consume the same total compute budget.
    The parallel mode trades wall-clock time for exploration diversity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BudgetTracker:
    """Tracks wall-clock budget for one agent session.

    Args:
        wall_clock_budget_seconds: hard ceiling enforced by orchestrator
        train_time_budget_seconds: per-training-run timeout (default 300s)
        max_iterations: optional cap on number of training runs
        startup_deadline_seconds: time allowed before first successful turn
    """

    wall_clock_budget_seconds: int
    train_time_budget_seconds: int = 300
    max_iterations: Optional[int] = None
    startup_deadline_seconds: int = 600

    _start_time: float = field(default_factory=time.monotonic, init=False, repr=False)
    _budget_start_time: Optional[float] = field(default=None, init=False, repr=False)
    _iteration_count: int = field(default=0, init=False, repr=False)

    def start_budget_clock(self) -> None:
        """Call after the first successful agent turn to start the research clock."""
        if self._budget_start_time is None:
            self._budget_start_time = time.monotonic()

    def budget_started(self) -> bool:
        return self._budget_start_time is not None

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def budget_elapsed_seconds(self) -> float:
        if self._budget_start_time is None:
            return 0.0
        return time.monotonic() - self._budget_start_time

    def remaining_seconds(self) -> float:
        if self._budget_start_time is None:
            return float(self.wall_clock_budget_seconds)
        return max(0.0, self.wall_clock_budget_seconds - self.budget_elapsed_seconds())

    def remaining_minutes(self) -> int:
        return int(self.remaining_seconds() // 60)

    def is_expired(self) -> bool:
        if self._budget_start_time is None:
            return False
        return self.remaining_seconds() <= 30

    def startup_expired(self) -> bool:
        return (not self.budget_started()) and (
            self.elapsed_seconds() >= self.startup_deadline_seconds
        )

    def record_iteration(self) -> None:
        self._iteration_count += 1

    def iterations_exhausted(self) -> bool:
        if self.max_iterations is None:
            return False
        return self._iteration_count >= self.max_iterations

    def should_stop(self) -> bool:
        return self.is_expired() or self.iterations_exhausted()

    def refund_seconds(self, seconds: float) -> None:
        """Refund wasted time from failed turns back to the budget."""
        if self._budget_start_time is not None:
            self._budget_start_time += seconds
