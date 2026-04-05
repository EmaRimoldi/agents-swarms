"""Abstract AgentRunner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from agent_swarms.config import AgentConfig


class AgentRunner(ABC):
    """Base class for agent runners."""

    def __init__(self, config: AgentConfig, workspace: Path, agent_dir: Path):
        self.config = config
        self.workspace = workspace
        self.agent_dir = agent_dir
        self.results_dir = agent_dir / "results"
        self.logs_dir = agent_dir / "logs"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(
        self,
        run_id: str,
        experiment_id: str,
        system_prompt: str,
        first_message: str,
    ) -> None:
        """Run the agent until budget expires. Write metadata.json when done."""
        ...
