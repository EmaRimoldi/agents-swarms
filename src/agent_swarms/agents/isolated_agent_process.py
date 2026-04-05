"""Wraps a ClaudeAgentRunner in an isolated subprocess.

Each agent runs in a completely separate Python process to ensure
zero cross-agent context, state, or file access during the run.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from agent_swarms.config import AgentConfig


def _agent_worker(
    agent_config_dict: dict,
    workspace_str: str,
    agent_dir_str: str,
    run_id: str,
    experiment_id: str,
    system_prompt: str,
    first_message: str,
) -> None:
    """Worker function run in a separate process. Imports are local to avoid
    any shared state from the parent process."""
    # Re-add src to path since the child process starts fresh
    src_dir = Path(__file__).parents[3]  # agent_parallelisation_new/src
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from agent_swarms.config import AgentConfig
    from agent_swarms.agents.claude_agent_runner import ClaudeAgentRunner

    config = AgentConfig(**agent_config_dict)
    workspace = Path(workspace_str)
    agent_dir = Path(agent_dir_str)

    runner = ClaudeAgentRunner(config=config, workspace=workspace, agent_dir=agent_dir)

    # When the orchestrator terminates this process (SIGTERM), also kill the
    # active claude subprocess so it doesn't linger as an orphan.
    def _sigterm_handler(signum, frame):
        if runner._active_proc is not None:
            try:
                runner._active_proc.kill()
            except Exception:
                pass
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    runner.run(
        run_id=run_id,
        experiment_id=experiment_id,
        system_prompt=system_prompt,
        first_message=first_message,
    )


class IsolatedAgentProcess:
    """Runs one agent in an isolated subprocess."""

    def __init__(
        self,
        config: AgentConfig,
        workspace: Path,
        agent_dir: Path,
        run_id: str,
        experiment_id: str,
        system_prompt: str,
        first_message: str,
    ):
        self.config = config
        self.workspace = workspace
        self.agent_dir = agent_dir
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.system_prompt = system_prompt
        self.first_message = first_message
        self._process: Optional[multiprocessing.Process] = None

    def start(self) -> None:
        """Launch the agent in a new process."""
        self._process = multiprocessing.Process(
            target=_agent_worker,
            args=(
                self.config.to_dict(),
                str(self.workspace),
                str(self.agent_dir),
                self.run_id,
                self.experiment_id,
                self.system_prompt,
                self.first_message,
            ),
            daemon=True,
        )
        self._process.start()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._process is not None:
            self._process.join(timeout=timeout)

    def terminate(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()

    def kill(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.kill()

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None
