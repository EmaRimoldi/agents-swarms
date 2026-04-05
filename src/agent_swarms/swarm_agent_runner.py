"""SwarmAgentRunner — ClaudeAgentRunner extended for swarm coordination.

Changes vs ClaudeAgentRunner:
1. SINGLE_TURN_MODE = True: each agent runs as one long claude --print call.
2. _build_env() adds SWARM_MEMORY_PATH so coordinator.py works inside Claude.
3. _watch_workspace_events() logs completed runs to run_agent.log only.
   Publishing to the blackboard is done exclusively by the agent via coordinator.py
   to avoid duplicate ENTRY_RESULT entries.

Everything else (budget tracking, SIGTERM handling, SLURM watchers)
is inherited from ClaudeAgentRunner unchanged.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Ensure src/ is on sys.path when this module is imported from a fresh subprocess context.
_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_swarms.agents.claude_agent_runner import (  # noqa: E402
    ClaudeAgentRunner,
    _log,
    _ts,
    _log_train_diff,
    _dump_slurm_failure_logs,
)
from agent_swarms.config import AgentConfig  # noqa: E402
from agent_swarms.shared_memory import SharedMemory  # noqa: E402


class SwarmAgentRunner(ClaudeAgentRunner):
    """ClaudeAgentRunner for swarm mode.

    Key differences vs ClaudeAgentRunner:
    - SINGLE_TURN_MODE = True: each agent runs as one long claude --print call
      (no continuation turns). The full time budget is used as the turn timeout.
    - Sets SWARM_MEMORY_PATH in the subprocess environment so that
      coordinator.py (placed in the workspace) can find the blackboard.
    - Does NOT automatically write to the blackboard: the agent does this
      by calling `python coordinator.py publish` as part of the swarm loop.
    """

    SINGLE_TURN_MODE: bool = True

    def __init__(
        self,
        config: AgentConfig,
        workspace: Path,
        agent_dir: Path,
        shared_memory: Optional[SharedMemory] = None,
    ) -> None:
        super().__init__(config=config, workspace=workspace, agent_dir=agent_dir)
        self.shared_memory = shared_memory

    # ------------------------------------------------------------------
    # Override: inject SWARM_MEMORY_PATH into subprocess environment
    # ------------------------------------------------------------------

    def _build_env(self, run_id: str, experiment_id: str) -> dict:
        env = super()._build_env(run_id, experiment_id)
        if self.shared_memory is not None:
            env["SWARM_MEMORY_PATH"] = str(self.shared_memory.path)
        return env

    # ------------------------------------------------------------------
    # Override: workspace event watcher — no automatic blackboard write
    # ------------------------------------------------------------------

    def _watch_workspace_events(
        self,
        log_fh,
        stop_event: threading.Event,
        observed_val_bpbs: list,
    ) -> None:
        """Log workspace events. The agent publishes results via coordinator.py."""
        ws = self.workspace
        agent_id = self.config.agent_id

        trigger = ws / "run.trigger"
        result = ws / "run.result"
        train_out = ws / "logs" / "train_current.out"
        train_py = ws / "train.py"
        results_tsv = ws / "results" / "results.tsv"

        trigger_seen = False
        result_seen = False
        run_count = 0
        run_wall_start: Optional[float] = None
        train_py_mtime: Optional[float] = None
        results_tsv_lines = 0
        train_out_lines = 0
        coordinator_log = ws / "logs" / "coordinator.log"
        coordinator_log_lines = 0

        # Phase keywords emitted by train.py's _phase() function
        _PHASE_KEYWORDS = (
            "Run started",
            "Compile phase:",
            "Warmup phase:",
            "Measured training phase:",
            "Evaluation phase:",
            "Run finished",
        )

        while not stop_event.is_set():
            # train.py modified → log diff
            try:
                mtime = train_py.stat().st_mtime if train_py.exists() else None
                if mtime is not None and mtime != train_py_mtime:
                    if train_py_mtime is not None:
                        _log(log_fh, f"[{agent_id}] train.py modified.")
                        _log_train_diff(train_py, log_fh, agent_id)
                    train_py_mtime = mtime
            except OSError:
                pass

            # results.tsv new row → log it
            try:
                if results_tsv.exists():
                    lines = [l for l in results_tsv.read_text().splitlines() if l.strip()]
                    if len(lines) > results_tsv_lines:
                        for row in lines[results_tsv_lines:]:
                            if not row.startswith("commit"):
                                _log(log_fh, f"[{agent_id}] results.tsv: {row}")
                        results_tsv_lines = len(lines)
            except OSError:
                pass

            # coordinator.log new entries → stream to run_agent.log
            try:
                if coordinator_log.exists():
                    all_lines = coordinator_log.read_text().splitlines()
                    new_lines = all_lines[coordinator_log_lines:]
                    for line in new_lines:
                        _log(log_fh, f"[{agent_id}] coord | {line}")
                    coordinator_log_lines = len(all_lines)
            except OSError:
                pass

            # run.trigger appeared → training started
            if not trigger_seen and trigger.exists():
                trigger_seen = True
                result_seen = False
                run_count += 1
                run_wall_start = time.time()
                # Start streaming from current end of file to avoid replaying
                # the previous run's output (train_current.out may not be truncated yet)
                try:
                    train_out_lines = len(train_out.read_text().splitlines()) if train_out.exists() else 0
                except OSError:
                    train_out_lines = 0
                _log(log_fh, f"[{agent_id}] Training run #{run_count} started.")

            # Stream phase lines from train_current.out while a run is active
            if trigger_seen and not result_seen:
                try:
                    if train_out.exists():
                        all_lines = train_out.read_text().splitlines()
                        new_lines = all_lines[train_out_lines:]
                        for line in new_lines:
                            if any(kw in line for kw in _PHASE_KEYWORDS):
                                log_fh.write(f"[{agent_id}]   {line.strip()}\n")
                        if new_lines:
                            log_fh.flush()
                        train_out_lines = len(all_lines)
                except OSError:
                    pass

            # run.result appeared → training finished
            if trigger_seen and not result_seen and result.exists():
                result_seen = True
                trigger_seen = False
                elapsed = (
                    f"{time.time() - run_wall_start:.0f}s"
                    if run_wall_start else "?s"
                )
                val_bpb = None
                try:
                    for src in (result, train_out):
                        if src.exists():
                            for line in src.read_text().splitlines():
                                if line.startswith("val_bpb:"):
                                    val_bpb = line.split(":", 1)[1].strip()
                                    break
                        if val_bpb:
                            break
                except OSError:
                    pass

                if val_bpb:
                    try:
                        observed_val_bpbs.append(float(val_bpb))
                    except ValueError:
                        pass
                    _log(
                        log_fh,
                        f"[{agent_id}] Training run #{run_count} done — "
                        f"val_bpb: {val_bpb} (elapsed: {elapsed})",
                    )
                else:
                    status = ""
                    try:
                        status = (
                            result.read_text().strip().splitlines()[0]
                            if result.exists() else "no result"
                        )
                    except OSError:
                        pass
                    _log(
                        log_fh,
                        f"[{agent_id}] Training run #{run_count} done — "
                        f"{status} (elapsed: {elapsed})",
                    )
                    _dump_slurm_failure_logs(ws, agent_id, run_count, log_fh)

            stop_event.wait(2)



# ---------------------------------------------------------------------------
# Isolated process wrapper for swarm agents
# ---------------------------------------------------------------------------


def _swarm_agent_worker(
    agent_config_dict: dict,
    workspace_str: str,
    agent_dir_str: str,
    run_id: str,
    experiment_id: str,
    system_prompt: str,
    first_message: str,
    shared_memory_path_str: Optional[str],
) -> None:
    """Target function for a swarm agent subprocess."""
    src_dir = Path(__file__).parent.parent  # agent_swarms/src/
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from agent_swarms.config import AgentConfig
    from agent_swarms.swarm_agent_runner import SwarmAgentRunner
    from agent_swarms.shared_memory import SharedMemory

    config = AgentConfig(**agent_config_dict)
    workspace = Path(workspace_str)
    agent_dir = Path(agent_dir_str)

    shared_memory: Optional[SharedMemory] = None
    if shared_memory_path_str:
        shared_memory = SharedMemory(
            Path(shared_memory_path_str),
            max_context_entries=20,
        )

    runner = SwarmAgentRunner(
        config=config,
        workspace=workspace,
        agent_dir=agent_dir,
        shared_memory=shared_memory,
    )

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


class IsolatedSwarmAgentProcess:
    """Runs one swarm agent in an isolated subprocess with shared memory access."""

    def __init__(
        self,
        config: AgentConfig,
        workspace: Path,
        agent_dir: Path,
        run_id: str,
        experiment_id: str,
        system_prompt: str,
        first_message: str,
        shared_memory_path: Optional[Path] = None,
    ):
        self.config = config
        self.workspace = workspace
        self.agent_dir = agent_dir
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.system_prompt = system_prompt
        self.first_message = first_message
        self.shared_memory_path = shared_memory_path
        self._process: Optional[multiprocessing.Process] = None

    def start(self) -> None:
        self._process = multiprocessing.Process(
            target=_swarm_agent_worker,
            args=(
                self.config.to_dict(),
                str(self.workspace),
                str(self.agent_dir),
                self.run_id,
                self.experiment_id,
                self.system_prompt,
                self.first_message,
                str(self.shared_memory_path) if self.shared_memory_path else None,
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
