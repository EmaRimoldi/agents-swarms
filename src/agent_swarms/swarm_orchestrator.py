"""SwarmOrchestrator — Orchestrator extended for inter-agent blackboard communication.

Differences from the base Orchestrator:
- Creates a single shared_memory.jsonl file before any agent starts.
- Launches agents via IsolatedSwarmAgentProcess (which passes the blackboard
  path into each worker) instead of IsolatedAgentProcess.
- run_swarm() replaces run_parallel() for swarm experiments.
- All other behaviour (poll loop, hard deadline, SIGTERM cleanup, atexit) is
  inherited unchanged.

The docstring constraint "Must NOT read one agent's results and pass them to
another" applies to the orchestrator itself.  In swarm mode, agents read each
other's results directly from the shared blackboard file — the orchestrator
only creates the file and passes its path to all agents.  The orchestrator
itself never reads or interprets the blackboard contents.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path when this module is imported from a fresh subprocess context.
_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_swarms.config import AgentConfig, ExperimentConfig  # noqa: E402
from agent_swarms.orchestrator import Orchestrator, _render_first_message  # noqa: E402
from agent_swarms.utils.workspace import create_workspace  # noqa: E402
from agent_swarms.shared_memory import SharedMemory  # noqa: E402
from agent_swarms.swarm_agent_runner import IsolatedSwarmAgentProcess  # noqa: E402
from agent_swarms.swarm_config import SwarmConfig  # noqa: E402


class SwarmOrchestrator(Orchestrator):
    """Coordinates a swarm experiment where agents share a blackboard.

    The orchestrator creates ONE shared_memory.jsonl file before agents start
    and passes its path to every agent.  The orchestrator itself never reads
    the blackboard — agents communicate peer-to-peer through the file.

    All other orchestrator behaviour (poll loop, hard deadline, SIGTERM cleanup,
    atexit, SLURM worker cancellation) is inherited from Orchestrator unchanged.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        repo_root: Path,
        swarm_config: SwarmConfig,
    ) -> None:
        super().__init__(config=config, repo_root=repo_root)
        self.swarm_config = swarm_config
        # Set during run_swarm so _setup_swarm_agent can pass it to create_workspace.
        self._swarm_memory_path: Path | None = None

    # ------------------------------------------------------------------
    # Public: run_swarm  (analogous to run_parallel)
    # ------------------------------------------------------------------

    def run_swarm(
        self,
        experiment_dir: Path,
        system_prompt: str,
        first_message_template: str,
    ) -> None:
        """Launch all swarm agents simultaneously with a shared blackboard.

        Steps:
        1. Create the mode directory and write experiment manifest.
        2. Create the shared blackboard file.
        3. Set up one isolated workspace per agent.
        4. Instantiate IsolatedSwarmAgentProcess for each agent (passing the
           blackboard path).
        5. Start all processes simultaneously.
        6. Poll until all finish or hit hard deadlines.
        """
        self._validate_gpu_assignments()
        mode_dir = experiment_dir / "mode_swarm"
        mode_dir.mkdir(parents=True, exist_ok=True)
        run_id = self.config.experiment_id

        # Write experiment manifest
        manifest_path = experiment_dir / "config.json"
        manifest_path.write_text(json.dumps(self.config.to_dict(), indent=2))

        # ── Create shared blackboard ────────────────────────────────────
        shared_memory_path = mode_dir / self.swarm_config.shared_memory_file
        self._swarm_memory_path = shared_memory_path  # for _setup_swarm_agent
        SharedMemory(
            path=shared_memory_path,
            max_context_entries=self.swarm_config.max_context_entries,
        )
        print(f"[swarm-orchestrator] Shared blackboard: {shared_memory_path}", flush=True)

        # ── Set up workspaces and build swarm processes ─────────────────
        processes: list[IsolatedSwarmAgentProcess] = []
        hard_deadlines: list[float] = []

        for agent_config in self.config.agents:
            agent_dir, workspace = self._setup_swarm_agent(agent_config, mode_dir, run_id)
            first_message = _render_first_message(
                template=first_message_template,
                agent_config=agent_config,
                run_id=run_id,
                experiment_id=self.config.experiment_id,
                workspace=workspace,
                branch_prefix="swarm",
            )
            proc = IsolatedSwarmAgentProcess(
                config=agent_config,
                workspace=workspace,
                agent_dir=agent_dir,
                run_id=run_id,
                experiment_id=self.config.experiment_id,
                system_prompt=system_prompt,
                first_message=first_message,
                shared_memory_path=shared_memory_path,
            )
            processes.append(proc)
            hard_deadlines.append(
                time.monotonic() + agent_config.time_budget_minutes * 60 * 3
            )

        self._register_cleanup()
        self._processes = processes  # type: ignore[assignment]

        # ── Launch all agents simultaneously ────────────────────────────
        for proc in processes:
            proc.start()

        print(
            f"[swarm-orchestrator] Launched {len(processes)} swarm agent(s) simultaneously.",
            flush=True,
        )

        # ── Wait for all agents to finish ───────────────────────────────
        self._wait_for_swarm(processes, hard_deadlines)

        print(
            f"[swarm-orchestrator] All {len(processes)} swarm agents finished.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Internal: swarm workspace setup (extends base _setup_agent)
    # ------------------------------------------------------------------

    def _setup_swarm_agent(
        self, agent_config: AgentConfig, mode_dir: Path, run_id: str
    ) -> tuple[Path, Path]:
        """Like Orchestrator._setup_agent but injects swarm_memory_path into workspace."""
        from agent_swarms.utils.workspace import create_workspace

        agent_dir = mode_dir / agent_config.agent_id
        workspace = agent_dir / "workspace"
        results_root = agent_dir / "results"

        branch_name = f"swarm/{self.config.experiment_id}/{agent_config.agent_id}"

        create_workspace(
            autoresearch_dir=self.autoresearch_dir,
            workspace_path=workspace,
            branch_name=branch_name,
            train_budget_seconds=agent_config.train_time_budget_seconds,
            run_id=run_id,
            agent_id=agent_config.agent_id,
            results_root=results_root,
            slurm_partition=self.config.slurm_partition,
            slurm_gres=self.config.slurm_gres,
            slurm_time=self.config.slurm_time,
            agent_time_budget_minutes=agent_config.time_budget_minutes,
            swarm_memory_path=self._swarm_memory_path,
        )
        (agent_dir / "logs").mkdir(parents=True, exist_ok=True)
        return agent_dir, workspace

    # ------------------------------------------------------------------
    # Internal: swarm-aware poll loop
    # ------------------------------------------------------------------

    def _wait_for_swarm(
        self,
        processes: list[IsolatedSwarmAgentProcess],
        hard_deadlines: list[float],
    ) -> None:
        """Poll until all swarm processes finish or hard deadlines hit.

        Identical logic to Orchestrator._wait_for_all but typed for
        IsolatedSwarmAgentProcess.
        """
        while True:
            now = time.monotonic()
            all_done = True
            for proc, deadline in zip(processes, hard_deadlines):
                if proc.is_alive():
                    if now >= deadline:
                        print(
                            f"[swarm-orchestrator] Hard deadline reached for "
                            f"{proc.config.agent_id}, sending SIGTERM.",
                            flush=True,
                        )
                        proc.terminate()
                        time.sleep(2)
                        proc.kill()
                    else:
                        all_done = False
            if all_done:
                break
            time.sleep(self.POLL_INTERVAL_SEC)
