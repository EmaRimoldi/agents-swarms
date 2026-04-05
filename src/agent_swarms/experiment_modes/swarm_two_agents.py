"""Mode 3: N swarm agents × T budget, communicating via shared blackboard."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable from subprocess contexts.
_SRC = Path(__file__).parents[2]          # src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_swarms.config import ExperimentConfig  # noqa: E402
from agent_swarms.outputs.collector import collect_experiment  # noqa: E402
from agent_swarms.outputs.reporter import write_experiment_report  # noqa: E402
from agent_swarms.swarm_config import SwarmConfig  # noqa: E402
from agent_swarms.swarm_orchestrator import SwarmOrchestrator  # noqa: E402


def run_swarm_experiment(
    config: ExperimentConfig,
    experiment_dir: Path,
    repo_root: Path,
    system_prompt: str,
    first_message_template: str,
    swarm_config: SwarmConfig | None = None,
):
    """Run Mode 3: N swarm agents × T budget with shared blackboard.

    Agents:
    - Run simultaneously (same as parallel mode).
    - Each has the same time budget T.
    - Write results to a shared JSONL blackboard after each training run.
    - Receive a summary of other agents' findings in every continuation message.

    Total compute = N×T.  Wall-clock time ≈ T.
    """
    assert config.mode == "swarm", f"Expected mode=swarm, got {config.mode}"
    assert len(config.agents) >= 1, (
        f"Swarm mode expects at least 1 agent, got {len(config.agents)}"
    )

    if swarm_config is None:
        swarm_config = SwarmConfig()

    orchestrator = SwarmOrchestrator(
        config=config,
        repo_root=repo_root,
        swarm_config=swarm_config,
    )
    orchestrator.run_swarm(
        experiment_dir=experiment_dir,
        system_prompt=system_prompt,
        first_message_template=first_message_template,
    )

    # Collect after all agents finish — same as parallel mode.
    agent_ids = [a.agent_id for a in config.agents]
    summary = collect_experiment(
        experiment_dir=experiment_dir,
        experiment_id=config.experiment_id,
        mode="swarm",
        agent_ids=agent_ids,
    )

    mode_dir = experiment_dir / "mode_swarm"
    write_experiment_report(summary, mode_dir)

    return summary
