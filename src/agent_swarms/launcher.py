"""Entry point: parse args, run swarm experiment."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

from agent_swarms.config import ExperimentConfig
from agent_swarms.experiment_modes.swarm_two_agents import run_swarm_experiment
from agent_swarms.swarm_config import SwarmConfig


def _repo_root() -> Path:
    return Path(__file__).parents[2]  # src/ → agent_swarms/


def _load_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text()


def _render_first_message(template: str, train_budget_seconds: int) -> str:
    train_min = max(1, train_budget_seconds // 60)
    return template.replace("{{TRAIN_TIME_BUDGET_MIN}}", str(train_min))


def _make_experiment_id(prefix: str = "swarm") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main_swarm(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run swarm experiment (agents communicate via shared blackboard)"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment.yaml. If provided, all other flags are ignored.")
    parser.add_argument("--time-budget", type=int, default=30, help="Budget per agent (minutes)")
    parser.add_argument("--train-budget", type=int, default=300, help="Budget per training run (seconds)")
    parser.add_argument("--n-agents", type=int, default=2, help="Number of swarm agents")
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default="runs")
    args = parser.parse_args(argv)

    repo_root = _repo_root()

    if args.config:
        config = ExperimentConfig.from_yaml(Path(args.config), repo_root=str(repo_root))
        if config.mode != "swarm":
            config = dataclasses.replace(config, mode="swarm")
    else:
        experiment_id = args.experiment_id or _make_experiment_id()
        config = ExperimentConfig.make_n_parallel(
            experiment_id=experiment_id,
            n_agents=args.n_agents,
            time_budget_minutes=args.time_budget,
            train_time_budget_seconds=args.train_budget,
            repo_root=str(repo_root),
        )
        config = dataclasses.replace(config, mode="swarm")

    runs_dir = repo_root / (args.runs_dir if not args.config else "runs")
    experiment_dir = runs_dir / f"experiment_{config.experiment_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = _load_template(repo_root / config.system_prompt_file)
    first_message_tmpl = _render_first_message(
        _load_template(repo_root / config.first_message_file),
        config.train_time_budget_seconds,
    )

    swarm_config = SwarmConfig(
        shared_memory_file=config.swarm_shared_memory_file,
        sync_interval_seconds=config.swarm_sync_interval_seconds,
        max_context_entries=config.swarm_max_context_entries,
    )

    print(f"[launcher] Starting swarm experiment: {config.experiment_id}")
    print(f"[launcher] Agents: {len(config.agents)}  |  Budget: {config.base_time_budget_minutes} min  |  Train: {config.train_time_budget_seconds} s")
    print(f"[launcher] SLURM: partition={config.slurm_partition}  gres={config.slurm_gres}  time={config.slurm_time}")
    print(f"[launcher] Blackboard: {swarm_config.shared_memory_file}  max_context: {swarm_config.max_context_entries}")
    print(f"[launcher] Output directory: {experiment_dir}")

    run_swarm_experiment(
        config=config,
        experiment_dir=experiment_dir,
        repo_root=repo_root,
        system_prompt=system_prompt,
        first_message_template=first_message_tmpl,
        swarm_config=swarm_config,
    )
    print(f"[launcher] Swarm experiment complete. Results: {experiment_dir}")
