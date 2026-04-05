"""Experiment and agent configuration dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class AgentConfig:
    """Per-agent configuration."""
    agent_id: str
    time_budget_minutes: int = 30
    train_time_budget_seconds: int = 300
    cuda_device: str = "0"
    model: str = "claude-sonnet-4-6"
    temperature: Optional[float] = None
    system_prompt_file: str = "templates/agent_system_prompt.md"
    first_message_file: str = "templates/agent_first_message.md"

    @classmethod
    def from_json(cls, path: Path) -> "AgentConfig":
        data = json.loads(path.read_text())
        # Map old-style fields
        if "google_model" in data:
            data.pop("google_model", None)
        if "provider" in data:
            data.pop("provider", None)
        if "thinking" in data:
            data.pop("thinking", None)
        if "prompt_file" in data:
            data.pop("prompt_file", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    experiment_id: str
    mode: str  # "parallel" | "single_long" | "swarm" | "parallel_capacity_benchmark" | "merge_search"
    base_time_budget_minutes: int = 30
    train_time_budget_seconds: int = 300
    autoresearch_dir: str = "autoresearch"
    results_root: str = "results"
    agents: list[AgentConfig] = field(default_factory=list)
    repo_root: str = ""
    # SLURM settings (threaded through to create_workspace)
    slurm_partition: str = "pi_tpoggio"
    slurm_gres: str = "gpu:1"
    slurm_time: str = "00:10:00"
    # Template file paths (relative to repo root)
    system_prompt_file: str = "templates/agent_system_prompt.md"
    first_message_file: str = "templates/agent_first_message.md"
    # Swarm-specific settings (only used when mode == "swarm")
    swarm_shared_memory_file: str = "shared_memory.jsonl"
    swarm_sync_interval_seconds: int = 10
    swarm_max_context_entries: int = 20

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def make_parallel(
        cls,
        experiment_id: str,
        time_budget_minutes: int,
        train_time_budget_seconds: int,
        repo_root: str,
    ) -> "ExperimentConfig":
        agents = [
            AgentConfig(
                agent_id="agent_0",
                time_budget_minutes=time_budget_minutes,
                train_time_budget_seconds=train_time_budget_seconds,
                cuda_device="0",
            ),
            AgentConfig(
                agent_id="agent_1",
                time_budget_minutes=time_budget_minutes,
                train_time_budget_seconds=train_time_budget_seconds,
                cuda_device="1",
            ),
        ]
        return cls(
            experiment_id=experiment_id,
            mode="parallel",
            base_time_budget_minutes=time_budget_minutes,
            train_time_budget_seconds=train_time_budget_seconds,
            repo_root=repo_root,
            agents=agents,
        )

    @classmethod
    def make_n_parallel(
        cls,
        experiment_id: str,
        n_agents: int,
        time_budget_minutes: int,
        train_time_budget_seconds: int,
        repo_root: str,
        cuda_devices: Optional[list[str]] = None,
    ) -> "ExperimentConfig":
        """Create config for N independent parallel agents.

        If cuda_devices is not provided, agents are assigned devices 0..N-1.
        For SLURM-based training this device assignment controls CUDA_VISIBLE_DEVICES
        within each agent's environment.
        """
        if cuda_devices is None:
            cuda_devices = [str(i) for i in range(n_agents)]
        if len(cuda_devices) < n_agents:
            raise ValueError(
                f"Provide at least {n_agents} CUDA device strings "
                f"(got {len(cuda_devices)})."
            )
        agents = [
            AgentConfig(
                agent_id=f"agent_{i}",
                time_budget_minutes=time_budget_minutes,
                train_time_budget_seconds=train_time_budget_seconds,
                cuda_device=cuda_devices[i],
            )
            for i in range(n_agents)
        ]
        return cls(
            experiment_id=experiment_id,
            mode="parallel",
            base_time_budget_minutes=time_budget_minutes,
            train_time_budget_seconds=train_time_budget_seconds,
            repo_root=repo_root,
            agents=agents,
        )

    @classmethod
    def make_single_long(
        cls,
        experiment_id: str,
        time_budget_minutes: int,
        train_time_budget_seconds: int,
        repo_root: str,
    ) -> "ExperimentConfig":
        agents = [
            AgentConfig(
                agent_id="agent_0",
                time_budget_minutes=time_budget_minutes * 2,
                train_time_budget_seconds=train_time_budget_seconds,
                cuda_device="0",
            ),
        ]
        return cls(
            experiment_id=experiment_id,
            mode="single_long",
            base_time_budget_minutes=time_budget_minutes,
            train_time_budget_seconds=train_time_budget_seconds,
            repo_root=repo_root,
            agents=agents,
        )

    @classmethod
    def from_yaml(cls, path: Path, repo_root: str = "") -> "ExperimentConfig":
        """Load ExperimentConfig from an experiment.yaml file.

        The YAML schema is documented in configs/experiment.yaml.
        """
        import yaml

        raw = yaml.safe_load(path.read_text())
        exp   = raw.get("experiment", {})
        ag    = raw.get("agents", {})
        slurm = raw.get("slurm", {})
        tmpl  = raw.get("templates", {})
        swarm = raw.get("swarm", {})

        mode     = exp.get("mode", "parallel")
        budget   = int(ag.get("time_budget_minutes", 30))
        train_s  = int(ag.get("train_time_budget_seconds", 300))
        n_agents = int(ag.get("n", 2))
        model    = ag.get("model", "claude-haiku-4-5-20251001")
        temp     = ag.get("temperature", None)
        devices  = ag.get("cuda_devices", None)
        overrides: dict[str, dict] = {
            o["agent_id"]: o for o in ag.get("overrides", []) if "agent_id" in o
        }

        if devices is None:
            devices = [str(i) for i in range(n_agents)]
        devices = [str(d) for d in devices]

        experiment_id = exp.get("id") or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Build per-agent configs, applying any overrides
        if mode in ("single_long",):
            agent_list = [
                AgentConfig(
                    agent_id="agent_0",
                    time_budget_minutes=budget * 2,
                    train_time_budget_seconds=train_s,
                    cuda_device=devices[0],
                    model=model,
                    temperature=temp,
                )
            ]
        else:
            agent_list = []
            for i in range(n_agents):
                aid = f"agent_{i}"
                ov  = overrides.get(aid, {})
                agent_list.append(AgentConfig(
                    agent_id=aid,
                    time_budget_minutes=int(ov.get("time_budget_minutes", budget)),
                    train_time_budget_seconds=int(ov.get("train_time_budget_seconds", train_s)),
                    cuda_device=str(ov.get("cuda_device", devices[i])),
                    model=ov.get("model", model),
                    temperature=ov.get("temperature", temp),
                    system_prompt_file=ov.get(
                        "system_prompt_file",
                        tmpl.get("system_prompt", "templates/agent_system_prompt.md"),
                    ),
                    first_message_file=ov.get(
                        "first_message_file",
                        tmpl.get("first_message", "templates/agent_first_message.md"),
                    ),
                ))

        return cls(
            experiment_id=experiment_id,
            mode=mode,
            base_time_budget_minutes=budget,
            train_time_budget_seconds=train_s,
            repo_root=repo_root,
            agents=agent_list,
            slurm_partition=slurm.get("partition", "pi_tpoggio"),
            slurm_gres=slurm.get("gres", "gpu:1"),
            slurm_time=slurm.get("time", "00:10:00"),
            system_prompt_file=tmpl.get("system_prompt", "templates/agent_system_prompt.md"),
            first_message_file=tmpl.get("first_message", "templates/agent_first_message.md"),
            swarm_shared_memory_file=swarm.get("shared_memory_file", "shared_memory.jsonl"),
            swarm_sync_interval_seconds=int(swarm.get("sync_interval_seconds", 10)),
            swarm_max_context_entries=int(swarm.get("max_context_entries", 20)),
        )
