"""Swarm-specific configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class SwarmConfig:
    """Configuration for swarm communication behaviour.

    These fields are read from the ``swarm:`` section of experiment.yaml and
    forwarded to each agent via SwarmOrchestrator.

    Attributes
    ----------
    shared_memory_file:
        Filename (relative to the experiment's mode directory) for the shared
        JSONL blackboard.
    sync_interval_seconds:
        How often (in seconds) the workspace-event watcher thread sleeps
        between blackboard reads.  Smaller = more responsive; larger = less
        I/O overhead.  Not currently enforced at the agent level (the watcher
        already polls every 2 s for workspace events, which is sufficient).
    max_context_entries:
        Maximum number of other-agent blackboard entries injected into each
        continuation message.  Keeps the context window bounded.
    """

    shared_memory_file: str = "shared_memory.jsonl"
    sync_interval_seconds: int = 10
    max_context_entries: int = 20

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SwarmConfig":
        return cls(
            shared_memory_file=d.get("shared_memory_file", "shared_memory.jsonl"),
            sync_interval_seconds=int(d.get("sync_interval_seconds", 10)),
            max_context_entries=int(d.get("max_context_entries", 20)),
        )
