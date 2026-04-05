"""coordinator.py — swarm coordination CLI for agents.

Implements the AutoResearch at Home THINK → CLAIM → RUN → PUBLISH protocol.
This script is copied into every agent workspace at setup time and called
directly by the Claude Code sub-agent during its research loop.

Environment variables (set by the orchestrator):
    SWARM_MEMORY_PATH   — absolute path to shared_memory.jsonl
    AGENT_ID            — this agent's identifier (e.g. "agent_0")

Usage
-----
    python coordinator.py think
        Print current swarm state: active claims, recent results, global best.

    python coordinator.py claim "lower LEARNING_RATE to 1e-4"
        Claim a hypothesis. Exits 0 on success, 1 if a duplicate is active.
        Prints CLAIM_ID=<id> on success (source this or capture it).

    python coordinator.py release <claim_id>
        Release a previously granted claim (called automatically by publish).

    python coordinator.py publish <val_bpb> <accepted> [claim_id]
        Publish the result of the latest training run.
        accepted: 1 = kept, 0 = reverted.
        Automatically updates the global best if this is an improvement.
        Automatically releases the claim if claim_id is provided.

    python coordinator.py pull-best
        Write the current global-best train.py to ./best_train.py.
        Prints the current best val_bpb.  Exits 1 if no best exists yet.

    python coordinator.py best
        Print current best val_bpb (or "none" if not yet set).

    python coordinator.py push-hyp "hypothesis text"
        Share a hypothesis idea with the swarm (does not reserve it).

    python coordinator.py status "message"
        Broadcast a status update to other agents.

    python coordinator.py log [path/to/shared_memory.jsonl]
        Print a human-readable summary of the blackboard.
        Uses SWARM_MEMORY_PATH if no path argument is given.
        Can be called from outside an agent session by passing the path directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the package is importable when called from a workspace directory
_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_swarms.shared_memory import (  # noqa: E402
    SharedMemory,
    ENTRY_HYPOTHESIS,
    ENTRY_RESULT,
    ENTRY_STATUS,
    DEFAULT_CLAIM_TTL,
)


# ── Environment ───────────────────────────────────────────────────────────────

def _get_memory() -> SharedMemory:
    path_str = os.environ.get("SWARM_MEMORY_PATH", "")
    if not path_str:
        _die("SWARM_MEMORY_PATH not set. Is this workspace managed by SwarmOrchestrator?")
    return SharedMemory(Path(path_str))


def _get_agent_id() -> str:
    agent_id = os.environ.get("AGENT_ID", "")
    if not agent_id:
        _die("AGENT_ID not set.")
    return agent_id


def _die(msg: str) -> None:
    print(f"[coordinator] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ── Memory access log ─────────────────────────────────────────────────────────

def _clog(entry: str) -> None:
    """Append a timestamped line to logs/coordinator.log in the workspace.

    The watcher in SwarmAgentRunner streams this file to run_agent.log,
    giving a structured trace of every memory interaction.
    """
    from datetime import datetime
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logs_dir / "coordinator.log", "a") as f:
        f.write(f"[{ts}] {entry}\n")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_think(_args: list[str]) -> None:
    """Print a concise summary of current swarm state."""
    mem = _get_memory()
    agent_id = _get_agent_id()

    # Global best
    best = mem.get_best()
    if best:
        print(f"[swarm] Global best: val_bpb={best['val_bpb']} "
              f"by {best['agent_id']} at {best['timestamp']}")
    else:
        print("[swarm] Global best: none yet — first agent to publish sets the baseline.")

    # Active claims
    claims = mem.get_active_claims()
    own_claims = [c for c in claims if c["agent_id"] == agent_id]
    other_claims = [c for c in claims if c["agent_id"] != agent_id]

    if own_claims:
        print(f"[swarm] Your active claim: {own_claims[0]['hypothesis']} "
              f"(expires {own_claims[0]['expires_at']})")
    if other_claims:
        print(f"[swarm] {len(other_claims)} other active claim(s):")
        for c in other_claims:
            print(f"  [{c['agent_id']}] {c['hypothesis']}")
    else:
        print("[swarm] No other active claims.")

    # Recent results (last 10)
    all_entries = mem.read_all()
    # Exclude watcher-auto-published entries (source=watcher, no hypothesis context)
    results = [
        e for e in all_entries
        if e.get("entry_type") == ENTRY_RESULT
        and e.get("content", {}).get("source") != "watcher"
    ]
    if results:
        print(f"[swarm] All results ({len(results)} total):")
        for r in results:
            c = r.get("content", {})
            status = "✓" if c.get("accepted") else "✗"
            hyp = c.get("hypothesis") or "(no hypothesis recorded)"
            print(f"  {status} [{r['agent_id']}] {hyp} → val_bpb={c.get('val_bpb','?')}")
    else:
        print("[swarm] No results published yet.")

    # Log memory access
    best_str = f"val_bpb={best['val_bpb']} by={best['agent_id']}" if best else "none"
    other_hyps = [c["hypothesis"][:40] for c in other_claims]
    result_strs = [
        f"{'✓' if e.get('content',{}).get('accepted') else '✗'} "
        f"{e.get('agent_id')}:{e.get('content',{}).get('val_bpb')}"
        for e in results
    ]
    _clog(
        f"think | global_best={best_str} "
        f"| other_claims={other_hyps} "
        f"| all_results=[{', '.join(result_strs)}]"
    )


def cmd_claim(args: list[str]) -> None:
    """Claim a hypothesis. Exits 0 on success, 1 if duplicate."""
    if not args:
        _die("claim requires a hypothesis string. Example: python coordinator.py claim 'lower LR'")

    hypothesis = args[0]
    mem = _get_memory()
    agent_id = _get_agent_id()

    ok, claim_id = mem.claim(agent_id, hypothesis, ttl_seconds=DEFAULT_CLAIM_TTL)
    if not ok:
        print(f"[coordinator] DUPLICATE: a similar hypothesis is already claimed or has been tested.")
        print("[coordinator] Run `python coordinator.py think` to see what's active, then choose a different direction.")
        _clog(f"claim | DUPLICATE | hypothesis={hypothesis[:80]}")
        sys.exit(1)

    print(f"[coordinator] Claimed: {hypothesis}")
    print(f"CLAIM_ID={claim_id}")  # agent can capture this
    _clog(f"claim | ACCEPTED | claim_id={claim_id[:8]} | hypothesis={hypothesis[:80]}")
    sys.exit(0)


def cmd_release(args: list[str]) -> None:
    """Release a claim by claim_id."""
    if not args:
        _die("release requires a claim_id. Example: python coordinator.py release <claim_id>")

    claim_id = args[0]
    mem = _get_memory()
    agent_id = _get_agent_id()
    mem.release_claim(agent_id, claim_id)
    print(f"[coordinator] Released claim {claim_id}")


def cmd_publish(args: list[str]) -> None:
    """Publish a training result and optionally release its claim."""
    if len(args) < 2:
        _die("publish requires: <val_bpb> <accepted 0|1> [claim_id] [hypothesis]")

    try:
        val_bpb = float(args[0])
    except ValueError:
        _die(f"val_bpb must be a float, got: {args[0]!r}")

    accepted = args[1].strip() in ("1", "true", "True", "yes")
    claim_id = args[2] if len(args) > 2 else None
    hypothesis = args[3] if len(args) > 3 else "(not specified)"

    mem = _get_memory()
    agent_id = _get_agent_id()

    # Write result to event log
    mem.write(agent_id, ENTRY_RESULT, {
        "hypothesis": hypothesis,
        "val_bpb": val_bpb,
        "accepted": accepted,
        "reason": "accepted — val_bpb improved" if accepted else "rejected — no improvement",
    })
    print(f"[coordinator] Published: val_bpb={val_bpb} accepted={accepted}")

    # Update global best if this run improved things
    if accepted:
        train_py = Path("train.py")
        if train_py.exists():
            src = train_py.read_text(encoding="utf-8")
            updated = mem.update_best(agent_id, val_bpb, src)
            if updated:
                print(f"[coordinator] NEW GLOBAL BEST: val_bpb={val_bpb} — "
                      "other agents will pick this up on their next THINK step.")
            else:
                current = mem.get_best()
                if current:
                    print(f"[coordinator] Not a new best (current best: {current['val_bpb']})")
        else:
            print("[coordinator] WARNING: train.py not found — best not updated.")

    # Release claim automatically
    if claim_id:
        mem.release_claim(agent_id, claim_id)
        print(f"[coordinator] Released claim {claim_id}")

    _clog(
        f"publish | val_bpb={val_bpb} accepted={accepted} "
        f"hypothesis={hypothesis[:80]} claim_id={claim_id}"
    )


def cmd_pull_best(args: list[str]) -> None:
    """Write the current best train.py to ./best_train.py.

    After calling this, copy it over your working train.py:
        cp best_train.py train.py
        git add train.py
        git commit -m "sync: pull global best train.py from swarm"
    """
    mem = _get_memory()
    best = mem.get_best()
    if best is None:
        _clog("pull-best | no global best yet")
        print("[coordinator] No global best available yet. "
              "Continuing from current train.py.")
        sys.exit(1)

    out = Path("best_train.py")
    out.write_text(best["train_py_src"], encoding="utf-8")
    _clog(f"pull-best | val_bpb={best['val_bpb']} agent={best['agent_id']}")
    print(f"[coordinator] Pulled best train.py (val_bpb={best['val_bpb']}) → best_train.py")
    print(f"[coordinator] To adopt it: cp best_train.py train.py && git add train.py && "
          "git commit -m 'sync: pull global best'")
    sys.exit(0)


def cmd_best(_args: list[str]) -> None:
    """Print current best val_bpb."""
    mem = _get_memory()
    best = mem.get_best()
    if best:
        print(f"val_bpb={best['val_bpb']} agent={best['agent_id']} ts={best['timestamp']}")
    else:
        print("none")


def cmd_push_hyp(args: list[str]) -> None:
    """Share a hypothesis with the swarm (does not reserve it)."""
    if not args:
        _die("push-hyp requires a hypothesis string.")

    hypothesis = args[0]
    mem = _get_memory()
    agent_id = _get_agent_id()
    mem.write(agent_id, ENTRY_HYPOTHESIS, {"hypothesis": hypothesis})
    print(f"[coordinator] Shared hypothesis: {hypothesis}")


def cmd_status(args: list[str]) -> None:
    """Broadcast a status message."""
    if not args:
        _die("status requires a message string.")

    mem = _get_memory()
    agent_id = _get_agent_id()
    mem.write(agent_id, ENTRY_STATUS, {"message": args[0]})
    print(f"[coordinator] Status broadcast: {args[0]}")


def cmd_reason(args: list[str]) -> None:
    """Log memory-conditioned reasoning after a THINK step.

    Called by the agent immediately after reading the blackboard, this captures
    *why* the agent chose its next hypothesis given what the swarm has tried.

    Example:
        python coordinator.py reason "global best is 1.10 (warmup 0.05). \
            agent_1 is testing depth=10.  I will try reducing WARMDOWN_RATIO \
            to 0.3 since no one has tested LR schedule shape yet."
    """
    if not args:
        _die("reason requires a reasoning text string.")

    reasoning = args[0]
    agent_id = _get_agent_id()
    _clog(f"reason | {reasoning}")
    print("[coordinator] Reasoning logged.")


def cmd_log(args: list[str]) -> None:
    """Print a human-readable summary of the blackboard.

    Accepts an optional path argument so it can be called from outside an
    agent session without SWARM_MEMORY_PATH being set:

        python coordinator.py log runs/experiment_exp_.../mode_swarm/shared_memory.jsonl
    """
    if args:
        path = Path(args[0])
        if not path.exists():
            _die(f"File not found: {path}")
        mem = SharedMemory(path)
    else:
        path_str = os.environ.get("SWARM_MEMORY_PATH", "")
        if not path_str:
            _die(
                "No path given and SWARM_MEMORY_PATH is not set.\n"
                "Usage: python coordinator.py log <path/to/shared_memory.jsonl>"
            )
        mem = SharedMemory(Path(path_str))

    print(mem.format_summary())


# ── Dispatch ──────────────────────────────────────────────────────────────────

_COMMANDS = {
    "think": cmd_think,
    "claim": cmd_claim,
    "release": cmd_release,
    "publish": cmd_publish,
    "pull-best": cmd_pull_best,
    "best": cmd_best,
    "push-hyp": cmd_push_hyp,
    "status": cmd_status,
    "reason": cmd_reason,
    "log": cmd_log,
}


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    args = sys.argv[2:]

    handler = _COMMANDS.get(command)
    if handler is None:
        _die(f"Unknown command: {command!r}. Available: {', '.join(_COMMANDS)}")

    handler(args)


if __name__ == "__main__":
    main()
