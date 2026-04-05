# agent_swarms

Multi-agent Claude Code swarm for autonomous LLM training research, built on the
[autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home) protocol.

Multiple agents run in parallel, each modifying `train.py` on an isolated GPU worktree,
while sharing results in real time through a JSONL blackboard.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          run-swarm (CLI entry point)                        │
│                       configs/experiment.yaml                               │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SwarmOrchestrator                                  │
│                                                                             │
│  • Creates one git worktree per agent  (autoresearch/ → workspace/)         │
│  • Copies coordinator.py, collab.md, program.md into each workspace         │
│  • Writes .swarm_env with SWARM_MEMORY_PATH                                 │
│  • Spawns N SwarmAgentRunner threads in parallel                            │
│  • Tears down worktrees on exit                                             │
└──────────┬──────────────────────────────────────────────┬────────────────────┘
           │ agent_0                                      │ agent_1 … agent_N
           ▼                                              ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│    SwarmAgentRunner     │                   │    SwarmAgentRunner     │
│                         │                   │                         │
│  ① start_gpu_worker.sh  │                   │  ① start_gpu_worker.sh  │
│    (SLURM sbatch)       │                   │    (SLURM sbatch)       │
│                         │                   │                         │
│  ② ClaudeAgentRunner    │                   │  ClaudeAgentRunner      │
│    claude --print       │                   │    claude --print       │
│    (single turn mode)   │                   │    (single turn mode)   │
│                         │                   │                         │
│  ③ Watcher threads:     │                   │  Watcher threads:       │
│    • gpu_allocated_at   │                   │    • gpu_allocated_at   │
│    • train.py diff      │                   │    • train.py diff      │
│    • phase keywords     │                   │    • phase keywords     │
│    • coordinator.log    │                   │    • coordinator.log    │
│    • auto-publish bpb   │                   │    • auto-publish bpb   │
└──────────┬──────────────┘                   └──────────┬──────────────┘
           │ workspace/                                   │ workspace/
           ▼                                              ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│   Agent Workspace       │                   │   Agent Workspace       │
│  (git worktree)         │                   │  (git worktree)         │
│                         │                   │                         │
│  train.py               │                   │  train.py               │
│  prepare.py             │                   │  prepare.py             │
│  program.md  ◄──────────┼── verbatim from   │  program.md             │
│  collab.md   ◄──────────┼── autoresearch/   │  collab.md              │
│  coordinator.py         │                   │  coordinator.py         │
│  run_on_worker.sh       │                   │  run_on_worker.sh       │
│  start_gpu_worker.sh    │                   │  start_gpu_worker.sh    │
│  stop_gpu_worker.sh     │                   │  stop_gpu_worker.sh     │
│  .venv  ──symlink──►────┼── shared venv     │  .venv  ──symlink──►    │
│  data/  ──symlink──►────┼── shared dataset  │  data/  ──symlink──►    │
└──────────┬──────────────┘                   └──────────┬──────────────┘
           │ reads/writes                                 │ reads/writes
           └──────────────────┬───────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Shared JSONL Blackboard                                 │
│                   runs/<exp>/mode_swarm/shared_memory.jsonl                 │
│                     (fcntl.flock — process-safe appends)                    │
│                                                                             │
│  Entry types:                                                               │
│    announce   — agent startup signal                                        │
│    claim      — reserve an experiment (with TTL to prevent starvation)      │
│    result     — val_bpb + accepted + train.py source                        │
│    insight    — free-text observation from a completed run                  │
│    hypothesis — proposed next experiment                                    │
│    watcher    — auto-published val_bpb from orchestrator safeguard          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Protocol (autoresearch-at-home)

Each agent follows the `program.md` / `collab.md` protocol verbatim from the
[autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home) repo:

```
┌─────────────────────────────────────────────────────────┐
│               Agent Loop (per agent)                    │
│                                                         │
│  START                                                  │
│    │                                                    │
│    ▼                                                    │
│  announce()          ← tell the swarm you're alive      │
│  analyze_swarm()     ← read all current results         │
│    │                                                    │
│    ▼                                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │               LOOP FOREVER                      │   │
│  │                                                 │   │
│  │  1. THINK    form hypothesis from swarm state   │   │
│  │       │                                         │   │
│  │       ▼                                         │   │
│  │  2. CLAIM    claim_experiment(description)      │   │
│  │              → reserve slot (TTL-protected)     │   │
│  │       │                                         │   │
│  │       ▼                                         │   │
│  │  3. RUN      edit train.py → git commit         │   │
│  │              → bash run_on_worker.sh            │   │
│  │              → blocks until val_bpb returned    │   │
│  │       │                                         │   │
│  │       ▼                                         │   │
│  │  4. PUBLISH  publish_result(val_bpb, ...)       │   │
│  │              post_insight(observation)          │   │
│  │              publish_hypothesis(next_idea)      │   │
│  │       │                                         │   │
│  │       ▼                                         │   │
│  │  5. SYNC     analyze_swarm()                    │   │
│  │              → read all new results/insights    │   │
│  │              → pull_best_config() if useful     │   │
│  │       │                                         │   │
│  │       └──────────────────────────────────────── ┘   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## SLURM Worker Model

```
Login node                        Compute node (GPU)
──────────────────                ──────────────────────────────────
ClaudeAgentRunner
  │
  ├─ bash start_gpu_worker.sh  ──►  SLURM job starts
  │   (once at session start)        writes gpu_allocated_at
  │                                  writes worker_ready_at
  │                                  polls for run.trigger
  │
  ├─ bash run_on_worker.sh     ──►  writes run.trigger
  │   (once per train run)           worker detects trigger
  │   blocks waiting for             runs: uv run train.py
  │   run.result                     writes run.result (val_bpb)
  │◄─────────────────────────────── SwarmAgentRunner reads result
  │                                  auto-publishes to blackboard
  │
  └─ bash stop_gpu_worker.sh   ──►  scancel job → GPU released
      (on session end)
```

---

## Logging

Per-agent log at `runs/<exp>/mode_swarm/<agent_id>/logs/run_agent.log`:

```
[2026-04-05 01:22:42] [agent_0] GPU allocated — budget clock started (1200s)
[2026-04-05 01:22:53] [agent_0] coord | [01:22:52] claim | ACCEPTED | key=ember--baseline
[2026-04-05 01:23:06] [agent_0] Training run #1 started.
[agent_0]   Step 100/953 | loss 2.345
[agent_0]   Evaluation phase ...
[2026-04-05 01:30:38] [agent_0] Training run #1 done — val_bpb: 1.127 (elapsed: 452s)
[2026-04-05 01:30:41] [agent_0]   diff | -DEPTH = 8
[2026-04-05 01:30:41] [agent_0]   diff | +DEPTH = 10
[2026-04-05 01:31:53] [agent_0] coord | [01:31:52] publish | val_bpb=1.127 accepted=True
[2026-04-05 01:31:53] [agent_0] coord | [01:31:52] insight | Baseline depth=8 MFU=14%...
```

---

## Quick Start

```bash
# Install
uv sync

# Configure
vim configs/experiment.yaml   # set agents.n, time_budget_minutes, slurm.*

# Run
run-swarm --config configs/experiment.yaml
```

### Key config options (`configs/experiment.yaml`)

```yaml
agents:
  n: 2                          # number of parallel agents
  model: claude-haiku-4-5-20251001
  time_budget_minutes: 25
  train_time_budget_seconds: 300

slurm:
  partition: pi_tpoggio
  gres: gpu:1
  time: "00:30:00"
```

---

## Repository Structure

```
agent_swarms/
├── autoresearch/               ← git submodule (mutable-state-inc/autoresearch-at-home)
│   ├── train.py                   agent edits this
│   ├── prepare.py                 fixed evaluation harness
│   ├── program.md                 agent loop instructions (verbatim, copied to workspace)
│   └── collab.md                  swarm coordination protocol (verbatim, copied to workspace)
├── configs/
│   └── experiment.yaml         ← main entry point
├── templates/
│   ├── agent_system_prompt.md
│   └── agent_first_message.md
└── src/agent_swarms/
    ├── coordinator.py          ← CLI tool (cmd_think, cmd_publish, cmd_pull_best)
    ├── coordinator_local.py    ← Python shim: autoresearch-at-home API over local JSONL
    ├── shared_memory.py        ← append-only JSONL blackboard (fcntl locking)
    ├── swarm_orchestrator.py   ← launches N agents, manages workspaces
    ├── swarm_agent_runner.py   ← per-agent watcher loop + SLURM integration
    ├── agents/
    │   └── claude_agent_runner.py  ← claude --print wrapper, diff/phase logging
    ├── compatibility/
    │   └── training_harness.py ← generates SLURM worker scripts
    └── utils/
        └── workspace.py        ← git worktree creation, symlinks, tool install
```
