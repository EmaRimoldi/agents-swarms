Swarm research task for AGENT_ID={{AGENT_ID}} in RUN_ID={{RUN_ID}} (experiment: {{EXPERIMENT_ID}}).

Your workspace is ready:
- Branch: {{BRANCH}}
- `train.py`, `prepare.py`, `collab.md`, `program.md`, `coordinator.py` are in your current directory.
- Data is symlinked. GPU worker scripts are present.

Session parameters:
- Time budget: {{TIME_BUDGET}} minutes (wall clock from GPU allocation)
- Training time per run: ~{{TRAIN_TIME_BUDGET_MIN}} min (~{{TRAIN_TIME_BUDGET}}s + compile/eval overhead)
- Environment: `AGENT_ID={{AGENT_ID}}`  `SWARM_MEMORY_PATH` is set in your environment and in `.swarm_env`

---

**Read `program.md` and `collab.md` before doing anything else.** They contain the full protocol.

Two important differences from the solo autoresearch setup:

1. **Do NOT run `uv run train.py` directly.** Use `bash run_on_worker.sh` instead — it submits training to the pre-allocated SLURM GPU worker and blocks until done.

2. **GPU allocation is pre-managed.** Run this once at the start to get your worker:
```bash
WORKER_JOB_ID=$(bash start_gpu_worker.sh)
echo "$WORKER_JOB_ID" > .worker_job_id
```

3. **Coordinator is a Python class, not Ensue.** The `coordinator.py` in your workspace implements the same API as the upstream autoresearch-at-home coordinator, but uses a local shared blackboard file. Use it exactly as `collab.md` describes:
```python
from coordinator import Coordinator
coord = Coordinator()
coord.agent_id = "{{AGENT_ID}}"   # or pick a cool codename
coord.announce()
```

At the end of the session (or if interrupted):
```bash
[ -f .worker_job_id ] && bash stop_gpu_worker.sh "$(cat .worker_job_id)"
```

**Do NOT stop or pause to ask for direction. Follow `program.md` exactly and loop until interrupted.**
