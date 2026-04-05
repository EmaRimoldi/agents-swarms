# Swarm Research Protocol

Every agent in the swarm shares one objective: drive `val_bpb` as low as possible.
You do this by running focused experiments, publishing every result — success or
failure — and always building on the swarm's best known starting point.

## Core principle

Before each experiment you must check what the rest of the swarm has already tried.
You must **claim** your hypothesis before running, so no other agent duplicates your work.
After each experiment you must publish your result so others can build on it.
Every iteration follows this order:

```
THINK  →  REASON  →  CLAIM  →  (pull-best)  →  RUN  →  PUBLISH
```

---

## Step-by-step

### THINK — sync with the swarm

```bash
python coordinator.py think
```

This prints:
- The current global best `val_bpb` and who holds it
- **All results from all agents** — every hypothesis tried and its outcome
- Active claims — what other agents are currently testing

**Read the full results list carefully.** Your job is to find a direction that
nobody has tried yet. Do not repeat an experiment that already has a result,
even if a different agent ran it.

After reading the swarm state, log your reasoning — explicitly name what others
have tried and why your choice is different:

```bash
python coordinator.py reason "Best is X (agent_Y). Tried so far: [list]. I will try <hypothesis> because <rationale>."
```

### CLAIM — reserve your hypothesis

Before touching `train.py`, claim your hypothesis. This prevents another agent
from running the same experiment in parallel:

```bash
CLAIM_ID=$(python coordinator.py claim "<one-line hypothesis>" | grep CLAIM_ID | cut -d= -f2)
```

If `claim` exits with code 1, the hypothesis is already active or has been tested.
Go back to THINK and pick a different direction.

### SYNC — pull the global best

```bash
python coordinator.py pull-best   # writes best_train.py; exits 1 if none yet
```

If `pull-best` succeeds (exit 0), adopt it as your starting point:

```bash
cp best_train.py train.py
git add train.py
git commit -m "sync: pull global best from swarm"
```

If no best exists yet (you are the first agent), continue from the current
`train.py` as-is.

### RUN — apply change, train, record

Make **one focused change** to `train.py`. Put the hypothesis in the commit message:

```bash
# edit train.py
git commit -am "exp: <brief description of hypothesis>"
```

Save a pre-training snapshot:

```bash
python save_snapshot.py $STEP "<hypothesis>" "<expected_effect>" [prev_val_bpb]
```

Run training. **Do not paste full training log output into the transcript** — it
fills the context window. Use `tail` if you need to inspect output:

```bash
bash run_on_worker.sh          # blocks until done; prints val_bpb or TRAINING FAILED
tail -50 logs/train_current.out   # only if you need to diagnose a crash
```

Once `val_bpb` is known, update your records:

```bash
python update_snapshot.py $STEP <val_bpb> <true|false> "<reason>" "<next_step>"
```

Keep or revert:
- `val_bpb` improved → keep the commit.
- Equal or worse → `git reset --hard HEAD~1`.

### PUBLISH — share your result with the swarm

After the outcome is final, publish **exactly once**, including the hypothesis text
and your claim ID so the claim is released automatically:

```bash
python coordinator.py publish <val_bpb> <1_if_kept|0_if_reverted> "$CLAIM_ID" "<hypothesis>"
```

Examples:
```bash
python coordinator.py publish 1.0823 1 "$CLAIM_ID" "increase MATRIX_LR from 0.04 to 0.06"
python coordinator.py publish 1.1042 0 "$CLAIM_ID" "add WARMUP_RATIO=0.05"
python coordinator.py publish 0.000000 0 "$CLAIM_ID" "DEPTH=12 — OOM crash"
```

`publish` automatically:
1. Writes your result **with hypothesis text** to the shared log (others see it on their next THINK).
2. Updates the global best if your `val_bpb` is lower than the current best.
3. Releases your claim so others know this direction is resolved.

**Do not call `publish` before `val_bpb` is known.**

---

## GPU worker lifecycle

Start the worker **once per session** and persist the job ID immediately:

```bash
WORKER_JOB_ID=$(bash start_gpu_worker.sh)
echo "$WORKER_JOB_ID" > .worker_job_id
```

> **Important:** A bash `trap` does **not** persist across separate Claude Bash
> tool invocations — each tool call starts a new shell. Save the job ID to
> `.worker_job_id` so it is always recoverable regardless of how the session ends.

On every orderly shutdown (end of session or interruption):

```bash
[ -f .worker_job_id ] && bash stop_gpu_worker.sh "$(cat .worker_job_id)"
```

---

## Rules

1. **Always THINK first.** Read the full results history before touching `train.py`.
2. **Always REASON after THINK.** Explicitly name what others tried and why your direction is different.
3. **Always CLAIM before running.** No claim = no run. If claim is rejected, pick something else.
4. **Always start from the global best.** Call `pull-best` at every cycle.
5. **One change per iteration.** Small, focused diffs are easier to attribute.
6. **Publish every result** with hypothesis text, including failures and crashes (`accepted=0`).
7. **One publish per run.** Do not publish before `val_bpb` is known.
8. **Context hygiene.** Never dump full training logs into the transcript.
   Use `tail -50 logs/train_current.out` to inspect; report only metrics and key decisions.
9. **Do not stop.** Loop forever until externally interrupted. After each PUBLISH, return to THINK.

---

## Quick reference

| Command | Description |
|---------|-------------|
| `python coordinator.py think` | Print full swarm state (all results + active claims) |
| `python coordinator.py reason "<text>"` | Log reasoning after THINK |
| `python coordinator.py claim "<hypothesis>"` | Reserve hypothesis; prints `CLAIM_ID=<id>` |
| `python coordinator.py pull-best` | Write global best → `best_train.py` (exit 1 if none) |
| `python coordinator.py publish <bpb> <1\|0> "$CLAIM_ID" "<hypothesis>"` | Publish result + release claim |
| `python coordinator.py best` | Print current best val_bpb |
| `bash start_gpu_worker.sh` | Allocate GPU |
| `bash run_on_worker.sh` | Train (blocks until done) |
| `bash stop_gpu_worker.sh "$(cat .worker_job_id)"` | Release GPU |
