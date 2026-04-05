"""Run a Claude Code sub-agent session.

Replaces run_single_agent.sh + OpenClaw invocation entirely.
Uses `claude --print` (non-interactive) with session continuation to run
a multi-turn agent loop that manages its own time budget.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_swarms.agents.base import AgentRunner


def _ts() -> str:
    """Return current local time as ISO-8601 string with second precision."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(log_fh, msg: str) -> None:
    """Write a timestamped system-event line and flush."""
    log_fh.write(f"[{_ts()}] {msg}\n")
    log_fh.flush()
from agent_swarms.budgeting import BudgetTracker
from agent_swarms.config import AgentConfig


class ClaudeAgentRunner(AgentRunner):
    """Runs a Claude Code sub-agent via the `claude` CLI.

    The agent is invoked in a loop:
    - First turn: full first_message with context
    - Subsequent turns: "Continue. ~N min remaining. Keep experimenting."
    - Budget clock starts after first successful turn

    Failure modes handled:
    - API errors: retry with exponential backoff (no time refund)
    - Rate limits: backoff aggressively (no time refund)
    - No-reply turns: track count, rotate session after MAX_NOREPLY
    - Budget exceeded: break loop
    - Startup timeout: exit(2)
    """

    MAX_NOREPLY = 5
    MIN_TURN_INTERVAL_SEC = 5
    INITIAL_BACKOFF_SEC = 5
    MAX_BACKOFF_SEC = 60
    FIRST_TURN_TIMEOUT_SEC = 900  # 15 min: start_gpu_worker + SLURM queue + compile + first run
    MAX_TURN_TIMEOUT_SEC = 900    # 15 min per subsequent turn (ignored in SINGLE_TURN_MODE)
    SINGLE_TURN_MODE: bool = False  # if True, exit after the first successful turn

    @staticmethod
    def _temperature_directive(temperature: Optional[float]) -> str:
        if temperature is None:
            return ""
        if temperature >= 1.0:
            return (
                "\n\n[SEARCH STYLE: Be creative and exploratory. "
                "Prefer bold, diverse changes over incremental refinement. "
                "Try unconventional hyperparameter combinations that you would not "
                "normally attempt. High variance in search is desirable.]"
            )
        if temperature < 0.5:
            return (
                "\n\n[SEARCH STYLE: Be conservative and methodical. "
                "Make only small, well-motivated incremental changes. "
                "Exploit the best-known region before exploring new directions. "
                "Low variance and high reliability are desirable.]"
            )
        return ""

    def run(
        self,
        run_id: str,
        experiment_id: str,
        system_prompt: str,
        first_message: str,
    ) -> None:
        """Run the agent loop until budget expires. Writes metadata.json at end."""
        config = self.config

        effective_system_prompt = system_prompt + self._temperature_directive(
            config.temperature
        )

        self._active_proc: Optional[subprocess.Popen] = None

        budget = BudgetTracker(
            wall_clock_budget_seconds=config.time_budget_minutes * 60,
            train_time_budget_seconds=config.train_time_budget_seconds,
            startup_deadline_seconds=config.time_budget_minutes * 60 + 300,
        )

        session_id = f"{experiment_id}-{config.agent_id}-{int(time.time())}-{os.getpid()}"
        env = self._build_env(run_id, experiment_id)
        session_log = self.logs_dir / "run_agent.log"

        start_time = datetime.now(timezone.utc).isoformat()
        total_turns = 0
        backoff = self.INITIAL_BACKOFF_SEC
        noreply_count = 0
        first_turn = True

        with open(session_log, "w") as log_fh:
            _log(log_fh, f"[{config.agent_id}] Session starting: {session_id}")
            _log(log_fh, f"[{config.agent_id}] Config: model={config.model}"
                         f" | budget={config.time_budget_minutes}min"
                         f" | train_budget={config.train_time_budget_seconds}s"
                         f" | single_turn={self.SINGLE_TURN_MODE}"
                         f" | cuda={config.cuda_device}")

            _stop_watcher = threading.Event()
            _observed_val_bpbs: list[float] = []

            # GPU allocation watcher — starts budget clock
            threading.Thread(
                target=self._watch_gpu_allocation,
                args=(budget, log_fh, _stop_watcher),
                daemon=True,
            ).start()

            # Workspace event watcher — training trigger/result/file changes
            threading.Thread(
                target=self._watch_workspace_events,
                args=(log_fh, _stop_watcher, _observed_val_bpbs),
                daemon=True,
            ).start()

            while True:
                if budget.startup_expired():
                    msg = f"[{config.agent_id}] ABORT: no successful turn within startup deadline."
                    _log(log_fh, msg)
                    sys.stderr.write(msg + "\n")
                    break

                if budget.should_stop():
                    remaining = int(budget.remaining_seconds())
                    _log(log_fh, f"[{config.agent_id}] Budget expired — stopping. ({remaining}s remaining)")
                    break

                if first_turn:
                    turn_msg = first_message
                    turn_timeout = max(self.FIRST_TURN_TIMEOUT_SEC, int(budget.remaining_seconds()))
                    _log(log_fh, f"[{config.agent_id}] Turn {total_turns} starting (first turn, timeout={turn_timeout}s).")
                else:
                    secs_left = int(budget.remaining_seconds())
                    mins_left = secs_left // 60
                    turn_msg = self._build_continuation_message(budget)
                    turn_timeout = min(secs_left, self.MAX_TURN_TIMEOUT_SEC)
                    _log(log_fh, f"[{config.agent_id}] Turn {total_turns} starting (~{mins_left} min remaining, timeout={turn_timeout}s).")

                turn_start = time.monotonic()

                exit_code, output = self._run_turn(
                    turn_msg=turn_msg,
                    session_id=session_id,
                    system_prompt=effective_system_prompt,
                    timeout_seconds=turn_timeout,
                    env=env,
                    log_fh=log_fh,
                )
                turn_elapsed = time.monotonic() - turn_start

                is_timeout = exit_code == -1
                is_noreply = "No reply from agent" in output or (not output.strip() and exit_code == 0)
                is_ratelimit = "rate limit" in output.lower() or "rate_limit" in output.lower()
                is_error = exit_code != 0 and not is_timeout

                if is_timeout:
                    _log(log_fh,
                        f"[{config.agent_id}] Turn {total_turns} timed out after {turn_elapsed:.1f}s "
                        f"(limit={turn_timeout}s) — fatal, stopping.")
                    break
                elif is_error:
                    _log(log_fh,
                        f"[{config.agent_id}] Turn {total_turns} failed: exit={exit_code} elapsed={turn_elapsed:.1f}s "
                        f"— retrying in {backoff}s...")
                    if output:
                        log_fh.write(output[:2000] + ("\n...(truncated)\n" if len(output) > 2000 else "\n"))
                        log_fh.flush()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF_SEC)
                elif is_ratelimit:
                    _log(log_fh,
                        f"[{config.agent_id}] Turn {total_turns} hit rate limit (elapsed={turn_elapsed:.1f}s) "
                        f"— backing off {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF_SEC)
                elif is_noreply:
                    noreply_count += 1
                    _log(log_fh,
                        f"[{config.agent_id}] Turn {total_turns} got no reply "
                        f"(#{noreply_count}/{self.MAX_NOREPLY}, elapsed={turn_elapsed:.1f}s)")
                    if noreply_count >= self.MAX_NOREPLY:
                        noreply_count = 0
                        session_id = f"{experiment_id}-{config.agent_id}-{int(time.time())}-{os.getpid()}"
                        first_turn = True
                        _log(log_fh, f"[{config.agent_id}] Session rotated to {session_id}")
                    _enforce_min_interval(turn_elapsed, self.MIN_TURN_INTERVAL_SEC)
                else:
                    _log(log_fh,
                        f"[{config.agent_id}] Turn {total_turns} completed successfully (elapsed={turn_elapsed:.1f}s).")
                    backoff = self.INITIAL_BACKOFF_SEC
                    noreply_count = 0
                    total_turns += 1

                    if not budget.budget_started():
                        budget.start_budget_clock()
                        _log(log_fh,
                            f"[{config.agent_id}] Budget clock started (fallback, no gpu_allocated_at) — "
                            f"{budget.wall_clock_budget_seconds}s remaining.")
                    first_turn = False
                    if self.SINGLE_TURN_MODE:
                        _log(log_fh, f"[{config.agent_id}] Single-turn mode: session complete.")
                        break
                    _enforce_min_interval(turn_elapsed, self.MIN_TURN_INTERVAL_SEC)

        _stop_watcher.set()

        end_time = datetime.now(timezone.utc).isoformat()
        self._write_metadata(
            run_id=run_id,
            experiment_id=experiment_id,
            start_time=start_time,
            end_time=end_time,
            total_turns=total_turns,
            budget_seconds=config.time_budget_minutes * 60,
            observed_val_bpbs=_observed_val_bpbs,
        )

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _watch_gpu_allocation(
        self,
        budget: BudgetTracker,
        log_fh,
        stop_event: threading.Event,
    ) -> None:
        """Start budget clock when gpu_allocated_at appears."""
        marker = self.workspace / "gpu_allocated_at"
        while not stop_event.is_set():
            if not budget.budget_started() and marker.exists():
                budget.start_budget_clock()
                ts = marker.read_text().strip()
                _log(log_fh,
                    f"[{self.config.agent_id}] GPU allocated at {ts} — "
                    f"budget clock started ({budget.wall_clock_budget_seconds}s).")
                return
            stop_event.wait(2)

    def _watch_workspace_events(
        self,
        log_fh,
        stop_event: threading.Event,
        observed_val_bpbs: list,
    ) -> None:
        """Log key workspace file events: trigger, result, train.py edits, results.tsv rows."""
        ws = self.workspace
        agent_id = self.config.agent_id

        trigger = ws / "run.trigger"
        result = ws / "run.result"
        train_out = ws / "logs" / "train_current.out"
        train_py = ws / "train.py"
        results_tsv = ws / "results" / "results.tsv"
        worker_ready = ws / "worker_ready_at"

        trigger_seen = False
        result_seen = False
        worker_ready_seen = False
        run_count = 0
        run_wall_start: Optional[float] = None
        train_py_mtime: Optional[float] = None
        results_tsv_lines = 0
        train_out_lines = 0

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
            # train.py modified → log diff vs baseline so we see what changed
            try:
                mtime = train_py.stat().st_mtime if train_py.exists() else None
                if mtime is not None and mtime != train_py_mtime:
                    if train_py_mtime is not None:
                        _log(log_fh, f"[{agent_id}] train.py modified.")
                        _log_train_diff(train_py, log_fh, agent_id)
                    train_py_mtime = mtime
            except OSError:
                pass

            # results.tsv new row → agent logged a result
            try:
                if results_tsv.exists():
                    lines = [l for l in results_tsv.read_text().splitlines() if l.strip()]
                    if len(lines) > results_tsv_lines:
                        for row in lines[results_tsv_lines:]:
                            if not row.startswith("commit"):  # skip header
                                _log(log_fh, f"[{agent_id}] results.tsv: {row}")
                        results_tsv_lines = len(lines)
            except OSError:
                pass

            # worker_ready_at appeared → worker loop is polling, agent can now write run.trigger
            if not worker_ready_seen and worker_ready.exists():
                worker_ready_seen = True
                _log(log_fh, f"[{agent_id}] Worker ready — polling for run.trigger.")

            # run.trigger appeared → training started
            if not trigger_seen and trigger.exists():
                trigger_seen = True
                result_seen = False
                run_count += 1
                run_wall_start = time.time()
                train_out_lines = 0  # reset cursor for new run
                _log(log_fh, f"[{agent_id}] Training run #{run_count} started.")

            # stream phase lines from train_current.out while a run is active
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
                elapsed = f"{time.time() - run_wall_start:.0f}s" if run_wall_start else "?s"
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
                    _log(log_fh, f"[{agent_id}] Training run #{run_count} done — val_bpb: {val_bpb} (elapsed: {elapsed})")
                else:
                    status = ""
                    try:
                        status = result.read_text().strip().splitlines()[0] if result.exists() else "no result"
                    except OSError:
                        pass
                    _log(log_fh, f"[{agent_id}] Training run #{run_count} done — {status} (elapsed: {elapsed})")
                    _dump_slurm_failure_logs(ws, agent_id, run_count, log_fh)

            stop_event.wait(2)

    # ------------------------------------------------------------------
    # Continuation message (overridable by subclasses, e.g. SwarmAgentRunner)
    # ------------------------------------------------------------------

    def _build_continuation_message(self, budget: "BudgetTracker") -> str:
        """Build the continuation message for non-first turns.

        Subclasses can override this to inject additional context (e.g. swarm
        blackboard updates) while keeping the base timing guidance intact.
        """
        config = self.config
        mins_left = budget.remaining_minutes()
        secs_left = int(budget.remaining_seconds())
        run_wall_sec = config.train_time_budget_seconds + 90
        run_wall_min = round(run_wall_sec / 60)
        if secs_left < run_wall_sec + 60:
            time_guidance = (
                f"WARNING: only ~{mins_left} min left — NOT ENOUGH for another "
                f"training run (~{run_wall_min} min each). "
                f"Do NOT start a new run. Instead review results.tsv, "
                f"ensure the best result is committed, and stop."
            )
        else:
            runs_remaining = secs_left // run_wall_sec
            time_guidance = (
                f"Each training run takes ~{run_wall_min} min. "
                f"You can fit approximately {runs_remaining} more run(s)."
            )
        return (
            f"Continue the research. ~{mins_left} min remaining in budget. "
            f"{time_guidance} "
            f"Keep modifying train.py and running experiments to improve val_bpb."
        )

    # ------------------------------------------------------------------
    # Core turn execution
    # ------------------------------------------------------------------

    def _run_turn(
        self,
        turn_msg: str,
        session_id: str,
        system_prompt: str,
        timeout_seconds: int,
        env: dict,
        log_fh,
    ) -> tuple[int, str]:
        """Invoke `claude --print` for one turn, streaming output to log in real-time."""
        cmd = [
            "claude",
            "--print",
            "--output-format", "text",
            "--dangerously-skip-permissions",
        ]
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]
        cmd += [turn_msg]

        output_lines: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.workspace),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._active_proc = proc

            # Stream stdout in real-time
            def _stream_stdout():
                for line in proc.stdout:
                    output_lines.append(line)
                    log_fh.write(f"  {line}" if not line.startswith("[") else line)
                    log_fh.flush()

            stdout_thread = threading.Thread(target=_stream_stdout, daemon=True)
            stdout_thread.start()

            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return -1, f"[timeout after {timeout_seconds}s]"

            stdout_thread.join(timeout=5)
            stderr = proc.stderr.read()
            self._active_proc = None
            output = "".join(output_lines)
            if stderr:
                output += "\n[stderr]\n" + stderr
            return proc.returncode, output

        except FileNotFoundError:
            return -2, "[claude CLI not found in PATH]"
        except Exception as e:
            return -3, f"[exception: {e}]"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_env(self, run_id: str, experiment_id: str) -> dict:
        env = os.environ.copy()
        env["RUN_ID"] = run_id
        env["AGENT_ID"] = self.config.agent_id
        env["RESULTS_ROOT"] = str(self.results_dir)
        env["AUTOSEARCH_TIME_BUDGET"] = str(self.config.train_time_budget_seconds)
        env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_device
        env["EXPERIMENT_ID"] = experiment_id
        extra_path = ":".join([
            str(Path.home() / ".local" / "bin"),
            str(Path.home() / "miniforge3" / "bin"),
        ])
        env["PATH"] = extra_path + ":" + env.get("PATH", "")
        return env

    def _write_metadata(
        self,
        run_id: str,
        experiment_id: str,
        start_time: str,
        end_time: str,
        total_turns: int,
        budget_seconds: int,
        observed_val_bpbs: list | None = None,
    ) -> None:
        metadata = {
            "agent_id": self.config.agent_id,
            "run_id": run_id,
            "experiment_id": experiment_id,
            "start_time": start_time,
            "end_time": end_time,
            "total_turns": total_turns,
            "budget_seconds": budget_seconds,
            "model": self.config.model,
        }
        meta_path = self.results_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        # Build trajectory.jsonl — primary source: observed val_bpb values captured
        # by the workspace watcher. Fallback: workspace/results/results.tsv.
        traj_path = self.results_dir / "trajectory.jsonl"
        traj_bpbs: list[float] = list(observed_val_bpbs) if observed_val_bpbs else []
        if not traj_bpbs:
            results_tsv = self.workspace / "results" / "results.tsv"
            if results_tsv.exists():
                for row in results_tsv.read_text().splitlines():
                    if not row.strip() or row.startswith("commit"):
                        continue
                    parts = row.split("\t")
                    if len(parts) >= 2:
                        try:
                            traj_bpbs.append(float(parts[1]))
                        except ValueError:
                            pass
        if traj_bpbs:
            traj_lines = [json.dumps({"step": i, "val_bpb": v}) for i, v in enumerate(traj_bpbs)]
            traj_path.write_text("\n".join(traj_lines) + "\n")

        if traj_path.exists():
            lines = [l for l in traj_path.read_text().splitlines() if l.strip()]
            metadata["total_training_runs"] = len(lines)
            if lines:
                bpbs = [json.loads(l).get("val_bpb") for l in lines if l.strip()]
                bpbs = [b for b in bpbs if b is not None]
                if bpbs:
                    metadata["best_val_bpb"] = min(bpbs)
            meta_path.write_text(json.dumps(metadata, indent=2))


def _enforce_min_interval(elapsed: float, min_interval: float) -> None:
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)


def _log_train_diff(train_py: Path, log_fh, agent_id: str) -> None:
    """Log only the changed lines (+/-) of train.py vs train.py.baseline."""
    import difflib
    baseline = train_py.parent / "train.py.baseline"
    if not baseline.exists():
        return
    try:
        old = baseline.read_text().splitlines()
        new = train_py.read_text().splitlines()
        diff = list(difflib.unified_diff(old, new, lineterm=""))
        changed = [l for l in diff if (l.startswith("+") or l.startswith("-")) and not l.startswith(("---", "+++"))]
        if not changed:
            return
        for line in changed:
            _log(log_fh, f"[{agent_id}]   diff | {line}")
    except OSError:
        pass


def _dump_slurm_failure_logs(
    workspace: Path,
    agent_id: str,
    run_count: int,
    log_fh,
    tail_lines: int = 50,
) -> None:
    """Append SLURM training logs to the agent log on training failure.

    Dumps (up to tail_lines each):
    - workspace/logs/train_current.out  — stdout of the failing train.py run
    - workspace/logs/worker_*.err       — stderr of the SLURM worker job
    """
    logs_dir = workspace / "logs"

    train_out = logs_dir / "train_current.out"
    if train_out.exists():
        try:
            lines = train_out.read_text().splitlines()
            tail = lines[-tail_lines:]
            log_fh.write(f"[{agent_id}] --- train_current.out (last {len(tail)} lines) ---\n")
            for line in tail:
                log_fh.write(f"[{agent_id}]   {line}\n")
            log_fh.write(f"[{agent_id}] --- end train_current.out ---\n")
        except OSError:
            pass

    try:
        err_files = sorted(logs_dir.glob("worker_*.err"))
        if err_files:
            latest_err = err_files[-1]
            lines = latest_err.read_text().splitlines()
            if lines:
                tail = lines[-tail_lines:]
                log_fh.write(f"[{agent_id}] --- {latest_err.name} (last {len(tail)} lines) ---\n")
                for line in tail:
                    log_fh.write(f"[{agent_id}]   {line}\n")
                log_fh.write(f"[{agent_id}] --- end {latest_err.name} ---\n")
    except OSError:
        pass

    log_fh.flush()
