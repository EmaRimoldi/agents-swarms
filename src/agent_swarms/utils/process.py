"""Subprocess helpers: poll, kill, run with timeout."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional


def run_with_timeout(
    cmd: list[str],
    timeout_seconds: int,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    stdout_path: Optional[Path] = None,
) -> tuple[int, str]:
    """Run a command, capture output, kill after timeout_seconds.

    Returns (exit_code, combined_output_str).
    """
    stdout_fh = None
    try:
        if stdout_path:
            stdout_fh = open(stdout_path, "w")
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=stdout_fh,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            output = ""
        else:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,
            )

        try:
            out, _ = proc.communicate(timeout=timeout_seconds)
            output = out or ""
            return proc.returncode, output
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(2)
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return -1, output if stdout_path else (proc.stdout.read() if proc.stdout else "")
    finally:
        if stdout_fh:
            stdout_fh.close()


def kill_by_pattern(pattern: str) -> None:
    """Kill all processes matching a pgrep pattern."""
    try:
        subprocess.run(
            ["pkill", "-f", pattern],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        pass


def is_running(pattern: str) -> bool:
    """Return True if any process matches the pgrep pattern."""
    result = subprocess.run(
        ["pgrep", "-f", pattern],
        capture_output=True,
    )
    return result.returncode == 0


def send_sigterm(pid: int) -> None:
    """Send SIGTERM to a process, ignoring errors if it's already gone."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
