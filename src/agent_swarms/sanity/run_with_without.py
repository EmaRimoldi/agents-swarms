"""Run the §9.6 sanity experiment: elicit proposals with and
without W_t in context.

K=300 independent single-proposal `claude --print` subprocess
calls per condition (2*K=600 subject calls total). Each call is
a fresh invocation; no session reuse. This matches the
production agent spawning pattern in
agents/claude_agent_runner.py:431-478 so the experiment runs on
the Claude subscription rather than per-token API billing.

Temperature is NOT pinned: `claude --print` does not expose a
temperature flag, so whatever the CLI default is applies. The
pre-registered nominal temperature=1.0 in the config YAML is
therefore a *documentation* of intent, not an enforced value.
This is a known deviation from production `config.json` (which
had `temperature: null`, i.e. SDK default) and from any
SDK-based reproduction attempt. Documented in the final report.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


PROPOSAL_SYSTEM_PROMPT = """You are an ML research agent working on the AutoResearch nanochat training task. A `train.py` script exists in your workspace. Your job: propose ONE concrete edit to `train.py` that you believe will reduce validation bits-per-byte (val_bpb).

Respond in the following format, and nothing else:

HYPOTHESIS: <one sentence naming the change>
CHANGE: <2-4 sentences describing the edit concretely>
EXPECTED EFFECT: <one sentence>

Propose exactly one edit. Do not list alternatives."""


PROPOSAL_USER_TEMPLATE_NO_W = """You are about to start work on the nanochat training task. You have no prior information from other agents.

Propose ONE edit to train.py."""


PROPOSAL_USER_TEMPLATE_WITH_W = """You are about to start work on the nanochat training task. Before you propose, here is the current state of the swarm shared memory (produced by `python coordinator.py think`):

{w_payload}

Propose ONE edit to train.py."""


@dataclass
class CallRecord:
    call_id: str
    condition: str               # "without_w" | "with_w"
    model: str
    ts_start: float
    ts_end: float
    wallclock_seconds: float
    prompt_sha256: str
    prompt_char_len: int
    response_text: str
    response_char_len: int
    # Char-based rough token proxies (len/4), explicitly estimates.
    est_input_tokens: int
    est_output_tokens: int
    return_code: int
    error: str | None = None


def _sha256_of(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_user_message(condition: str, w_payload: str) -> str:
    if condition == "without_w":
        return PROPOSAL_USER_TEMPLATE_NO_W
    if condition == "with_w":
        return PROPOSAL_USER_TEMPLATE_WITH_W.format(w_payload=w_payload)
    raise ValueError(condition)


def render_side_by_side(w_payload: str) -> str:
    """For the review step: print the exact system+user prompts
    for both conditions, byte-identical to what will be sent.
    """
    a = build_user_message("without_w", w_payload)
    b = build_user_message("with_w", w_payload)
    return (
        "=== SYSTEM PROMPT (identical in both conditions) ===\n"
        f"{PROPOSAL_SYSTEM_PROMPT}\n\n"
        "=== WITHOUT_W user message ===\n"
        f"{a}\n\n"
        "=== WITH_W user message ===\n"
        f"{b}\n"
    )


def _invoke_claude_print(
    model: str,
    system_prompt: str,
    user_message: str,
    cwd: Path,
    timeout_seconds: int = 180,
) -> tuple[int, str, str]:
    """Invoke `claude --print` exactly as agents/claude_agent_runner.py
    does (lines 431-478), with an added --model flag. Returns
    (returncode, stdout, stderr).
    """
    cmd = [
        "claude",
        "--print",
        "--output-format", "text",
        "--dangerously-skip-permissions",
        "--model", model,
        "--system-prompt", system_prompt,
        user_message,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"[timeout after {timeout_seconds}s]"
    except FileNotFoundError:
        return -2, "", "[claude CLI not found in PATH]"


def run_condition(
    model: str,
    condition: str,
    w_payload: str,
    n_calls: int,
    out_jsonl: Path,
    cwd: Path | None = None,
    timeout_seconds: int = 180,
) -> list[CallRecord]:
    user_msg = build_user_message(condition, w_payload)
    full_prompt = PROPOSAL_SYSTEM_PROMPT + "\n" + user_msg
    prompt_sha = _sha256_of(full_prompt)
    prompt_char_len = len(full_prompt)
    est_in = prompt_char_len // 4
    cwd = cwd or Path.cwd()
    records: list[CallRecord] = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a", encoding="utf-8") as f:
        for k in range(n_calls):
            t0 = time.time()
            rc, stdout, stderr = _invoke_claude_print(
                model=model,
                system_prompt=PROPOSAL_SYSTEM_PROMPT,
                user_message=user_msg,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            )
            t1 = time.time()
            text = stdout or ""
            err = None if rc == 0 else (stderr.strip() or f"rc={rc}")
            rec = CallRecord(
                call_id=f"{condition}_{k:04d}",
                condition=condition,
                model=model,
                ts_start=t0,
                ts_end=t1,
                wallclock_seconds=t1 - t0,
                prompt_sha256=prompt_sha,
                prompt_char_len=prompt_char_len,
                response_text=text,
                response_char_len=len(text),
                est_input_tokens=est_in,
                est_output_tokens=len(text) // 4,
                return_code=rc,
                error=err,
            )
            records.append(rec)
            f.write(json.dumps(asdict(rec)) + "\n")
            f.flush()
    return records


def load_records(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
