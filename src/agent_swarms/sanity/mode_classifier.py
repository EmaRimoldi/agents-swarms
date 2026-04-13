"""Pre-registered mode classifier for §9.6.

Invokes `claude --print` in a subprocess (same pattern as the
production swarm runner) with a fixed system prompt that
enumerates the taxonomy. Deterministic temperature is not
available via the CLI; the classifier relies on the tight
single-label output format and the default low-temperature
behavior of `claude --print` for short classifications.

The classifier model is intentionally separate from the subject
model so the same classifier can be reused at every escalation
level (see docs/sanity_escalation_plan.md).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CLASSIFIER_SYSTEM_PROMPT = """You are a deterministic classifier. You are given a single candidate proposal to modify `train.py` for a neural language model training task. Classify the PRIMARY category of the edit into exactly one of the modes listed in the user message. Output only the mode name, lowercase, no punctuation, no explanation.

If the proposal's primary edit does not fit any listed mode, output `other`.
"""


def build_classifier_user_message(proposal_text: str,
                                  taxonomy: Sequence[str]) -> str:
    modes = "\n".join(f"- {m}" for m in taxonomy)
    return (
        f"Taxonomy (valid labels):\n{modes}\n\n"
        f"Proposal:\n'''\n{proposal_text.strip()}\n'''\n\n"
        f"Respond with exactly one label from the taxonomy."
    )


@dataclass(frozen=True)
class ClassificationResult:
    raw: str
    label: str
    return_code: int


def normalize_label(raw: str, taxonomy: Sequence[str]) -> str:
    r = raw.strip().lower().splitlines()[0] if raw.strip() else ""
    r = r.strip("`'\".,; ")
    if r in taxonomy:
        return r
    for m in taxonomy:
        if m in r:
            return m
    return "other" if "other" in taxonomy else taxonomy[-1]


def classify_proposal(
    model: str,
    proposal_text: str,
    taxonomy: Sequence[str],
    cwd: Path | None = None,
    timeout_seconds: int = 60,
) -> ClassificationResult:
    user_msg = build_classifier_user_message(proposal_text, taxonomy)
    cmd = [
        "claude",
        "--print",
        "--output-format", "text",
        "--dangerously-skip-permissions",
        "--model", model,
        "--system-prompt", CLASSIFIER_SYSTEM_PROMPT,
        user_msg,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or Path.cwd()),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        raw = proc.stdout or ""
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        raw = ""
        rc = -1
    except FileNotFoundError:
        raw = ""
        rc = -2
    return ClassificationResult(
        raw=raw,
        label=normalize_label(raw, taxonomy),
        return_code=rc,
    )
