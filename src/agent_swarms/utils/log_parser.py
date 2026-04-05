"""Parse training output from run.log."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def parse_val_bpb(log_path: Path) -> Optional[float]:
    """Extract val_bpb from a completed training log. Returns None if not found."""
    try:
        text = log_path.read_text()
    except (FileNotFoundError, OSError):
        return None
    m = re.search(r"^val_bpb:\s*([\d.]+)", text, re.MULTILINE)
    if m:
        return float(m.group(1))
    return None


def parse_training_seconds(log_path: Path) -> Optional[float]:
    """Extract training_seconds from run.log."""
    try:
        text = log_path.read_text()
    except (FileNotFoundError, OSError):
        return None
    m = re.search(r"^training_seconds:\s*([\d.]+)", text, re.MULTILINE)
    if m:
        return float(m.group(1))
    return None


def parse_peak_vram_mb(log_path: Path) -> Optional[float]:
    """Extract peak_vram_mb from run.log."""
    try:
        text = log_path.read_text()
    except (FileNotFoundError, OSError):
        return None
    m = re.search(r"^peak_vram_mb:\s*([\d.]+)", text, re.MULTILINE)
    if m:
        return float(m.group(1))
    return None


def training_completed(log_path: Path) -> bool:
    """Return True if training completed successfully (val_bpb present, no FAIL)."""
    return parse_val_bpb(log_path) is not None


def training_crashed(log_path: Path) -> bool:
    """Return True if training crashed (FAIL present or process exited without val_bpb)."""
    try:
        text = log_path.read_text()
    except (FileNotFoundError, OSError):
        return False
    has_fail = "FAIL" in text
    has_result = bool(re.search(r"^val_bpb:", text, re.MULTILINE))
    # Crashed = has FAIL marker OR we have some content but no result
    return has_fail or (len(text.strip()) > 0 and not has_result)


def parse_all_metrics(log_path: Path) -> dict:
    """Return all available metrics from run.log as a dict."""
    return {
        "val_bpb": parse_val_bpb(log_path),
        "training_seconds": parse_training_seconds(log_path),
        "peak_vram_mb": parse_peak_vram_mb(log_path),
        "completed": training_completed(log_path),
        "crashed": training_crashed(log_path),
    }
