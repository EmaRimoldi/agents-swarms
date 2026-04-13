"""Shared blackboard for swarm agent coordination.

Implements the AutoResearch at Home coordination primitives on top of a
local append-only JSONL file:

  - result / hypothesis / insight / status  — event log (existing)
  - claim / claim_release                   — work reservation with TTL
  - best                                    — current globally-best train.py

All reads and writes are protected by fcntl.flock() to prevent corruption
when multiple agent processes write concurrently from separate OS processes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Entry types ─────────────────────────────────────────────────────────────

# Event-log entries (existing)
ENTRY_RESULT = "result"
ENTRY_HYPOTHESIS = "hypothesis"
ENTRY_INSIGHT = "insight"
ENTRY_STATUS = "status"

# Coordination entries (new)
ENTRY_CLAIM = "claim"            # work reservation
ENTRY_CLAIM_RELEASE = "claim_release"  # explicit release
ENTRY_BEST = "best"              # global-best update

# Default claim TTL (seconds) — matches AutoResearch at Home
DEFAULT_CLAIM_TTL = 900  # 15 minutes

# Jaccard similarity threshold above which a hypothesis is "duplicate"
DUPLICATE_THRESHOLD = 0.50


class SharedMemory:
    """Append-only JSONL blackboard shared across all swarm agents.

    Thread-safe and process-safe via fcntl.flock() exclusive locks on writes
    and shared locks on reads.

    Schema per entry (one JSON object per line):
        {
            "timestamp":  "2026-04-03T14:22:05+00:00",
            "agent_id":   "agent_0",
            "entry_type": "result" | "hypothesis" | "insight" | "status"
                        | "claim" | "claim_release" | "best",
            "content": { ... }
        }

    Content schemas:

    result:
        step, hypothesis, val_bpb, accepted, reason, commit

    claim:
        claim_id (uuid4 hex), hypothesis, expires_at (ISO-8601)

    claim_release:
        claim_id

    best:
        val_bpb, train_py_src (full source text)
    """

    def __init__(self, path: Path, max_context_entries: int = 20) -> None:
        self.path = path
        self.max_context_entries = max_context_entries
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch()

    # ── Low-level I/O ────────────────────────────────────────────────────────

    def write(self, agent_id: str, entry_type: str, content: dict) -> None:
        """Append one entry to the blackboard (exclusive lock)."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "entry_type": entry_type,
            "content": content,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _read_all_raw(self) -> list[dict]:
        """Return every entry in the blackboard (shared lock)."""
        entries: list[dict] = []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                fcntl.flock(fh, fcntl.LOCK_SH)
                try:
                    raw = fh.read()
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        except OSError:
            pass
        return entries

    # ── Event log (for agent continuation messages) ──────────────────────────

    def read_others(self, agent_id: str) -> list[dict]:
        """Recent event-log entries written by agents other than agent_id."""
        event_types = {ENTRY_RESULT, ENTRY_HYPOTHESIS, ENTRY_INSIGHT, ENTRY_STATUS}
        entries = [
            e for e in self._read_all_raw()
            if e.get("agent_id") != agent_id
            and e.get("entry_type") in event_types
        ]
        return entries[-self.max_context_entries:]

    def read_all(self) -> list[dict]:
        """Every entry in the blackboard (for post-run analysis)."""
        return self._read_all_raw()

    # ── Claim / deduplication ─────────────────────────────────────────────────

    def claim(
        self,
        agent_id: str,
        hypothesis: str,
        ttl_seconds: int = DEFAULT_CLAIM_TTL,
    ) -> tuple[bool, str]:
        """Reserve a hypothesis for this agent.

        Returns (True, claim_id) if the claim was accepted.
        Returns (False, "") if a semantically similar active claim already exists.

        Deduplication uses Jaccard similarity on word tokens.
        """
        if self.is_duplicate(hypothesis):
            return False, ""

        claim_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        expires_at = now.timestamp() + ttl_seconds
        expires_iso = datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat()
        self.write(agent_id, ENTRY_CLAIM, {
            "claim_id": claim_id,
            "hypothesis": hypothesis,
            "expires_at": expires_iso,
        })
        return True, claim_id

    def release_claim(self, agent_id: str, claim_id: str) -> None:
        """Explicitly release a previously granted claim."""
        self.write(agent_id, ENTRY_CLAIM_RELEASE, {"claim_id": claim_id})

    def get_active_claims(self) -> list[dict]:
        """Return claims that have not expired and have not been released."""
        now = datetime.now(timezone.utc).timestamp()
        all_entries = self._read_all_raw()

        released: set[str] = set()
        for e in all_entries:
            if e.get("entry_type") == ENTRY_CLAIM_RELEASE:
                cid = e.get("content", {}).get("claim_id")
                if cid:
                    released.add(cid)

        active: list[dict] = []
        for e in all_entries:
            if e.get("entry_type") != ENTRY_CLAIM:
                continue
            c = e.get("content", {})
            cid = c.get("claim_id", "")
            if cid in released:
                continue
            try:
                exp = datetime.fromisoformat(c["expires_at"]).timestamp()
            except (KeyError, ValueError):
                continue
            if exp > now:
                active.append({
                    "claim_id": cid,
                    "agent_id": e.get("agent_id"),
                    "hypothesis": c.get("hypothesis", ""),
                    "expires_at": c.get("expires_at"),
                })
        return active

    def is_duplicate(
        self,
        hypothesis: str,
        threshold: float = DUPLICATE_THRESHOLD,
    ) -> bool:
        """True if an active claim is semantically similar to hypothesis.

        Uses word-level Jaccard similarity (no external dependencies).
        Also checks completed results so agents don't re-run failed experiments.
        """
        tokens_new = _tokenise(hypothesis)
        if not tokens_new:
            return False

        # Check active claims
        for claim in self.get_active_claims():
            tokens_existing = _tokenise(claim.get("hypothesis", ""))
            if _jaccard(tokens_new, tokens_existing) >= threshold:
                return True

        # Check completed results (don't repeat finished experiments)
        for e in self._read_all_raw():
            if e.get("entry_type") != ENTRY_RESULT:
                continue
            h = e.get("content", {}).get("hypothesis", "")
            if _jaccard(tokens_new, _tokenise(h)) >= threshold:
                return True

        return False

    # ── Global best ───────────────────────────────────────────────────────────

    def update_best(
        self,
        agent_id: str,
        val_bpb: float,
        train_py_src: str,
    ) -> bool:
        """Update the global best if val_bpb is lower than the current best.

        The train.py source is written to a sidecar file
        (<blackboard_dir>/best_<sha256[:8]>.py) to keep the JSONL readable.

        Returns True if the best was updated, False if not an improvement.
        """
        current = self.get_best()
        if current is not None and current["val_bpb"] <= val_bpb:
            return False

        src_hash = hashlib.sha256(train_py_src.encode()).hexdigest()[:8]
        sidecar = self.path.parent / f"best_{src_hash}.py"
        sidecar.write_text(train_py_src, encoding="utf-8")

        self.write(agent_id, ENTRY_BEST, {
            "val_bpb": val_bpb,
            "train_py_file": sidecar.name,
            "train_py_sha256": src_hash,
        })
        return True

    def get_best(self) -> Optional[dict]:
        """Return {val_bpb, train_py_src, agent_id} for the current global best.

        Scans the log in reverse to find the most recent ENTRY_BEST entry.
        Reads train_py_src from the sidecar file if present, falls back to
        inline 'train_py_src' for backwards compatibility.
        Returns None if no best has been published yet.
        """
        for e in reversed(self._read_all_raw()):
            if e.get("entry_type") != ENTRY_BEST:
                continue
            c = e.get("content", {})

            # Load source from sidecar file (new format)
            train_py_src = ""
            sidecar_name = c.get("train_py_file")
            if sidecar_name:
                sidecar = self.path.parent / sidecar_name
                try:
                    train_py_src = sidecar.read_text(encoding="utf-8")
                except OSError:
                    pass
            else:
                # Backwards compatibility: source stored inline
                train_py_src = c.get("train_py_src", "")

            return {
                "val_bpb": c.get("val_bpb"),
                "train_py_src": train_py_src,
                "train_py_file": sidecar_name,
                "agent_id": e.get("agent_id"),
                "timestamp": e.get("timestamp"),
            }
        return None

    # ── Human-readable summary ────────────────────────────────────────────────

    def format_summary(self) -> str:
        """Return a compact, human-readable summary of the entire blackboard."""
        entries = self._read_all_raw()
        if not entries:
            return "(blackboard is empty)"

        lines: list[str] = [
            f"Blackboard: {self.path}",
            f"Entries: {len(entries)}",
            "",
        ]

        for e in entries:
            ts = e.get("timestamp", "?")[:19].replace("T", " ")  # trim to "YYYY-MM-DD HH:MM:SS"
            agent = e.get("agent_id", "?")
            etype = e.get("entry_type", "?")
            c = e.get("content", {})

            if etype == ENTRY_RESULT:
                status = "✓" if c.get("accepted") else "✗"
                lines.append(
                    f"  [{ts}] {agent:8s}  RESULT    {status} val_bpb={c.get('val_bpb', '?')}"
                    f"  reason={c.get('reason', '')}"
                )
            elif etype == ENTRY_BEST:
                src_file = c.get("train_py_file", "(inline)")
                lines.append(
                    f"  [{ts}] {agent:8s}  BEST      val_bpb={c.get('val_bpb', '?')}"
                    f"  src={src_file}"
                )
            elif etype == ENTRY_HYPOTHESIS:
                hyp = c.get("hypothesis", "?")[:60]
                lines.append(f"  [{ts}] {agent:8s}  HYPOTHESIS  step={c.get('step', '?')}  {hyp}")
            elif etype == ENTRY_INSIGHT:
                lines.append(f"  [{ts}] {agent:8s}  INSIGHT   {c.get('insight', '?')[:80]}")
            elif etype == ENTRY_STATUS:
                lines.append(f"  [{ts}] {agent:8s}  STATUS    {c.get('message', '?')[:80]}")
            elif etype == ENTRY_CLAIM:
                exp = c.get("expires_at", "?")[:19].replace("T", " ")
                hyp = c.get("hypothesis", "?")[:50]
                lines.append(
                    f"  [{ts}] {agent:8s}  CLAIM     id={c.get('claim_id', '?')[:8]}  expires={exp}  {hyp}"
                )
            elif etype == ENTRY_CLAIM_RELEASE:
                lines.append(
                    f"  [{ts}] {agent:8s}  RELEASE   id={c.get('claim_id', '?')[:8]}"
                )
            else:
                lines.append(f"  [{ts}] {agent:8s}  {etype.upper():<10}  {str(c)[:80]}")

        best = self.get_best()
        if best:
            lines += [
                "",
                f"Global best: val_bpb={best['val_bpb']}  by={best['agent_id']}"
                f"  file={best.get('train_py_file', '(inline)')}",
            ]
        return "\n".join(lines)

    # ── Formatting for continuation messages ─────────────────────────────────

    # ── Opt-in byte-identical payload dump (§9.1 novel instrumentation) ──
    # Set SharedMemory._dump_target to a Path and every call to
    # format_for_context will append the rendered string to that file,
    # byte-identically to what is returned to the caller. Off by default;
    # production code is unaffected.
    _dump_target: "Optional[Path]" = None

    def format_for_context(self, entries: list[dict]) -> str:
        """Format event-log entries as a compact text block for agent messages."""
        if not entries:
            rendered = ""
            self._maybe_dump(rendered)
            return rendered

        lines = ["=== SWARM UPDATE (from other agents) ==="]
        for entry in entries:
            agent_id = entry.get("agent_id", "?")
            entry_type = entry.get("entry_type", "?")
            content = entry.get("content", {})

            if entry_type == ENTRY_RESULT:
                step = content.get("step", "?")
                val_bpb = content.get("val_bpb", "?")
                accepted = content.get("accepted")
                hypothesis = content.get("hypothesis", "?")
                reason = content.get("reason", "")
                status = "ACCEPTED" if accepted is True else "REJECTED" if accepted is False else str(accepted).upper()
                reason_str = f". {reason}" if reason else ""
                lines.append(
                    f"[{agent_id}] Step {step}: {hypothesis}. "
                    f"val_bpb={val_bpb} ({status}){reason_str}"
                )
            elif entry_type == ENTRY_HYPOTHESIS:
                step = content.get("step", "?")
                hypothesis = content.get("hypothesis", "?")
                lines.append(f"[{agent_id}] Planning Step {step}: {hypothesis}")
            elif entry_type == ENTRY_INSIGHT:
                lines.append(f"[{agent_id}] Insight: {content.get('insight', '?')}")
            elif entry_type == ENTRY_STATUS:
                lines.append(f"[{agent_id}] Status: {content.get('message', '?')}")

        lines.append("===")
        best = self.get_best()
        if best:
            lines.append(
                f"Current global best: val_bpb={best['val_bpb']} "
                f"(by {best['agent_id']}) — run `python coordinator.py pull-best` "
                "to update your train.py."
            )
        lines.append(
            "Avoid hypotheses already tried. "
            "Run `python coordinator.py think` for full swarm state."
        )
        rendered = "\n".join(lines)
        self._maybe_dump(rendered)
        return rendered

    def _maybe_dump(self, rendered: str) -> None:
        """Byte-identical append of the rendered payload for §9.1 G^W
        measurement. Opt-in via ``SharedMemory._dump_target``.
        """
        target = type(self)._dump_target
        if target is None:
            return
        ts = datetime.now(timezone.utc).isoformat()
        header = f"\n===== DUMP {ts} bytes={len(rendered.encode('utf-8'))} =====\n"
        with open(target, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(rendered)
            if not rendered.endswith("\n"):
                f.write("\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    """Return a set of lowercase word tokens from text."""
    import re
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
