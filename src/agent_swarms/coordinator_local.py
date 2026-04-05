"""coordinator_local.py — local shim implementing the autoresearch-at-home Coordinator API.

Provides the same `Coordinator` class interface as the Ensue-based
coordinator.py from https://github.com/mutable-state-inc/autoresearch-at-home,
but backed by a local append-only JSONL blackboard (SharedMemory) instead of
the Ensue network.

Agents use it exactly as documented in collab.md:
    from coordinator import Coordinator
    coord = Coordinator()
    coord.agent_id = "nova"
    coord.announce()

When SWARM_MEMORY_PATH is not set, all coordination methods are no-ops so
the agent can still run in solo mode without crashing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Bootstrap: ensure the package is importable from a workspace directory
# ---------------------------------------------------------------------------

_THIS = Path(__file__).resolve()
# coordinator_local.py may be copied to a workspace directory; we need to find
# the actual package to import SharedMemory.
_CANDIDATES = [
    _THIS.parent.parent,                          # src/ when running from package
    Path(os.environ.get("AGENT_SWARMS_SRC", "")),  # explicit override
]
for _cand in _CANDIDATES:
    if (_cand / "agent_swarms" / "shared_memory.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

try:
    from agent_swarms.shared_memory import (  # noqa: E402
        SharedMemory,
        ENTRY_RESULT,
        ENTRY_HYPOTHESIS,
        ENTRY_INSIGHT,
        ENTRY_STATUS,
        DEFAULT_CLAIM_TTL,
    )
    _HAS_SHARED_MEMORY = True
except ImportError:
    _HAS_SHARED_MEMORY = False

# ---------------------------------------------------------------------------
# VRAM tier detection (mirrors autoresearch-at-home logic)
# ---------------------------------------------------------------------------

_VRAM_TIERS = [("small", 16), ("medium", 24), ("large", 48)]


def _detect_vram() -> tuple[float, str]:
    """Return (vram_gb, tier). Falls back to (0.0, 'unknown') on error."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip().splitlines()[0]
        vram_mb = float(out.strip())
        vram_gb = vram_mb / 1024.0
        tier = "xl"
        for name, threshold in _VRAM_TIERS:
            if vram_gb <= threshold:
                tier = name
                break
        return vram_gb, tier
    except Exception:
        return 0.0, "unknown"


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    """Local shim implementing the autoresearch-at-home Coordinator interface.

    All network-dependent features (Ensue hub, leaderboard, ask_swarm NLP
    search) gracefully degrade to local equivalents or no-ops.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        # agent identity — set by caller: coord.agent_id = "nova"
        self._agent_id: str = os.environ.get("AGENT_ID", "agent")

        # shared memory
        self._mem: Optional[SharedMemory] = None
        if _HAS_SHARED_MEMORY:
            path_str = os.environ.get("SWARM_MEMORY_PATH", "")
            if not path_str:
                # try .swarm_env in cwd
                swarm_env = Path(".swarm_env")
                if swarm_env.exists():
                    for line in swarm_env.read_text().splitlines():
                        if line.startswith("SWARM_MEMORY_PATH="):
                            path_str = line.split("=", 1)[1].strip()
                            break
            if path_str:
                self._mem = SharedMemory(Path(path_str))

        # VRAM info
        self.vram_gb, self.vram_tier = _detect_vram()

        # experiment counter for periodic sync
        self._exp_count = 0

        # Ensue connectivity flag (always False in local mode)
        self._connected = False

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @agent_id.setter
    def agent_id(self, value: str) -> None:
        self._agent_id = value

    def connected(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Hub setup (no-ops in local mode)
    # ------------------------------------------------------------------

    def join_hub(self, invite_token: str = "") -> dict:
        self._log("Running in local mode — Ensue hub not connected.")
        return {}

    def announce(self) -> None:
        """Print startup banner with current swarm state."""
        print(f"\n{'='*60}")
        print(f"  autoresearch-at-home  ·  local swarm mode")
        print(f"  agent: {self._agent_id}  ·  VRAM: {self.vram_gb:.1f} GB ({self.vram_tier})")
        if self._mem is None:
            print("  WARNING: no SWARM_MEMORY_PATH — running in solo mode")
        else:
            print(f"  blackboard: {self._mem.path}")
        print(f"{'='*60}\n")
        if self._mem is not None:
            self.analyze_swarm()
        self._clog("announce | startup")

    # ------------------------------------------------------------------
    # THINK — swarm analysis
    # ------------------------------------------------------------------

    def analyze_swarm(self) -> dict:
        """Print and return a structured summary of the swarm state."""
        if self._mem is None:
            print("[swarm] Solo mode — no shared memory.")
            return {}

        entries = self._mem.read_all()

        # Global best
        best = self._mem.get_best()
        if best:
            print(f"[swarm] Global best: val_bpb={best['val_bpb']} "
                  f"by {best['agent_id']} at {best['timestamp']}")
        else:
            print("[swarm] Global best: none yet.")

        # Active claims (what others are testing right now)
        claims = self._mem.get_active_claims()
        own = [c for c in claims if c["agent_id"] == self._agent_id]
        others = [c for c in claims if c["agent_id"] != self._agent_id]
        if own:
            print(f"[swarm] Your active claim: {own[0]['hypothesis']}")
        if others:
            print(f"[swarm] Active claims by others:")
            for c in others:
                print(f"  [{c['agent_id']}] {c['hypothesis']}")
        else:
            print("[swarm] No active claims by other agents.")

        # All results (hypothesis + val_bpb)
        results = [
            e for e in entries
            if e.get("entry_type") == ENTRY_RESULT
            and e.get("content", {}).get("source") != "watcher"
        ]
        if results:
            print(f"\n[swarm] Experiment history ({len(results)} runs):")
            for r in results:
                c = r.get("content", {})
                status = "✓ keep" if c.get("accepted") else "✗ discard"
                hyp = c.get("hypothesis") or c.get("description") or "(no description)"
                val = c.get("val_bpb", "?")
                print(f"  {status}  [{r['agent_id']}]  {hyp}  →  val_bpb={val}")
        else:
            print("[swarm] No experiment results yet.")

        # Recent insights
        insights = [e for e in entries if e.get("entry_type") == ENTRY_INSIGHT][-5:]
        if insights:
            print(f"\n[swarm] Recent insights ({len(insights)} shown):")
            for i in insights:
                print(f"  [{i['agent_id']}] {i.get('content', {}).get('insight', '?')[:120]}")

        # Unclaimed hypotheses
        unclaimed = self.get_unclaimed_hypotheses(limit=5)
        if unclaimed:
            print(f"\n[swarm] Unclaimed hypotheses ({len(unclaimed)} available):")
            for h in unclaimed:
                print(f"  [{h['agent_id']}] {h['title']}: {h['hypothesis'][:100]}")

        self._clog(
            f"analyze_swarm | best={'none' if not best else best['val_bpb']} "
            f"| results={len(results)} | insights={len(insights)}"
        )
        return {
            "best": best,
            "active_claims": claims,
            "results": results,
            "insights": insights,
            "unclaimed_hypotheses": unclaimed,
        }

    def get_all_agent_bests(self) -> list[dict]:
        """Return each agent's best result, sorted by val_bpb."""
        if self._mem is None:
            return []
        entries = self._mem.read_all()
        results = [
            e for e in entries
            if e.get("entry_type") == ENTRY_RESULT
            and e.get("content", {}).get("accepted")
            and e.get("content", {}).get("source") != "watcher"
        ]
        bests: dict[str, dict] = {}
        for r in results:
            aid = r["agent_id"]
            val = r.get("content", {}).get("val_bpb")
            if val is None:
                continue
            if aid not in bests or val < bests[aid]["val_bpb"]:
                bests[aid] = {"agent_id": aid, "val_bpb": val, "timestamp": r["timestamp"]}
        return sorted(bests.values(), key=lambda x: x["val_bpb"])

    def get_all_tier_bests(self) -> dict[str, Optional[dict]]:
        """No tier tracking in local mode — returns empty dict."""
        return {}

    def get_tier_best(self, tier: str) -> Optional[dict]:
        return None

    # ------------------------------------------------------------------
    # CLAIM
    # ------------------------------------------------------------------

    def claim_experiment(self, description: str) -> Optional[str]:
        """Reserve an experiment. Returns experiment key or None if duplicate."""
        if self._mem is None:
            # solo mode — generate a dummy key so the loop continues
            return self._make_key(description)

        ok, claim_id = self._mem.claim(self._agent_id, description, ttl_seconds=DEFAULT_CLAIM_TTL)
        if not ok:
            print(f"[coord] CLAIM REJECTED: '{description[:60]}' is too similar to an active claim or past result.")
            print("[coord] Call analyze_swarm() to see what's active, then choose a different direction.")
            self._clog(f"claim | DUPLICATE | {description[:80]}")
            return None

        key = self._make_key(description)
        print(f"[coord] Claimed: {description[:60]}  (key={key[:20]}…  id={claim_id[:8]}…)")
        self._clog(f"claim | ACCEPTED | key={key[:20]} | {description[:80]}")
        # store claim_id so publish can release it
        self._last_claim_id = claim_id
        return key

    def _make_key(self, description: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", description.lower())[:40].strip("-")
        short_hash = hashlib.sha256(description.encode()).hexdigest()[:6]
        return f"{self._agent_id}--{slug}--{short_hash}"

    # ------------------------------------------------------------------
    # PUBLISH
    # ------------------------------------------------------------------

    def publish_result(
        self,
        exp_key: str,
        val_bpb: float,
        memory_gb: float,
        status: str,
        description: str,
        train_py_src: str,
        extra_metrics: Optional[dict] = None,
    ) -> None:
        """Publish experiment result to the blackboard."""
        accepted = status == "keep"
        claim_id = getattr(self, "_last_claim_id", None)

        if self._mem is not None:
            self._mem.write(self._agent_id, ENTRY_RESULT, {
                "hypothesis": description,
                "description": description,
                "val_bpb": val_bpb,
                "memory_gb": memory_gb,
                "accepted": accepted,
                "status": status,
                "exp_key": exp_key,
                **(extra_metrics or {}),
            })

            if accepted:
                updated = self._mem.update_best(self._agent_id, val_bpb, train_py_src)
                if updated:
                    print(f"[coord] NEW GLOBAL BEST: val_bpb={val_bpb} — other agents will see this on next analyze_swarm().")

            if claim_id:
                self._mem.release_claim(self._agent_id, claim_id)
                self._last_claim_id = None

        print(f"[coord] Published: {description[:60]}  val_bpb={val_bpb}  status={status}")
        self._clog(f"publish | val_bpb={val_bpb} accepted={accepted} key={exp_key[:20]} | {description[:80]}")
        self._exp_count += 1

    def should_sync(self) -> bool:
        """True every 5 experiments — prompt to pull global best."""
        return self._exp_count > 0 and self._exp_count % 5 == 0

    # ------------------------------------------------------------------
    # INSIGHTS
    # ------------------------------------------------------------------

    def post_insight(self, insight: str, evidence_keys: Optional[list[str]] = None) -> None:
        """Share a learning with the swarm."""
        if self._mem is not None:
            self._mem.write(self._agent_id, ENTRY_INSIGHT, {
                "insight": insight,
                "evidence_keys": evidence_keys or [],
            })
        print(f"[coord] Insight posted: {insight[:100]}")
        self._clog(f"insight | {insight[:120]}")

    def get_swarm_insights(self, topic: str = "") -> list[dict]:
        """Return insights, optionally filtered by topic keyword."""
        if self._mem is None:
            return []
        entries = self._mem.read_all()
        insights = [e for e in entries if e.get("entry_type") == ENTRY_INSIGHT]
        if topic:
            topic_lower = topic.lower()
            insights = [
                e for e in insights
                if topic_lower in e.get("content", {}).get("insight", "").lower()
            ]
        return [
            {
                "agent_id": e["agent_id"],
                "insight": e.get("content", {}).get("insight", ""),
                "evidence_keys": e.get("content", {}).get("evidence_keys", []),
                "timestamp": e["timestamp"],
            }
            for e in insights
        ]

    # ------------------------------------------------------------------
    # HYPOTHESES
    # ------------------------------------------------------------------

    def publish_hypothesis(
        self,
        title: str,
        hypothesis: str,
        suggested_config: Optional[dict] = None,
        evidence_keys: Optional[list[str]] = None,
        priority: int = 3,
    ) -> None:
        """Share a proposed experiment with the swarm."""
        if self._mem is not None:
            self._mem.write(self._agent_id, ENTRY_HYPOTHESIS, {
                "title": title,
                "hypothesis": hypothesis,
                "suggested_config": suggested_config or {},
                "evidence_keys": evidence_keys or [],
                "priority": priority,
                "status": "unclaimed",
            })
        print(f"[coord] Hypothesis published: {title} — {hypothesis[:80]}")
        self._clog(f"hypothesis | {title[:40]} | {hypothesis[:80]}")

    def get_unclaimed_hypotheses(self, limit: int = 10) -> list[dict]:
        """Return hypotheses not yet picked up by any agent's claim."""
        if self._mem is None:
            return []
        entries = self._mem.read_all()
        hyps = [e for e in entries if e.get("entry_type") == ENTRY_HYPOTHESIS]
        claims = self._mem.get_active_claims()
        claimed_texts = {c["hypothesis"].lower() for c in claims}
        results = self._mem.read_all()
        tested_texts = {
            e.get("content", {}).get("hypothesis", "").lower()
            for e in results
            if e.get("entry_type") == ENTRY_RESULT
        }
        unclaimed = []
        for e in reversed(hyps):
            c = e.get("content", {})
            hyp_text = c.get("hypothesis", "")
            if hyp_text.lower() not in claimed_texts and hyp_text.lower() not in tested_texts:
                unclaimed.append({
                    "agent_id": e["agent_id"],
                    "title": c.get("title", "(no title)"),
                    "hypothesis": hyp_text,
                    "suggested_config": c.get("suggested_config", {}),
                    "priority": c.get("priority", 3),
                    "timestamp": e["timestamp"],
                })
            if len(unclaimed) >= limit:
                break
        return unclaimed

    # ------------------------------------------------------------------
    # BEST CONFIG
    # ------------------------------------------------------------------

    def pull_best_config(self) -> Optional[tuple[str, dict]]:
        """Write global best train.py to ./best_train.py. Returns (src, metadata) or None."""
        if self._mem is None:
            print("[coord] Solo mode — no global best available.")
            return None
        best = self._mem.get_best()
        if best is None:
            print("[coord] No global best available yet. Continue from current train.py.")
            self._clog("pull_best | no global best yet")
            return None
        out = Path("best_train.py")
        out.write_text(best["train_py_src"], encoding="utf-8")
        metadata = {"val_bpb": best["val_bpb"], "agent_id": best["agent_id"], "timestamp": best["timestamp"]}
        print(f"[coord] Pulled best train.py (val_bpb={best['val_bpb']} from {best['agent_id']}) → best_train.py")
        print("[coord] To adopt: cp best_train.py train.py && git add train.py && git commit -m 'adopt global best'")
        self._clog(f"pull_best | val_bpb={best['val_bpb']} agent={best['agent_id']}")
        return best["train_py_src"], metadata

    def pull_best_config_for_tier(self, tier: Optional[str] = None) -> Optional[tuple[str, dict]]:
        """No tier tracking — delegates to pull_best_config()."""
        return self.pull_best_config()

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    def ask_swarm(self, question: str, namespace: str = "results") -> dict:
        """Simplified keyword search over local blackboard entries."""
        if self._mem is None:
            return {"answer": "Solo mode — no shared memory.", "entries": []}
        type_map = {"results": ENTRY_RESULT, "insights": ENTRY_INSIGHT, "hypotheses": ENTRY_HYPOTHESIS}
        entry_type = type_map.get(namespace)
        entries = self._mem.read_all()
        if entry_type:
            entries = [e for e in entries if e.get("entry_type") == entry_type]
        keywords = question.lower().split()
        matches = []
        for e in entries:
            text = json.dumps(e.get("content", {})).lower()
            if any(kw in text for kw in keywords):
                matches.append(e)
        print(f"[coord] ask_swarm({question!r}, ns={namespace}): {len(matches)} match(es)")
        for m in matches[:10]:
            c = m.get("content", {})
            print(f"  [{m['agent_id']}] {json.dumps(c)[:120]}")
        return {"question": question, "entries": matches}

    def list_namespace(self, namespace: str, limit: int = 50) -> list[dict]:
        """List entries in a given namespace (entry_type)."""
        if self._mem is None:
            return []
        type_map = {"results": ENTRY_RESULT, "insights": ENTRY_INSIGHT,
                    "hypotheses": ENTRY_HYPOTHESIS, "claims": "claim"}
        entry_type = type_map.get(namespace, namespace)
        entries = self._mem.read_all()
        filtered = [e for e in entries if e.get("entry_type") == entry_type][-limit:]
        return filtered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        print(f"[coord] {msg}")

    def _clog(self, entry: str) -> None:
        """Append timestamped line to logs/coordinator.log (streamed by watcher)."""
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(logs_dir / "coordinator.log", "a") as f:
                f.write(f"[{ts}] {entry}\n")
        except OSError:
            pass
