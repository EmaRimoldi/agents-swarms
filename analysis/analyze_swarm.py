"""
analyze_swarm.py  —  Swarm experiment analysis
===============================================

Generates 5 publication-quality figures for a two-agent swarm run.
Reuses the loading / metric infrastructure from agent_parallelisation_new
and extends it with swarm-specific visualisations:

  Fig 1  — Joint trajectory + global best  (primary scientific story)
  Fig 2  — Shared-memory event timeline    (novel swarm element)
  Fig 3  — Cross-agent influence           (did memory actually matter?)
  Fig 4  — Parameter exploration heatmap  (what each agent tried)
  Fig 5  — Cumulative improvement attribution  (who moved the frontier)

Usage:
    python analysis/analyze_swarm.py [--exp EXP_ID] [--runs-dir runs/]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ─── constants ────────────────────────────────────────────────────────────────
DPI = 150
AGENT_COLORS = {"atlas": "#1f77b4", "ember": "#ff7f0e"}   # blue / orange
FALLBACK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

ACCEPT_COLOR  = "#2ca02c"   # green
REJECT_COLOR  = "#d62728"   # red
CRASH_COLOR   = "#7f7f7f"   # grey
GLOBAL_COLOR  = "#9467bd"   # purple  — system-level best

EVENT_COLORS = {
    "claim":     "#aec7e8",
    "result":    "#1f77b4",
    "best":      "#ff7f0e",
    "insight":   "#2ca02c",
    "hypothesis":"#9467bd",
}
EVENT_MARKERS = {
    "claim": "^", "result": "o", "best": "*",
    "insight": "s", "hypothesis": "D",
}

PARAM_RE = re.compile(
    r"^(?P<name>[A-Z][A-Z0-9_]+)\s*=\s*(?P<value>.+?)(?:\s*#.*)?$",
    re.MULTILINE,
)

# ─── data loading ─────────────────────────────────────────────────────────────

def parse_log_runs(log_path: Path) -> list[dict]:
    """Parse run_agent.log → list of run dicts with timing + val_bpb."""
    runs: list[dict] = []
    gpu_alloc_ts: Optional[datetime] = None
    pending: dict = {}

    ts_re = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
    gpu_re = re.compile(r"GPU allocated at (.+?) —")
    start_re = re.compile(r"Training run #(\d+) started\.")
    done_re  = re.compile(r"Training run #(\d+) done — (?:val_bpb: ([\d.]+)|(.+)) \(elapsed: (\d+)s\)")

    for line in log_path.read_text().splitlines():
        ts_m = ts_re.match(line)
        if not ts_m:
            continue
        ts = datetime.strptime(ts_m.group(1), "%Y-%m-%d %H:%M:%S")  # naive local

        if m := gpu_re.search(line):
            try:
                # Parse as ISO but strip timezone so it stays in the same
                # naive local time as the log-line prefix timestamps.
                raw = m.group(1).replace("−", "-").strip()
                parsed = datetime.fromisoformat(raw)
                # Drop tzinfo — log prefixes are naive local time
                gpu_alloc_ts = parsed.replace(tzinfo=None)
            except Exception:
                gpu_alloc_ts = ts.replace(tzinfo=None)

        elif m := start_re.search(line):
            n = int(m.group(1))
            pending[n] = ts

        elif m := done_re.search(line):
            n        = int(m.group(1))
            val_str  = m.group(2)
            fail_str = m.group(3)
            elapsed  = int(m.group(4))
            start_ts = pending.pop(n, None)
            end_ts   = ts
            val_bpb  = float(val_str) if val_str else None
            runs.append({
                "run":      n,
                "start_ts": start_ts,
                "end_ts":   end_ts,
                "elapsed":  elapsed,
                "val_bpb":  val_bpb,
                "crashed":  val_bpb is None,
                "gpu_alloc_ts": gpu_alloc_ts,
            })

    # compute elapsed_min from GPU allocation
    t0 = gpu_alloc_ts
    for r in runs:
        if t0 and r["end_ts"]:
            r["end_min"]   = (r["end_ts"]   - t0).total_seconds() / 60
            r["start_min"] = r["end_min"] - r["elapsed"] / 60
        else:
            r["end_min"]   = None
            r["start_min"] = None

    return runs


def parse_log_coord_events(log_path: Path) -> list[dict]:
    """Parse coord | lines from run_agent.log for in-log timing."""
    events = []
    ts_re    = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
    coord_re = re.compile(r"coord \| \[(.+?)\] (\w+)")
    gpu_re   = re.compile(r"GPU allocated at (.+?) —")
    gpu_ts   = None

    for line in log_path.read_text().splitlines():
        ts_m = ts_re.match(line)
        if not ts_m:
            continue
        ts = datetime.strptime(ts_m.group(1), "%Y-%m-%d %H:%M:%S")  # naive local
        if gpu_re.search(line):
            gpu_ts = ts
        if m := coord_re.search(line):
            events.append({"ts": ts, "gpu_ts": gpu_ts, "kind": m.group(2), "raw": line})

    # resolve elapsed
    for e in events:
        if e["gpu_ts"]:
            e["elapsed_min"] = (e["ts"] - e["gpu_ts"]).total_seconds() / 60
        else:
            e["elapsed_min"] = None

    return events


def load_blackboard(jsonl_path: Path) -> pd.DataFrame:
    """Load shared_memory.jsonl into a DataFrame. Drops watcher duplicates."""
    entries = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        entries.append(e)

    df = pd.DataFrame(entries)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)

    # drop watcher-only entries (agent_id = 'agent_0' / 'agent_1')
    df = df[~df["agent_id"].isin(["agent_0", "agent_1"])].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def extract_params(text: str) -> dict:
    return {m.group("name"): m.group("value").strip() for m in PARAM_RE.finditer(text)}


def diff_params(before: str, after: str) -> dict[str, tuple]:
    p0, p1 = extract_params(before), extract_params(after)
    diffs = {}
    all_keys = set(p0) | set(p1)
    for k in all_keys:
        v0, v1 = p0.get(k), p1.get(k)
        if v0 != v1:
            diffs[k] = (v0, v1)
    return diffs


def load_snapshots(agent_dir: Path) -> list[dict]:
    snap_dir = agent_dir / "snapshots"
    snaps = []
    if not snap_dir.exists():
        return snaps
    for step_dir in sorted(snap_dir.iterdir()):
        meta_f = step_dir / "metadata.json"
        train_f = step_dir / "train.py"
        if meta_f.exists():
            meta = json.loads(meta_f.read_text())
            meta["train_py"] = train_f.read_text() if train_f.exists() else ""
            snaps.append(meta)
    return snaps


def load_experiment(exp_dir: Path, agent_ids: list[str] | None = None) -> dict:
    """Load everything from one experiment directory."""
    mode_dir = exp_dir / "mode_swarm"
    config   = json.loads((exp_dir / "config.json").read_text())

    # discover agents
    agent_dirs = [d for d in mode_dir.iterdir()
                  if d.is_dir() and d.name.startswith("agent_")]
    if agent_ids:
        agent_dirs = [d for d in agent_dirs if d.name in agent_ids]
    agent_dirs = sorted(agent_dirs)

    agents = {}
    for ad in agent_dirs:
        log   = ad / "logs" / "run_agent.log"
        traj  = ad / "results" / "trajectory.jsonl"
        meta  = ad / "results" / "metadata.json"
        runs  = parse_log_runs(log) if log.exists() else []
        snaps = load_snapshots(ad)
        traj_df = pd.DataFrame(
            [json.loads(l) for l in traj.read_text().splitlines() if l.strip()]
        ) if traj.exists() else pd.DataFrame(columns=["step", "val_bpb"])

        agents[ad.name] = {
            "runs":    runs,
            "snaps":   snaps,
            "traj":    traj_df,
            "meta":    json.loads(meta.read_text()) if meta.exists() else {},
            "log_path": log,
        }

    bb = load_blackboard(mode_dir / "shared_memory.jsonl")

    # resolve t0 = earliest GPU allocation across agents
    t0 = None
    for a in agents.values():
        for r in a["runs"]:
            if r.get("gpu_alloc_ts"):
                if t0 is None or r["gpu_alloc_ts"] < t0:
                    t0 = r["gpu_alloc_ts"]

    # add elapsed_min to blackboard events
    # bb["ts"] is UTC-aware; t0 is naive local (EDT = UTC-4), so offset by 4h
    if t0:
        t0_utc = pd.Timestamp(t0, tz="America/New_York").tz_convert("UTC")
        bb["elapsed_min"] = (bb["ts"] - t0_utc).dt.total_seconds() / 60

    return {"config": config, "agents": agents, "bb": bb, "t0": t0}


# ─── helpers for agent color ──────────────────────────────────────────────────

def agent_color(codename: str, idx: int = 0) -> str:
    return AGENT_COLORS.get(codename, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def codename_from_agent_dir(agent_dir_name: str, bb: pd.DataFrame) -> str:
    """Best-effort: find codename used by this agent in the blackboard."""
    # map agent_0 → atlas, agent_1 → ember from metadata / log
    # fall back to agent_dir_name
    return agent_dir_name


def resolve_codenames(agents: dict, bb: pd.DataFrame) -> dict[str, str]:
    """Return {agent_dir_name: codename} by matching runs to bb result timestamps."""
    names = {}
    bb_res = bb[bb["entry_type"] == "result"].copy()
    for aname, adata in agents.items():
        cands = set()
        for run in adata["runs"]:
            if run["val_bpb"] is None:
                continue
            close = bb_res[abs(bb_res["content"].apply(
                lambda c: c.get("val_bpb", 0)) - run["val_bpb"]) < 1e-5]
            cands.update(close["agent_id"].tolist())
        # pick the most frequent non-agent_ codename
        cands -= {"agent_0", "agent_1"}
        names[aname] = list(cands)[0] if cands else aname
    return names


# ─── figure 1: joint trajectory + global best ─────────────────────────────────

def fig1_joint_trajectory(exp: dict, out_dir: Path) -> None:
    """
    PURPOSE : Show each agent's val_bpb over wall-clock time, overlaid, with
              the system-level global best as a separate panel.

    WHY     : Primary scientific story — how fast does each agent improve, and
              how does the system minimum evolve relative to individual agents?

    CONCLUSION: If the two lines diverge it signals specialisation; if they
                converge it signals lock-in via shared memory.
    """
    agents = exp["agents"]
    bb     = exp["bb"]
    t0     = exp["t0"]
    cnames = resolve_codenames(agents, bb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # collect all (elapsed_min, val_bpb, accepted) per codename
    all_points: list[dict] = []
    for aname, adata in agents.items():
        cn  = cnames.get(aname, aname)
        col = agent_color(cn)
        traj_min: list[float] = []
        traj_bpb: list[float] = []

        for run in adata["runs"]:
            if run["val_bpb"] is None or run.get("end_min") is None:
                continue
            traj_min.append(run["end_min"])
            traj_bpb.append(run["val_bpb"])
            all_points.append({
                "elapsed_min": run["end_min"],
                "val_bpb":     run["val_bpb"],
                "agent":       cn,
            })

        if not traj_min:
            continue

        # — left panel: individual trajectories —
        ax = axes[0]
        ax.plot(traj_min, traj_bpb, "-o", color=col, lw=1.6, ms=5,
                label=cn, zorder=3)
        # mark best
        best_idx = int(np.argmin(traj_bpb))
        ax.scatter([traj_min[best_idx]], [traj_bpb[best_idx]],
                   marker="*", s=180, color=col, zorder=5,
                   edgecolors="white", linewidths=0.5)

    # best update markers on left panel
    bb_best = bb[bb["entry_type"] == "best"]
    for _, row in bb_best.iterrows():
        em = row.get("elapsed_min", None)
        if em is not None and 0 <= em <= 130:
            axes[0].axvline(em, color=EVENT_COLORS["best"], lw=0.8,
                            ls="--", alpha=0.5, zorder=1)

    axes[0].set_xlabel("Wall-clock time (min)", fontsize=11)
    axes[0].set_ylabel("val_bpb  (lower is better)", fontsize=11)
    axes[0].set_title("Agent Trajectories", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    # orange dashed = best-update event
    axes[0].plot([], [], "--", color=EVENT_COLORS["best"], lw=0.8,
                 label="global best update", alpha=0.7)
    axes[0].legend(fontsize=9)

    # — right panel: global best (system minimum) as step function —
    if all_points:
        pts = sorted(all_points, key=lambda x: x["elapsed_min"])
        times  = [p["elapsed_min"] for p in pts]
        bpbs   = [p["val_bpb"] for p in pts]
        agents_seq = [p["agent"] for p in pts]

        global_best = []
        cur_best    = np.inf
        cur_who     = ""
        for t, b, ag in zip(times, bpbs, agents_seq):
            if b < cur_best:
                cur_best = b
                cur_who  = ag
            global_best.append((t, cur_best, cur_who))

        gb_times = [g[0] for g in global_best]
        gb_bpbs  = [g[1] for g in global_best]
        gb_who   = [g[2] for g in global_best]

        ax2 = axes[1]
        ax2.step(gb_times, gb_bpbs, where="post", color=GLOBAL_COLOR,
                 lw=2.5, label="global best", zorder=3)

        # colour each improvement event by contributing agent
        seen = set()
        for i, (t, b, who) in enumerate(global_best):
            if i == 0 or b < global_best[i-1][1]:
                col = agent_color(who)
                label = who if who not in seen else None
                seen.add(who)
                ax2.scatter([t], [b], color=col, s=90, zorder=5,
                            edgecolors="white", lw=0.5, label=label)
                ax2.annotate(f"{b:.4f}", xy=(t, b),
                             xytext=(3, 4), textcoords="offset points",
                             fontsize=7, color=col)

        ax2.set_xlabel("Wall-clock time (min)", fontsize=11)
        ax2.set_ylabel("Global best val_bpb", fontsize=11)
        ax2.set_title("System-Level Best (min of both agents)", fontsize=13,
                      fontweight="bold")
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9, title="Contributor")

    fig.suptitle(
        f"Swarm Experiment — Two-Agent Dynamics\n"
        f"Experiment: {exp['config'].get('experiment_id', '')}  |  "
        f"Budget: {exp['config'].get('base_time_budget_minutes', '?')} min/agent",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    path = out_dir / "fig1_joint_trajectory.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Fig 1 saved → {path}")


# ─── figure 2: shared-memory event timeline ────────────────────────────────────

def fig2_memory_timeline(exp: dict, out_dir: Path) -> None:
    """
    PURPOSE : Show every blackboard event (claim / result / best / insight /
              hypothesis) on a single horizontal timeline, colour-coded by
              agent.  Size of result markers encodes val_bpb improvement.

    WHY     : The novel element of this setting.  Makes visible the rhythm of
              writes vs reads and whether the agents interleave or cluster.

    CONCLUSION: Temporal clustering of agent events → they are racing for the
                same slot; interleaving → healthy specialisation.
    """
    bb = exp["bb"].copy()
    bb = bb[bb["elapsed_min"].notna()].copy()
    bb = bb[bb["elapsed_min"].between(0, 135)].copy()

    order = ["claim", "result", "best", "insight", "hypothesis"]
    y_map = {et: i for i, et in enumerate(order)}

    fig, ax = plt.subplots(figsize=(14, 4.5))

    plotted_agents: set[str] = set()
    plotted_types:  set[str] = set()

    for _, row in bb.iterrows():
        et = row["entry_type"]
        if et not in y_map:
            continue
        y   = y_map[et]
        em  = row["elapsed_min"]
        ag  = row["agent_id"]
        col = agent_color(ag, list(exp["agents"].keys()).index(
            next((k for k in exp["agents"] if k != ag), "agent_0")))
        col = AGENT_COLORS.get(ag, FALLBACK_COLORS[0])
        mk  = EVENT_MARKERS.get(et, "o")

        # size: for results, scale by improvement vs baseline
        size = 60
        if et == "result":
            c   = row.get("content", {})
            bpb = c.get("val_bpb", 1.13) if isinstance(c, dict) else 1.13
            if bpb > 0:
                size = max(30, min(300, (1.13 - bpb) * 3000 + 50))

        ax.scatter([em], [y], marker=mk, s=size, color=col,
                   edgecolors="white", lw=0.5, zorder=3,
                   alpha=0.85)

        # annotate result val_bpb
        if et == "result":
            c   = row.get("content", {})
            bpb = c.get("val_bpb") if isinstance(c, dict) else None
            if bpb and bpb > 0:
                ax.annotate(f"{bpb:.4f}", xy=(em, y),
                            xytext=(0, 7), textcoords="offset points",
                            fontsize=6.5, ha="center", color=col)

        plotted_agents.add(ag)
        plotted_types.add(et)

    # ─ legends ─
    agent_patches = [
        mpatches.Patch(color=AGENT_COLORS.get(ag, FALLBACK_COLORS[i]),
                       label=ag)
        for i, ag in enumerate(sorted(plotted_agents))
    ]
    type_handles = [
        mlines.Line2D([], [], marker=EVENT_MARKERS.get(et, "o"),
                      color="grey", ms=7, ls="none", label=et)
        for et in order if et in plotted_types
    ]
    leg1 = ax.legend(handles=agent_patches, loc="upper right",
                     title="Agent", fontsize=9, framealpha=0.8)
    ax.add_artist(leg1)
    ax.legend(handles=type_handles, loc="lower right",
              title="Event type", fontsize=9, framealpha=0.8)

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()), fontsize=10)
    ax.set_xlabel("Wall-clock time (min from GPU allocation)", fontsize=11)
    ax.set_title("Shared Memory Event Timeline", fontsize=13,
                 fontweight="bold")
    ax.set_xlim(-2, 132)
    ax.grid(True, axis="x", alpha=0.25)

    # shade "plateau" after last improvement
    bb_results = bb[bb["entry_type"] == "result"]
    good = bb_results[bb_results["content"].apply(
        lambda c: isinstance(c, dict) and c.get("val_bpb", 0) > 1.04
        and c.get("accepted", True))]
    # last improvement event
    last_imp = bb[bb["entry_type"] == "best"]["elapsed_min"].max()
    if pd.notna(last_imp):
        ax.axvspan(last_imp, 132, alpha=0.07, color="grey",
                   label="plateau region")
        ax.axvline(last_imp, color="grey", lw=1.0, ls=":", alpha=0.6)
        ax.annotate("plateau", xy=(last_imp + 1, 0.1),
                    fontsize=8, color="grey", va="bottom")

    fig.tight_layout()
    path = out_dir / "fig2_memory_timeline.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Fig 2 saved → {path}")


# ─── figure 3: cross-agent influence ──────────────────────────────────────────

def fig3_cross_agent_influence(exp: dict, out_dir: Path) -> None:
    """
    PURPOSE : Overlaps both agents' val_bpb trajectories and marks each
              "global best update" event with a vertical dashed line.  Shows
              whether the agent that did NOT produce the improvement then
              improves in its next run.

    WHY     : The key question — does shared memory actually change behaviour?
              If agent B tends to improve AFTER reading agent A's best result,
              that is evidence of productive coupling.

    CONCLUSION: Look at whether the non-publishing agent's next run (after
                each best-update marker) is accepted or rejected.
    """
    agents = exp["agents"]
    bb     = exp["bb"]
    cnames = resolve_codenames(agents, bb)

    # build per-agent run series aligned to wall-clock time
    series: dict[str, list[dict]] = {}
    for aname, adata in agents.items():
        cn = cnames.get(aname, aname)
        series[cn] = [
            r for r in adata["runs"]
            if r["val_bpb"] is not None and r.get("end_min") is not None
        ]

    # best-update events from blackboard
    bb_best = bb[bb["entry_type"] == "best"].copy()
    bb_best = bb_best[bb_best["elapsed_min"].between(0, 135)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"hspace": 0.08})

    cn_list = sorted(series.keys())
    for idx, cn in enumerate(cn_list):
        ax  = axes[idx]
        col = agent_color(cn, idx)
        runs = series[cn]
        if not runs:
            continue
        times = [r["end_min"] for r in runs]
        bpbs  = [r["val_bpb"] for r in runs]
        best_so_far = np.minimum.accumulate(bpbs)

        # trajectory line
        ax.plot(times, bpbs, "o-", color=col, lw=1.5, ms=5, alpha=0.8,
                label=f"{cn} — all runs")
        ax.step(times, best_so_far, where="post", color=col,
                lw=2.5, ls="--", alpha=0.6, label=f"{cn} — running best")

        # best-update markers (vertical lines)
        for _, row in bb_best.iterrows():
            em      = row["elapsed_min"]
            who_pub = row["agent_id"]          # who published this best
            is_self = (who_pub == cn)

            lc = agent_color(who_pub, 0 if who_pub == cn_list[0] else 1)
            ax.axvline(em, color=lc, lw=1.0,
                       ls="-" if is_self else "--",
                       alpha=0.6, zorder=1)

            # find agent's NEXT run after this event
            next_runs = [r for r in runs if r["end_min"] > em]
            if next_runs:
                nr  = next_runs[0]
                # was this run better than the pre-event best?
                prev_bpbs = [r["val_bpb"] for r in runs if r["end_min"] <= em]
                pre_best  = min(prev_bpbs) if prev_bpbs else 9.9
                improved  = nr["val_bpb"] < pre_best
                mk_col    = ACCEPT_COLOR if improved else REJECT_COLOR
                ax.scatter([nr["end_min"]], [nr["val_bpb"]],
                           marker="^" if improved else "v",
                           s=100, color=mk_col, zorder=5,
                           edgecolors="white", lw=0.6)

        ax.set_ylabel(f"{cn}  val_bpb", fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    # shared legend for event markers
    axes[-1].set_xlabel("Wall-clock time (min)", fontsize=11)

    # shared title and legend patches
    patch_self   = mlines.Line2D([], [], color="grey",  lw=1.0, ls="-",
                                 label="own best-update event")
    patch_other  = mlines.Line2D([], [], color="grey",  lw=1.0, ls="--",
                                 label="other agent's best-update event")
    patch_pos    = mlines.Line2D([], [], marker="^", color=ACCEPT_COLOR,
                                 ls="none", ms=8,
                                 label="next run: improved (▲)")
    patch_neg    = mlines.Line2D([], [], marker="v", color=REJECT_COLOR,
                                 ls="none", ms=8,
                                 label="next run: no improvement (▼)")
    fig.legend(handles=[patch_self, patch_other, patch_pos, patch_neg],
               loc="upper center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.01), framealpha=0.9)

    fig.suptitle("Cross-Agent Influence via Shared Memory",
                 fontsize=13, fontweight="bold", y=1.04)
    path = out_dir / "fig3_cross_agent_influence.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Fig 3 saved → {path}")


# ─── figure 4: parameter exploration heatmap ──────────────────────────────────

def fig4_param_heatmap(exp: dict, out_dir: Path) -> None:
    """
    PURPOSE : Show which hyperparameters each agent explored and the average
              change in val_bpb associated with each parameter.

    WHY     : Reveals specialisation vs redundancy: if both agents touch the
              same parameters, the shared memory is not driving diversity.
              If they explore different regions, coordination is efficient.

    CONCLUSION: Blue cells = parameters that reliably improve val_bpb;
                red = harmful; ★ = explored by only one agent (exclusive
                territory).
    """
    agents = exp["agents"]
    bb     = exp["bb"]
    cnames = resolve_codenames(agents, bb)

    # build modification table from blackboard results
    # for each result entry, get hypothesis → parameter inferred from hypothesis text
    # since we don't have snapshots, parse diff lines from run_agent.log
    rows: list[dict] = []

    param_line_re = re.compile(r"diff \| [+-]([A-Z][A-Z0-9_]+)\s*=")

    for aname, adata in agents.items():
        cn = cnames.get(aname, aname)
        runs = adata["runs"]
        log_text = adata["log_path"].read_text()

        # extract diffs per run from log
        diff_blocks: dict[int, list[str]] = {}
        cur_block_run: Optional[int] = None

        run_start_re = re.compile(r"Training run #(\d+) started\.")
        diff_line_re = re.compile(r"diff \| ([+-][A-Z][A-Z0-9_].*)")

        # We'll gather parameter changes seen BEFORE each run start
        param_buf: list[str] = []
        run_params: dict[int, list[str]] = {}

        for line in log_text.splitlines():
            if m := run_start_re.search(line):
                n = int(m.group(1))
                run_params[n] = list(param_buf)
                param_buf = []
            elif m := diff_line_re.search(line):
                param_buf.append(m.group(1))

        for run in runs:
            n = run["run"]
            if run["val_bpb"] is None:
                continue
            params_changed = set()
            for p_line in run_params.get(n, []):
                if pm := re.search(r"[+-]([A-Z][A-Z0-9_]+)\s*=", p_line):
                    params_changed.add(pm.group(1))

            # delta vs previous accepted best
            prev_accepted = [r["val_bpb"] for r in runs
                             if r["run"] < n and r["val_bpb"] is not None]
            prev_best = min(prev_accepted) if prev_accepted else runs[0]["val_bpb"]
            delta = run["val_bpb"] - (prev_best if prev_best else run["val_bpb"])

            for param in params_changed:
                rows.append({
                    "agent":    cn,
                    "param":    param,
                    "delta":    delta,
                    "accepted": delta < 0,
                })

    if not rows:
        print("  ⚠ No parameter change data found — skipping Fig 4")
        return

    df = pd.DataFrame(rows)
    pivot_mean  = df.groupby(["param", "agent"])["delta"].mean().unstack(fill_value=np.nan)
    pivot_count = df.groupby(["param", "agent"])["delta"].count().unstack(fill_value=0)

    # sort params: most improvement first (lowest mean delta across agents)
    param_order = pivot_mean.mean(axis=1).sort_values().index.tolist()
    pivot_mean  = pivot_mean.loc[param_order]
    pivot_count = pivot_count.loc[param_order]

    n_params  = len(param_order)
    n_agents  = len(pivot_mean.columns)
    fig_h     = max(4, 0.55 * n_params + 2)
    fig_w     = 4 + 2 * n_agents + 2

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                              gridspec_kw={"width_ratios": [2 * n_agents, 2]})
    ax_hm, ax_bar = axes

    vals    = pivot_mean.values
    vmax    = max(abs(np.nanmax(vals)), abs(np.nanmin(vals)), 0.001)
    im      = ax_hm.imshow(vals, cmap="RdBu", vmin=-vmax, vmax=vmax,
                            aspect="auto")

    # cell annotations
    for i in range(n_params):
        for j in range(n_agents):
            v = vals[i, j]
            c = pivot_count.iloc[i, j]
            if np.isnan(v):
                ax_hm.text(j, i, "—", ha="center", va="center",
                            color="grey", fontsize=9)
            else:
                tc = "white" if abs(v) > 0.55 * vmax else "black"
                ax_hm.text(j, i, f"{v:+.4f}\n(n={int(c)})",
                            ha="center", va="center", color=tc, fontsize=7.5)

    # exclusive parameter markers
    for i, param in enumerate(param_order):
        row_counts = pivot_count.loc[param]
        non_zero   = (row_counts > 0).sum()
        if non_zero == 1:
            who = row_counts[row_counts > 0].index[0]
            j   = list(pivot_mean.columns).index(who)
            ax_hm.text(j + 0.45, i - 0.35, "★",
                       ha="center", va="center", fontsize=8, color="gold")

    ax_hm.set_xticks(range(n_agents))
    ax_hm.set_xticklabels(pivot_mean.columns, fontsize=10)
    ax_hm.set_yticks(range(n_params))
    ax_hm.set_yticklabels(param_order, fontsize=8)
    ax_hm.set_title("Mean Δval_bpb per parameter × agent",
                    fontsize=11, fontweight="bold")

    cb = fig.colorbar(im, ax=ax_hm, shrink=0.7)
    cb.set_label("mean Δval_bpb", fontsize=8)

    # right panel: aggregate mean delta
    row_means = pivot_mean.mean(axis=1)
    bar_colors = ["#2166ac" if v < 0 else "#d73027" for v in row_means]
    ax_bar.barh(range(n_params), row_means.values, color=bar_colors, alpha=0.8)
    ax_bar.axvline(0, color="black", lw=0.8)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Mean Δval_bpb\n(all agents)", fontsize=8)
    ax_bar.set_title("Aggregate", fontsize=10)
    ax_bar.grid(True, axis="x", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color="#2166ac", label="improvement (Δ < 0)"),
        mpatches.Patch(color="#d73027", label="worsening (Δ > 0)"),
        mpatches.Patch(color="gold",    label="★ exclusive to one agent"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    path = out_dir / "fig4_param_heatmap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Fig 4 saved → {path}")


# ─── figure 5: cumulative improvement attribution ─────────────────────────────

def fig5_attribution(exp: dict, out_dir: Path) -> None:
    """
    PURPOSE : Show the cumulative improvement in the global best over time,
              broken down by which agent was responsible for each improvement.
              Also shows the accept/reject ratio per agent.

    WHY     : Answers the question "who moved the frontier?".  If one agent
              contributes all improvements, the other is either exploring dead
              ends (wasteful) or discovering constraints (valuable insurance).

    CONCLUSION: Roughly equal contribution → healthy parallel exploration.
                Skewed contribution → one agent has a better strategy or was
                lucky; consider asymmetric budgets in future experiments.
    """
    agents = exp["agents"]
    bb     = exp["bb"]
    cnames = resolve_codenames(agents, bb)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── panel 1: cumulative global-best improvement ──────────────────────────
    ax1 = axes[0]
    all_runs: list[dict] = []
    for aname, adata in agents.items():
        cn = cnames.get(aname, aname)
        for r in adata["runs"]:
            if r["val_bpb"] is not None and r.get("end_min") is not None:
                all_runs.append({**r, "agent": cn})

    all_runs.sort(key=lambda x: x["end_min"])
    cur_best = np.inf
    cum_times, cum_best, cum_agent, cum_delta = [], [], [], []
    for r in all_runs:
        if r["val_bpb"] < cur_best:
            delta      = cur_best - r["val_bpb"]
            cur_best   = r["val_bpb"]
            cum_times.append(r["end_min"])
            cum_best.append(cur_best)
            cum_agent.append(r["agent"])
            cum_delta.append(delta)

    # stacked area by agent contribution
    unique_agents = sorted(set(cum_agent))
    bottom_vals   = np.zeros(len(cum_times))
    contrib: dict[str, list[float]] = {ag: [] for ag in unique_agents}

    for i, (t, delta, ag) in enumerate(zip(cum_times, cum_delta, cum_agent)):
        for a in unique_agents:
            contrib[a].append(delta if a == ag else 0.0)

    x    = cum_times
    prev = np.zeros(len(x))
    for ag in unique_agents:
        y   = np.array(contrib[ag])
        cum = np.cumsum(y)
        col = agent_color(ag, unique_agents.index(ag))
        ax1.bar(range(len(x)), y, bottom=prev, color=col,
                alpha=0.75, label=ag, width=0.6)
        prev += y

    ax1.set_xticks(range(len(x)))
    ax1.set_xticklabels([f"{t:.0f}'" for t in x], fontsize=7, rotation=45)
    ax1.set_xlabel("Improvement event (wall-clock time, min)", fontsize=9)
    ax1.set_ylabel("Δ val_bpb improvement", fontsize=9)
    ax1.set_title("Per-Improvement Attribution", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # ── panel 2: accept / reject counts ──────────────────────────────────────
    ax2 = axes[1]
    accept_counts = {}
    reject_counts = {}
    for aname, adata in agents.items():
        cn = cnames.get(aname, aname)
        prev_bests: list[float] = []
        for r in adata["runs"]:
            if r["val_bpb"] is None:
                continue
            pre = min(prev_bests) if prev_bests else r["val_bpb"] + 1
            if r["val_bpb"] < pre:
                accept_counts[cn] = accept_counts.get(cn, 0) + 1
            else:
                reject_counts[cn] = reject_counts.get(cn, 0) + 1
            prev_bests.append(r["val_bpb"])

    cnames_list = sorted(set(list(accept_counts) + list(reject_counts)))
    x_pos = np.arange(len(cnames_list))
    w     = 0.35
    ax2.bar(x_pos - w/2,
            [accept_counts.get(cn, 0) for cn in cnames_list],
            width=w, color=ACCEPT_COLOR, alpha=0.8, label="accepted")
    ax2.bar(x_pos + w/2,
            [reject_counts.get(cn, 0) for cn in cnames_list],
            width=w, color=REJECT_COLOR, alpha=0.8, label="rejected")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cnames_list, fontsize=10)
    ax2.set_ylabel("Number of runs", fontsize=9)
    ax2.set_title("Accept / Reject per Agent", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    for i, cn in enumerate(cnames_list):
        a = accept_counts.get(cn, 0)
        r = reject_counts.get(cn, 0)
        total = a + r
        rate  = a / total if total > 0 else 0
        ax2.text(i, a + 0.15, f"{rate*100:.0f}%\nacc.",
                 ha="center", fontsize=8, color=ACCEPT_COLOR)

    # ── panel 3: best val_bpb over run count ──────────────────────────────────
    ax3 = axes[2]
    for aname, adata in agents.items():
        cn  = cnames.get(aname, aname)
        col = agent_color(cn, list(agents.keys()).index(aname))
        bpbs = [r["val_bpb"] for r in adata["runs"] if r["val_bpb"] is not None]
        if not bpbs:
            continue
        cummin = np.minimum.accumulate(bpbs)
        ax3.plot(range(1, len(cummin)+1), cummin, "o-", color=col,
                 lw=2, ms=5, label=cn)
        ax3.annotate(f"{cummin[-1]:.4f}",
                     xy=(len(cummin), cummin[-1]),
                     xytext=(3, 2), textcoords="offset points",
                     fontsize=8, color=col)

    ax3.set_xlabel("Training run #", fontsize=9)
    ax3.set_ylabel("Best val_bpb so far", fontsize=9)
    ax3.set_title("Cumulative Best per Agent (vs run count)", fontsize=11,
                  fontweight="bold")
    ax3.invert_yaxis()
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Improvement Attribution & Search Efficiency",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "fig5_attribution.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Fig 5 saved → {path}")


# ─── report ───────────────────────────────────────────────────────────────────

FIGURE_DESCRIPTIONS = {
    "fig1_joint_trajectory.png": (
        "Joint Trajectory + Global Best",
        "Left: val_bpb over wall-clock time for each agent, with global-best-update markers (orange dashed). "
        "Right: system-level running minimum, colour-coded by contributing agent.",
        "Reveals how quickly each agent finds improvements and whether the system-minimum is driven "
        "by one agent or both."
    ),
    "fig2_memory_timeline.png": (
        "Shared Memory Event Timeline",
        "Each blackboard event (claim / result / best / insight / hypothesis) shown on a horizontal "
        "timeline. Colour = agent, marker shape = event type, size (for results) encodes improvement magnitude.",
        "Reveals the rhythm of writes and whether agents interleave (healthy) or cluster (racing). "
        "The shaded plateau region shows when no further improvements occurred."
    ),
    "fig3_cross_agent_influence.png": (
        "Cross-Agent Influence via Shared Memory",
        "Both agents' trajectories in separate panels. Vertical lines mark global-best-update events "
        "(solid = own event, dashed = other agent's). Triangles (▲/▼) mark each agent's *next* run after "
        "each best-update event.",
        "Key question: after one agent improves the global best, does the OTHER agent then improve too? "
        "▲ after dashed line = positive coupling; ▼ = no visible effect."
    ),
    "fig4_param_heatmap.png": (
        "Parameter Exploration Heatmap",
        "Rows = hyperparameters modified, columns = agents. Cell colour = mean Δval_bpb (blue=improvement, "
        "red=worsening), annotation = mean delta + count. ★ marks parameters explored by only one agent.",
        "Shows specialisation vs redundancy. Exclusive (★) parameters suggest the agents explored "
        "complementary regions; shared parameters may reflect memory-mediated convergence."
    ),
    "fig5_attribution.png": (
        "Cumulative Improvement Attribution",
        "Left: each improvement event shown as a bar coloured by responsible agent. "
        "Centre: accept/reject counts per agent with acceptance rate. "
        "Right: cumulative best val_bpb vs run count for each agent.",
        "Answers 'who moved the frontier?'. Equal contribution → efficient parallel exploration; "
        "skewed → one agent had a better strategy or luckier initialisation."
    ),
}


def write_report(out_dir: Path, exp: dict) -> None:
    lines = [
        f"# Swarm Analysis Report",
        f"",
        f"**Experiment:** {exp['config'].get('experiment_id', '')}  ",
        f"**Budget:** {exp['config'].get('base_time_budget_minutes', '?')} min/agent  ",
        f"**Agents:** {len(exp['agents'])}  ",
        f"",
        "## Figures",
        "",
    ]
    for fname, (title, what, conclusion) in FIGURE_DESCRIPTIONS.items():
        path = out_dir / fname
        if path.exists():
            lines += [
                f"### {title}",
                f"![{title}]({fname})",
                f"",
                f"**What it shows:** {what}",
                f"",
                f"**Interpretation:** {conclusion}",
                f"",
            ]

    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"  ✓ Report saved → {out_dir}/report.md")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Swarm experiment analysis")
    parser.add_argument("--exp",      default=None,
                        help="Experiment ID (e.g. exp_20260405_022850)")
    parser.add_argument("--runs-dir", default="runs",
                        help="Root directory for experiment runs")
    parser.add_argument("--out-dir",  default=None,
                        help="Output directory for figures (default: analysis/<exp_id>)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = Path(__file__).parent.parent / runs_dir

    # discover experiment
    if args.exp:
        candidates = [d for d in runs_dir.iterdir()
                      if d.is_dir() and args.exp in d.name]
    else:
        candidates = sorted([d for d in runs_dir.iterdir()
                              if d.is_dir() and (d / "config.json").exists()])

    if not candidates:
        sys.exit(f"No experiments found in {runs_dir}")

    exp_dir = candidates[-1]
    print(f"\n→ Analysing experiment: {exp_dir.name}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).parent / exp_dir.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"→ Output directory: {out_dir}\n")

    exp = load_experiment(exp_dir)

    print("Generating figures...")
    fig1_joint_trajectory(exp, out_dir)
    fig2_memory_timeline(exp, out_dir)
    fig3_cross_agent_influence(exp, out_dir)
    fig4_param_heatmap(exp, out_dir)
    fig5_attribution(exp, out_dir)
    write_report(out_dir, exp)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} figures in {out_dir}")


if __name__ == "__main__":
    main()
