#!/usr/bin/env python3
"""CLI driver for the §9.6 sanity experiment (Level 1 only).

Usage:
    python scripts/run_sanity_w_read.py --dry-run
    python scripts/run_sanity_w_read.py --select-w-only
    python scripts/run_sanity_w_read.py --execute           # AFTER approval

No API calls are made unless --execute is passed explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml   # noqa: E402

from agent_swarms.sanity.select_w import select_w                        # noqa: E402
from agent_swarms.sanity.run_with_without import (                        # noqa: E402
    render_side_by_side,
    run_condition,
)
from agent_swarms.sanity.mode_classifier import classify_proposal         # noqa: E402
from agent_swarms.sanity.stats import bootstrap_g_w, render_report        # noqa: E402


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sanity_w_read.yaml")
    p.add_argument("--out-dir", default="runs/sanity/level_1")
    p.add_argument("--dry-run", action="store_true",
                   help="Select W, render side-by-side prompts, exit.")
    p.add_argument("--select-w-only", action="store_true",
                   help="Only select W and exit.")
    p.add_argument("--execute", action="store_true",
                   help="Actually call the Anthropic API.")
    p.add_argument("--taxonomy", default="taxonomy_coarse",
                   choices=["taxonomy_coarse", "taxonomy_fine"])
    args = p.parse_args()

    cfg_path = (REPO_ROOT / args.config).resolve()
    cfg = load_config(cfg_path)
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: select W_t (always).
    source_run = (REPO_ROOT / cfg["source_swarm_run"]["path"]).resolve()
    sel = select_w(source_run=source_run, out_dir=out_dir, repo_root=REPO_ROOT)
    print(f"[sanity] W_t selected: {sel.truncated_jsonl}")
    print(f"[sanity]   cut_index={sel.cut_index}  ts={sel.cut_timestamp}")
    print(f"[sanity]   n_entries_in_slice={sel.n_entries_in_slice}  "
          f"n_results_in_slice={sel.n_results_in_slice}")
    print(f"[sanity]   payload_bytes={sel.payload_bytes}")
    w_payload = sel.payload_path.read_text(encoding="utf-8")

    # Step 2: render side-by-side prompts (always).
    side = render_side_by_side(w_payload)
    (out_dir / "prompts_side_by_side.txt").write_text(side, encoding="utf-8")
    print(f"[sanity] wrote {out_dir / 'prompts_side_by_side.txt'}")

    if args.select_w_only or args.dry_run:
        print("[sanity] dry-run / select-only mode → stopping before API.")
        return 0

    if not args.execute:
        print("[sanity] refusing to invoke `claude --print` without --execute flag.")
        return 2

    # Step 3: execute subject calls (requires --execute).
    # Uses subprocess `claude --print` (NOT the Anthropic SDK) to run on
    # the Claude subscription — same invocation pattern as
    # agents/claude_agent_runner.py:431-478.
    subj = cfg["subject"]
    taxonomy = list(cfg[args.taxonomy])
    llm_calls_path = out_dir / "llm_calls.jsonl"

    records_a = run_condition(
        model=subj["model"],
        condition="without_w",
        w_payload=w_payload,
        n_calls=subj["n_calls_per_condition"],
        out_jsonl=llm_calls_path,
        cwd=out_dir,
    )
    records_b = run_condition(
        model=subj["model"],
        condition="with_w",
        w_payload=w_payload,
        n_calls=subj["n_calls_per_condition"],
        out_jsonl=llm_calls_path,
        cwd=out_dir,
    )
    print(f"[sanity] wrote {len(records_a) + len(records_b)} records to "
          f"{llm_calls_path}")

    # Step 4: classify.
    cls_cfg = cfg["classifier"]
    labels_a: list[str] = []
    labels_b: list[str] = []
    cls_path = out_dir / "classifications.jsonl"
    with cls_path.open("w", encoding="utf-8") as f:
        for rec_list, labels in ((records_a, labels_a), (records_b, labels_b)):
            for rec in rec_list:
                if rec.error:
                    lbl = "other"
                else:
                    res = classify_proposal(
                        model=cls_cfg["model"],
                        proposal_text=rec.response_text,
                        taxonomy=taxonomy,
                        cwd=out_dir,
                    )
                    lbl = res.label
                labels.append(lbl)
                f.write(json.dumps({
                    "call_id": rec.call_id,
                    "condition": rec.condition,
                    "label": lbl,
                }) + "\n")

    # Step 5: bootstrap + report.
    boot = cfg["bootstrap"]
    result = bootstrap_g_w(
        labels_without=labels_a,
        labels_with=labels_b,
        modes=taxonomy,
        n_resamples=boot["n_resamples"],
        seed=boot["seed"],
    )
    report = render_report(result)
    (out_dir / "sanity_report.md").write_text(report, encoding="utf-8")
    (out_dir / "sanity_report.json").write_text(
        json.dumps({
            "verdict": result.verdict,
            "G_W_point": result.G_W_point,
            "G_W_ci_low": result.G_W_ci_low,
            "G_W_ci_high": result.G_W_ci_high,
            "H_without": result.H_without,
            "H_with": result.H_with,
            "TV": result.TV_point,
            "KL": result.KL_point,
            "modes": list(result.modes),
            "p_without": list(result.p_without),
            "p_with": list(result.p_with),
            "w_t_provenance_model": cfg["source_swarm_run"]["provenance_model"],
            "subject_model": subj["model"],
            "taxonomy": args.taxonomy,
            "n_without": result.n_without,
            "n_with": result.n_with,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[sanity] verdict: {result.verdict}")
    print(f"[sanity] G^W = {result.G_W_point:+.4f}  "
          f"CI=[{result.G_W_ci_low:+.4f}, {result.G_W_ci_high:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
