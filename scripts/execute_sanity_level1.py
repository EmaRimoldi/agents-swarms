#!/usr/bin/env python3
"""Execute §9.6 sanity experiment Level 1.

Protocol:
  1. select_w() → byte-identical W_t payload
  2. probe: 10 without_w + 10 with_w via ThreadPoolExecutor(max=8)
  3. if probe clean → full 300+300 parallel; else serial fallback
  4. classify all 600 proposals (parallel, same worker cap)
  5. bootstrap G^W, render enriched report
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_swarms.sanity.select_w import select_w                        # noqa: E402
from agent_swarms.sanity.run_with_without import (                        # noqa: E402
    CallRecord,
    PROPOSAL_SYSTEM_PROMPT,
    _invoke_claude_print,
    _sha256_of,
    build_user_message,
    render_side_by_side,
)
from agent_swarms.sanity.mode_classifier import classify_proposal         # noqa: E402
from agent_swarms.sanity.stats import (                                   # noqa: E402
    bootstrap_g_w,
    entropy_mm,
    render_report,
)

MAX_WORKERS = 8


def _one_subject_call(k: int, condition: str, model: str,
                      user_msg: str, prompt_sha: str, prompt_char_len: int,
                      cwd: Path, timeout_s: int) -> CallRecord:
    t0 = time.time()
    rc, stdout, stderr = _invoke_claude_print(
        model=model,
        system_prompt=PROPOSAL_SYSTEM_PROMPT,
        user_message=user_msg,
        cwd=cwd,
        timeout_seconds=timeout_s,
    )
    t1 = time.time()
    text = stdout or ""
    err = None if rc == 0 else (stderr.strip() or f"rc={rc}")
    return CallRecord(
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
        est_input_tokens=prompt_char_len // 4,
        est_output_tokens=len(text) // 4,
        return_code=rc,
        error=err,
    )


def run_subject_parallel(model: str, condition: str, w_payload: str,
                         n_calls: int, out_jsonl: Path, cwd: Path,
                         max_workers: int, timeout_s: int = 180,
                         start_k: int = 0) -> list[CallRecord]:
    user_msg = build_user_message(condition, w_payload)
    prompt_sha = _sha256_of(PROPOSAL_SYSTEM_PROMPT + "\n" + user_msg)
    prompt_char_len = len(PROPOSAL_SYSTEM_PROMPT) + 1 + len(user_msg)
    records: list[CallRecord] = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(_one_subject_call, start_k + k, condition, model,
                          user_msg, prompt_sha, prompt_char_len, cwd, timeout_s)
                for k in range(n_calls)
            ]
            for i, fut in enumerate(as_completed(futs)):
                rec = fut.result()
                records.append(rec)
                f.write(json.dumps(asdict(rec)) + "\n")
                f.flush()
                if (i + 1) % 25 == 0 or (i + 1) == n_calls:
                    errs = sum(1 for r in records if r.error)
                    print(f"  [{condition}] {i + 1}/{n_calls}  errors={errs}")
    return records


def run_subject_serial(model: str, condition: str, w_payload: str,
                       n_calls: int, out_jsonl: Path, cwd: Path,
                       timeout_s: int = 180, start_k: int = 0) -> list[CallRecord]:
    user_msg = build_user_message(condition, w_payload)
    prompt_sha = _sha256_of(PROPOSAL_SYSTEM_PROMPT + "\n" + user_msg)
    prompt_char_len = len(PROPOSAL_SYSTEM_PROMPT) + 1 + len(user_msg)
    records: list[CallRecord] = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a", encoding="utf-8") as f:
        for k in range(n_calls):
            rec = _one_subject_call(start_k + k, condition, model, user_msg,
                                    prompt_sha, prompt_char_len, cwd, timeout_s)
            records.append(rec)
            f.write(json.dumps(asdict(rec)) + "\n")
            f.flush()
            if (k + 1) % 25 == 0 or (k + 1) == n_calls:
                errs = sum(1 for r in records if r.error)
                print(f"  [{condition}] {k + 1}/{n_calls}  errors={errs}")
    return records


def classify_all_parallel(records: list[CallRecord], model: str,
                          taxonomy: list[str], cwd: Path,
                          max_workers: int, out_jsonl: Path) -> list[str]:
    def _one(rec: CallRecord):
        if rec.error or not rec.response_text.strip():
            return rec, "other", "(skipped: error or empty)", -9
        res = classify_proposal(model=model, proposal_text=rec.response_text,
                                taxonomy=taxonomy, cwd=cwd)
        return rec, res.label, res.raw, res.return_code

    labels: list[tuple[str, str]] = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_one, r) for r in records]
            for i, fut in enumerate(as_completed(futs)):
                rec, lbl, raw, rc = fut.result()
                labels.append((rec.call_id, lbl))
                f.write(json.dumps({
                    "call_id": rec.call_id,
                    "condition": rec.condition,
                    "label": lbl,
                    "raw": raw[:200],
                    "return_code": rc,
                }) + "\n")
                f.flush()
                if (i + 1) % 50 == 0 or (i + 1) == len(records):
                    print(f"  [classify] {i + 1}/{len(records)}")
    # Return labels in the original record order.
    order = {rec.call_id: i for i, rec in enumerate(records)}
    labels.sort(key=lambda x: order[x[0]])
    return [lbl for _, lbl in labels]


def top_k_modes(p: tuple[float, ...], modes: tuple[str, ...], k: int = 3):
    idx = sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:k]
    return [(modes[i], p[i]) for i in idx]


def qualitative_direction(res) -> str:
    dh = res.H_with - res.H_without   # positive → W disperses, negative → concentrates
    tv = res.TV_point
    # Rule of thumb: if |dH| is small relative to TV, the shift is lateral.
    if abs(dh) < 0.05 and tv > 0.10:
        return ("lateral: mass redistributes between modes at roughly "
                f"constant entropy (|ΔH| = {abs(dh):.3f} nats vs TV = {tv:.3f})")
    if dh < 0:
        return (f"concentration: W reduces entropy by {-dh:.3f} nats "
                "(prior tightens onto fewer modes)")
    if dh > 0:
        return (f"dispersion: W increases entropy by {dh:.3f} nats "
                "(prior broadens — W acts as a brainstorm expander)")
    return "no detectable shift"


def render_enriched_report(res, *, subject_model: str, taxonomy_name: str,
                           w_provenance: str, w_selection: dict,
                           probe_summary: dict) -> str:
    base = render_report(res)
    modes = res.modes
    top_a = top_k_modes(res.p_without, modes, 3)
    top_b = top_k_modes(res.p_with, modes, 3)
    argmax_a = modes[int(np.argmax(res.p_without))]
    argmax_b = modes[int(np.argmax(res.p_with))]
    direction = qualitative_direction(res)

    extras = [
        "",
        "## Argmax / top-3 per condition",
        f"- **Argmax without W:** `{argmax_a}` (p = {max(res.p_without):.3f})",
        f"- **Argmax with W:**    `{argmax_b}` (p = {max(res.p_with):.3f})",
        "",
        "| rank | without_w | p | with_w | p |",
        "|---|---|---|---|---|",
    ]
    for i in range(3):
        ma, pa = top_a[i]
        mb, pb = top_b[i]
        extras.append(f"| {i+1} | {ma} | {pa:.3f} | {mb} | {pb:.3f} |")

    extras += [
        "",
        "## Qualitative direction of shift",
        f"{direction}",
        "",
        "## Provenance and configuration",
        f"- **Subject model:** `{subject_model}` (invoked via `claude --print` subprocess)",
        f"- **W_t provenance:** `{w_provenance}` — this W_t was produced by a Sonnet swarm "
        f"run and is read here by Haiku. Interpretation note: a FAIL with Sonnet-produced "
        f"W_t is a strong negative signal (the content is richer than any Haiku-produced "
        f"W_t would be). A PASS cannot be cleanly disentangled from stylistic familiarity "
        f"with Sonnet's phrasing, though the framework's claim is only that W is read.",
        f"- **Taxonomy:** `{taxonomy_name}` (pre-registered in configs/sanity_w_read.yaml)",
        f"- **Source swarm run:** `{w_selection['source_run']}`",
        f"- **W_t cut index:** {w_selection['cut_index']}  timestamp: {w_selection['cut_timestamp']}",
        f"- **W_t entries:** {w_selection['n_entries_in_slice']} "
        f"(of which {w_selection['n_results_in_slice']} are results)",
        f"- **W_t payload size:** {w_selection['payload_bytes']} bytes",
        "",
        "## Parallelization probe",
        f"- max_workers: {probe_summary['max_workers']}",
        f"- probe n_without={probe_summary['n_without']} n_with={probe_summary['n_with']}",
        f"- probe errors: {probe_summary['errors']}",
        f"- probe wallclock: {probe_summary['wallclock']:.1f} s",
        f"- decision: **{probe_summary['decision']}**",
        "",
        "## Deviations from production config",
        "- `claude --print` subprocess invocation (same as production), "
        "but with an added `--model <model>` flag not used by the production runner.",
        "- Temperature is the `claude --print` CLI default (no `--temperature` flag). "
        "Nominal pre-registered temperature = 1.0 is documentation of intent only.",
        "- Mode classifier is a separate `claude --print` subprocess, not a rule-based "
        "labeler; pre-registered system prompt in `sanity/mode_classifier.py`.",
    ]
    return base + "\n" + "\n".join(extras)


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "configs/sanity_w_read.yaml").read_text())
    out_dir = REPO_ROOT / "runs/sanity/level_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STEP 1: select W_t")
    print("=" * 72)
    source_run = REPO_ROOT / cfg["source_swarm_run"]["path"]
    sel = select_w(source_run=source_run, out_dir=out_dir, repo_root=REPO_ROOT)
    print(f"  cut_index          : {sel.cut_index}")
    print(f"  cut_timestamp      : {sel.cut_timestamp}")
    print(f"  n_entries_in_slice : {sel.n_entries_in_slice}")
    print(f"  n_results_in_slice : {sel.n_results_in_slice}")
    print(f"  payload_bytes      : {sel.payload_bytes}")
    w_payload = sel.payload_path.read_text(encoding="utf-8")
    w_selection_dict = json.loads(
        (out_dir / "W_t_selection.json").read_text(encoding="utf-8")
    )

    (out_dir / "prompts_side_by_side.txt").write_text(
        render_side_by_side(w_payload), encoding="utf-8"
    )

    subj = cfg["subject"]
    model = subj["model"]
    timeout_s = subj["timeout_seconds"]
    taxonomy_name = "taxonomy_coarse"
    taxonomy = list(cfg[taxonomy_name])
    llm_calls_path = out_dir / "llm_calls.jsonl"
    # clean slate for this level-1 execution
    if llm_calls_path.exists():
        llm_calls_path.unlink()

    print()
    print("=" * 72)
    print("STEP 2: parallelization probe (10 without_w + 10 with_w, workers=8)")
    print("=" * 72)
    t0 = time.time()
    probe_a = run_subject_parallel(model, "without_w", w_payload, 10,
                                   llm_calls_path, out_dir, MAX_WORKERS,
                                   timeout_s, start_k=0)
    probe_b = run_subject_parallel(model, "with_w", w_payload, 10,
                                   llm_calls_path, out_dir, MAX_WORKERS,
                                   timeout_s, start_k=0)
    probe_wall = time.time() - t0
    probe_errors = sum(1 for r in probe_a + probe_b if r.error)
    probe_empty = sum(1 for r in probe_a + probe_b if not r.response_text.strip())
    print(f"  probe wallclock   : {probe_wall:.1f} s")
    print(f"  probe errors      : {probe_errors}")
    print(f"  probe empty resps : {probe_empty}")

    use_parallel = (probe_errors == 0 and probe_empty == 0)
    decision = ("parallel (workers=8)" if use_parallel
                else "SERIAL fallback (probe had errors/empties)")
    print(f"  decision          : {decision}")

    probe_summary = {
        "max_workers": MAX_WORKERS,
        "n_without": len(probe_a),
        "n_with": len(probe_b),
        "errors": probe_errors,
        "wallclock": probe_wall,
        "decision": decision,
    }

    print()
    print("=" * 72)
    print("STEP 3: full run")
    print("=" * 72)
    full_n = subj["n_calls_per_condition"]
    remaining_a = full_n - len(probe_a)
    remaining_b = full_n - len(probe_b)
    t0 = time.time()
    if use_parallel:
        rest_a = run_subject_parallel(model, "without_w", w_payload,
                                      remaining_a, llm_calls_path, out_dir,
                                      MAX_WORKERS, timeout_s,
                                      start_k=len(probe_a))
        rest_b = run_subject_parallel(model, "with_w", w_payload,
                                      remaining_b, llm_calls_path, out_dir,
                                      MAX_WORKERS, timeout_s,
                                      start_k=len(probe_b))
    else:
        rest_a = run_subject_serial(model, "without_w", w_payload, remaining_a,
                                    llm_calls_path, out_dir, timeout_s,
                                    start_k=len(probe_a))
        rest_b = run_subject_serial(model, "with_w", w_payload, remaining_b,
                                    llm_calls_path, out_dir, timeout_s,
                                    start_k=len(probe_b))
    full_wall = time.time() - t0
    records_a = probe_a + rest_a
    records_b = probe_b + rest_b
    n_errs_a = sum(1 for r in records_a if r.error)
    n_errs_b = sum(1 for r in records_b if r.error)
    print(f"  full run wallclock: {full_wall:.1f} s")
    print(f"  without_w: {len(records_a)} ({n_errs_a} errors)")
    print(f"  with_w:    {len(records_b)} ({n_errs_b} errors)")

    print()
    print("=" * 72)
    print("STEP 4: classify")
    print("=" * 72)
    cls_model = cfg["classifier"]["model"]
    cls_path = out_dir / "classifications.jsonl"
    all_records = records_a + records_b
    all_labels = classify_all_parallel(all_records, cls_model, taxonomy,
                                       out_dir, MAX_WORKERS, cls_path)
    labels_a = all_labels[:len(records_a)]
    labels_b = all_labels[len(records_a):]

    print()
    print("=" * 72)
    print("STEP 5: bootstrap + report")
    print("=" * 72)
    boot = cfg["bootstrap"]
    result = bootstrap_g_w(
        labels_without=labels_a,
        labels_with=labels_b,
        modes=taxonomy,
        n_resamples=boot["n_resamples"],
        seed=boot["seed"],
    )
    report = render_enriched_report(
        result,
        subject_model=model,
        taxonomy_name=taxonomy_name,
        w_provenance=cfg["source_swarm_run"]["provenance_model"],
        w_selection=w_selection_dict,
        probe_summary=probe_summary,
    )
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
            "n_without": result.n_without,
            "n_with": result.n_with,
            "subject_model": model,
            "w_t_provenance_model": cfg["source_swarm_run"]["provenance_model"],
            "taxonomy": taxonomy_name,
            "probe": probe_summary,
            "errors_without": n_errs_a,
            "errors_with": n_errs_b,
            "full_run_wallclock_s": full_wall,
        }, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"verdict : {result.verdict}")
    print(f"G^W     : {result.G_W_point:+.4f} nats")
    print(f"95% CI  : [{result.G_W_ci_low:+.4f}, {result.G_W_ci_high:+.4f}]")
    print(f"report  : {out_dir / 'sanity_report.md'}")
    print(f"json    : {out_dir / 'sanity_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
