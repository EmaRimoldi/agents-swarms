# §9.6 Precondition Sanity — Escalation Plan

**Status:** pre-registered BEFORE any data collection. Any deviation requires explicit approval.

## Primary observable

Let `p_A` be the empirical mode distribution elicited with `W_t` absent (= `π̂^Single`) and `p_B` the empirical mode distribution elicited with `W_t` present (= `π̂^Swarm | W_t`). With Miller–Madow-corrected Shannon entropy `Ĥ`, the primary observable is

    Ĝ^W := Ĥ(p_A) − Ĥ(p_B)    (nats)

Sign is retained. Negative `Ĝ^W` (W disperses the prior) is a valid PASS — it means the model reads W as a brainstorm-expander, which is informative and consistent with the framework's stratified account of shared memory.

CI: bootstrap B = 2000 resamples over per-proposal classified labels, independently per condition.

## Pass/margin/fail at each level

Applied uniformly at every escalation level:

- **PASS** — bootstrap 95% CI on `Ĝ^W` excludes 0 AND `|Ĝ^W|_lower_bound > 0.1 nats`
- **MARGIN** — CI excludes 0 but `|Ĝ^W|_lower_bound ≤ 0.1 nats`
- **FAIL** — CI on `Ĝ^W` includes 0

A MARGIN result at any level is partial evidence. It allows proceeding to the main pilot with a documented caveat in the pilot's pre-registration. MARGIN does NOT trigger escalation.

A PASS at any level ends escalation. A FAIL at a given level triggers the next level (if any).

## Escalation order (taxonomy refinement before model escalation)

The order deliberately exhausts taxonomy granularity at each model tier before escalating model size. Rationale: a FAIL from mode-granularity coarseness (the model shifts behavior *within* a coarse mode) is a taxonomy artifact, not a precondition failure, and the cheapest check against a false negative is to re-run the same model with a finer taxonomy.

| Level | Model                     | Taxonomy         | Trigger to run |
|-------|---------------------------|------------------|----------------|
| 1     | `claude-haiku-4-5-20251001` | `taxonomy_coarse` (7 modes) | always (this is the planned §9.6 experiment) |
| 2     | `claude-haiku-4-5-20251001` | `taxonomy_fine` (15–20 modes) | Level 1 = FAIL |
| 3     | `claude-sonnet-4-6`       | `taxonomy_coarse` | Level 2 = FAIL |
| 4     | `claude-sonnet-4-6`       | `taxonomy_fine`   | Level 3 = FAIL |
| 5     | `claude-opus-4-6`         | `taxonomy_coarse` | Level 4 = FAIL |
| 6     | `claude-opus-4-6`         | `taxonomy_fine`   | Level 5 = FAIL |
| 7     | — stop, reconsider framework assumptions | — | Level 6 = FAIL |

## Protocol invariants across levels

- `W_t` is the same slice at every level (truncated from the Q3-selected swarm run at the median `result` entry). The payload is captured byte-identically once and reused.
- Template system prompt and first-message template are identical across levels; only the `--model` parameter and the taxonomy YAML change.
- `K = 300` independent single-proposal API calls per condition, `B = 2000` bootstrap resamples, `temperature` fixed at the production default.
- Mode classifier is always `claude-haiku-4-5` at `temperature=0` regardless of the subject model, for deterministic labeling. When the taxonomy changes (coarse ↔ fine), the classifier prompt changes with it.
- Each level produces its own output directory under `runs/sanity/level_<N>/` with `llm_calls.jsonl`, `mode_counts.json`, `sanity_report.{md,json}`.

## Worst-case compute cap

Level 1 nominal inference count: 600 subject calls + 600 classifier calls = 1200 API calls (~$ few, ~30 min parallelized wallclock).

Worst case (all six levels execute before a PASS) = 6 × Level 1 ≈ 7200 API calls. **Pre-register this cap.** If any level's cost exceeds its nominal by >25% (due to retries, prompt edits, or classifier disagreement), stop and re-request approval before continuing the chain.

## W_t provenance caveat

The selected `W_t` was produced by `claude-sonnet-4-6` (swarm run `experiment_exp_20260406_024115`), not by Haiku. The W payload is therefore stylistically shaped by Sonnet's proposal history. This is an acknowledged entanglement and must appear in every level's report header. A FAIL at Level 1 with a PASS at Level 3 cannot be cleanly attributed to model strength vs. stylistic familiarity of the W content; this is the interpretive limit of re-using one fixed `W_t` across levels.

## Do NOT implement levels ≥2 now

Only Level 1 is in scope for the current sprint. Levels 2–6 are drafted for transparency and to pre-register the decision rule; their code and execution are gated on observing a Level 1 FAIL and obtaining fresh approval.
