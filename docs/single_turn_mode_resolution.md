# SINGLE_TURN_MODE vs. §9.2 Estimators — Resolution Options

**Status:** planning document. Blocks the main pilot, NOT §9.6. Do not implement any option in this document without separate approval.

## Background

`src/agent_swarms/swarm_agent_runner.py:53` sets `SINGLE_TURN_MODE = True`. A swarm agent is therefore executed as **one long `claude --print` call per agent per run**, inside which the agent autonomously loops — proposing, running training, reading the blackboard, proposing again — entirely within a single autoregressive LLM generation.

The BP-revised framework (§9.2) requires per-state estimators:
- `π̂^d(s)` — the agent's spontaneous mode distribution at *zero-feedback* states
- `π̂^d_D(s)` — posterior conditioned on a state bin `D`
- `q̂^d_D(s)` — empirical routing policy at state bin `D`
- `Ĝ(d) = H(π̂^d) − E_D[H(π̂^d_D)]`, `ε̂(d) = Σ_D p̂(D)·KL(π̂^d_D ‖ q̂^d_D)`

These estimators implicitly assume that the successive proposals produced during a run can be treated as samples whose dependence on the context state `D` is the main source of variation, with residual intra-call correlation absorbed into state-bin noise. Under `SINGLE_TURN_MODE=True` that assumption is stressed in three ways:

1. **Autoregressive coupling.** Proposals `s_i, s_{i+1}, …` inside one call are drawn from an autoregressive distribution that is not `π(· | c_i)` but `π(· | c_0, s_1, …, s_{i-1})`. Treating them as i.i.d. draws from a state-conditional distribution underestimates variance.
2. **State boundaries are not clean.** State bin `D` is defined by observable features (val_bpb bucket, recent-trend, last-mode). Inside a single call the LLM's internal representation carries far more than these features, so `π̂^d_D` conflates LLM private state with the observable state.
3. **Bootstrap validity.** Bootstrapping over proposals as if independent gives nominal CI that are too tight, inflating the false-positive rate on `Ĝ`, `ε̂`, and derivatively on `r̂` and `Ĝ^W`.

## Options

### Option A — Refactor to multi-turn, one LLM call per proposal

**Description.** Replace the single long call with a loop in the Python runner: each iteration builds a prompt containing `(system_prompt, local_history, W slice at read time, turn instructions)` and issues one `claude` API call returning exactly one proposal (plus optional reasoning trace). Training execution, blackboard writes, and blackboard reads happen in Python between calls. The agent's "thinking" is reset to the prompt boundary each turn.

**Framework-fidelity cost.** Minimal. Each proposal is drawn from a context the runner can freeze, log, and re-use for leave-one-out estimators. `π^d_D` is well-defined because `D` is literally the prompt content. `G^W(t)` is measurable because the W slice at proposal time is known and byte-addressable. Dual-lens estimators work as specified in §9.2. This is the option the framework as written tacitly assumes.

**Refactor cost.** Medium. Touches `swarm_agent_runner.py`, the system prompt (must be rewritten from "loop autonomously" to "respond with exactly one proposal"), the first-message template, and the agent/worker coordination (training must be triggered by the runner not by the model). The blackboard read/write logic migrates from `coordinator.py` (invoked by the model via bash) to Python callsites in the runner. Existing `coordinator.py` CLI can be retained as a thin compatibility shim for humans. Estimated ~400–800 LOC of churn in the swarm repo; no changes to the parallelisation repo.

**Pilot compute cost.** Neutral to slightly higher. Prompt token cost grows because each turn carries a rendered `local_history` and `W slice` instead of relying on the model's KV cache. Offset: more turns means more observable proposal events, so fewer repetitions are needed to hit K=10 per condition. Net: within ±15% of current.

**Risks.** (i) Agents may behave qualitatively differently when forced into one-shot turns vs. autonomous looping; this is itself a confound unless Option C is layered on top. (ii) The "read W slice" hook must be installed atomically with the refactor, otherwise turn-boundary W snapshots are inconsistent with the prompt that followed them.

### Option B — Reformulate §9.2 estimators for intra-call sequences

**Description.** Keep `SINGLE_TURN_MODE=True`. Log the ordered proposal sequence `(s_1, s_2, …, s_n)` from each run along with classified modes. Redefine estimators to work on blocks of correlated draws: (a) use a block bootstrap over contiguous sub-sequences rather than a proposal bootstrap, (b) redefine `π^d_D(s)` as the within-run empirical transition distribution conditioned on a state bin, (c) add an explicit autocorrelation diagnostic (effective sample size via integrated autocorrelation time) to every reported CI, (d) treat within-run proposal count as a random variable and marginalize over it.

**Framework-fidelity cost.** High. The framework as written defines `π^d(s)` as the distribution over proposals at *zero-feedback* states (§9.2) — the natural operationalization is the first proposal of each independent run. Under `SINGLE_TURN_MODE` there is exactly one zero-feedback state per run; `K=10` runs give 10 draws for `π̂^d`, which is too sparse for a useful Miller–Madow-corrected entropy estimate over ≥7 modes. The block-bootstrap patch relieves the variance-underestimation problem but does not fix the sparse-prior-estimator problem. `G^W` requires comparing `π^Single` to `π^Swarm | W_t` with enough samples to resolve a few-tenths-of-a-nat difference; with 10 first-proposal draws per condition, CI will not resolve anything useful.

**Refactor cost.** Low. Mostly an analysis-layer change plus a thin logging hook that numbers proposals within a run. Maybe 100–200 LOC in a new `analysis/` module.

**Pilot compute cost.** Lowest of the three. No change to runner behavior; K can be kept at 10.

**Risks.** (i) Under-powered estimators that deliver wide CI, making RQ2's mechanism-level claims unfalsifiable. (ii) The block-bootstrap's block length is a free parameter that becomes a sensitivity axis the framework never intended. (iii) Any finding may be challenged on the grounds that the autocorrelation was not adequately modeled.

### Option C — Run both configurations and compare

**Description.** Treat `SINGLE_TURN_MODE=True` and `SINGLE_TURN_MODE=False` as two separate swarm configurations in the pilot. Apply Option A to the multi-turn configuration and Option B to the single-turn configuration. Report them side by side. The comparison itself becomes an additional empirical observable: does the agent's per-configuration `∆̂, Ĝ, ε̂` change when the run topology is flipped?

**Framework-fidelity cost.** Low for the multi-turn leg (matches the framework). Medium for the single-turn leg (still constrained by Option B's sparse-prior issue) but serves as an ablation rather than the main measurement.

**Refactor cost.** Highest. Option A's refactor plus Option B's analysis plus extra experimental conditions plus the cross-configuration comparison protocol. Estimate 1.5× Option A.

**Pilot compute cost.** Largest. At least 1.5× the baseline pilot: the three core configurations (Single-2×, Parallel-2, Swarm-2) under the multi-turn runner, plus Swarm-2 under the single-turn runner as an ablation. Roughly 90 GPU-hours instead of 60 for the minimum viable pilot.

**Risks.** (i) Two ways of running the agent is two ways of being wrong; confound interpretation. (ii) Budget may not accommodate the extra condition.

## Recommendation

**Adopt Option A** for the main pilot.

Rationale: (1) the framework's estimators are defined against the multi-turn structure and were not designed to degrade gracefully under autoregressive coupling; (2) Option B's sparse-prior problem is not fixable by analysis alone without redefining `π^d` in a way that loses the §3.3 configuration-dependence interpretation; (3) the refactor cost is moderate and lands in one file-family (`swarm_agent_runner`, `coordinator`, templates); (4) Option C's extra comparison is defensible as a second-sprint ablation but should not gate the first pilot.

Concrete next step when approval is granted: scope the refactor in a follow-up planning doc, identify the minimum set of `coordinator.py` CLI commands that must be preserved for human use, and draft a migration of `cmd_think`/`cmd_claim`/`cmd_publish` into direct Python calls from the runner.

Do NOT implement any of this without separate approval.
