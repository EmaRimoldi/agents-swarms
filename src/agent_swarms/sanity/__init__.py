"""§9.6 precondition sanity experiment.

Implements the test that the LLM actually conditions its proposal
distribution on the shared-memory state W_t. See
BP_Revised_Draft.md §9.6 and docs/sanity_escalation_plan.md.

The package is read-only with respect to production swarm code.
The only production-code modification is the opt-in dump_payload
hook on shared_memory.SharedMemory.format_for_context.
"""

__all__ = [
    "select_w",
    "stats",
    "mode_classifier",
    "run_with_without",
]
