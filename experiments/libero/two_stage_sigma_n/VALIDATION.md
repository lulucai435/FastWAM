# Packaging validation — 2026-09-29

All checks used CPU only; no model weights were loaded and no GPU job was started.

1. `python -m unittest discover -s experiments/libero/two_stage_sigma_n -p 'test_*.py' -v`:
   **27 tests passed**. Includes actual model initialization extracted by AST,
   the repository scheduler, proper-N/shared-first equivalence, frozen s/d
   definitions, labels, leakage isolation and corrupted-artifact rejection.
2. Ran the portable `analyze.py` entrypoint on the already-reviewed 772-state
   Phase A error table (private samples, not committed). Every scientific summary
   field exactly matched the original analysis, and all 28 complete fold mappings
   matched exactly. The deduplicated Phase B requirement remains 828 pairs.
   File hashes/provenance differ as expected because the packaging paths differ.
3. The portable analysis independently recomputed and reviewed labels, mapping
   metrics, associations, task-bin spread and the Phase B pair union.

This is a code-packaging validation, not a new GPU experiment. GPU Phase B hidden
capture and compiled-path equivalence remain pending. The previous Phase A raw
artifact review remains the evidence for those measurements; trusting the compact
error table does not revalidate missing raw actions or input observations.
