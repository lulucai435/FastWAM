# Two-stage sigma → N audit: Phase A results

## Conclusion

**s → sigma: PARTIAL.** The frozen future-real consistency signal contains useful
information about required starting noise, but it is not a task-independent safe
selector. Pooled Spearman rho is 0.442, task-demeaned rho is 0.391, and 24/28 tasks
have positive correlations. The predeclared conservative held-out mapping still
violates the action-error threshold on 32/772 states (4.15%). No state permits
direct old-tail reuse at the primary threshold.

This is an interpretation of Phase A, not a prespecified statistical acceptance
test. Whether first-step layer29 dispersion adds information about N remains
untested. The full two-stage verdict and compute savings are pending Phase B.

## Setting and purpose

- Cohort: 772 frozen H32/E16 control states, 28 tasks, four LIBERO suites.
- Purpose: isolate whether s predicts how much starting noise is required, with
  refinement compute fixed at N=10; sigma and N are independent.
- Previous-tail prior p = previous normalized action chunk[16:32]. For the first
  16 positions, initialize (1−sigma)p + sigma epsilon; the remaining 16 start at
  unscaled epsilon. Reuse observations, instructions, proprioception and noise.
- Sweep sigma = 0, 0.1, ..., 1. Positive sigma uses independent proper N=10 runs.
  Sigma=0 directly returns p without a network call. Sigma=1 uses the original
  fresh Fast-WAM branch and also supplies the reference, without an extra run.
- Preserve the original continuous-flow scheduler, shift=5, compiled BF16 path.
- Error: float32 normalized RMSE over the first 16 × all seven action dimensions,
  relative to the same-state fresh sigma=1, N=10 result. Passing means error<0.01.
- Conservative sigma*: the lowest grid value for which it and every larger sigma
  pass. This is not merely the first individually passing candidate.

These are action-agreement results on frozen states, **not rollout success rates**.
Smaller sigma alone does not reduce NFE: every positive Phase A candidate uses 10.

## Main results

| Metric | Result |
|---|---:|
| States / tasks / suites | 772 / 28 / 4 |
| Pooled Spearman(s, sigma*) | 0.441574 |
| Spearman after within-task demeaning | 0.390813 |
| Positive / negative task correlations | 24 / 4 |
| Mean / median sigma* | 0.7680 / 0.8 |
| Conservative direct reuse (sigma*=0) | 0/772 |
| Direct old-tail action passes error<0.01 | 0/772 |
| States with any error increase >1e−6 as sigma increases | 464/772 |

Raw errors are not strictly monotone even though the threshold-pass counts are
nearly nested. The conservative suffix definition was retained unchanged.

| Oracle sigma* | 0–0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| States | 0 | 1 | 25 | 87 | 248 | 227 | 114 | 70 |

## Task dependence

Suite correlations below are pooled within each suite, not task-macro correlations.
Each suite contributes seven tasks; the cohort is not state-balanced across suites.

| Suite | States | Spearman(s, sigma*) | Mean sigma* | Held-out g threshold failures |
|---|---:|---:|---:|---:|
| LIBERO-10 | 327 | 0.2883 | 0.7505 | 14/327 |
| LIBERO-Goal | 164 | 0.4736 | 0.8354 | 9/164 |
| LIBERO-Object | 153 | 0.6687 | 0.7124 | 6/153 |
| LIBERO-Spatial | 128 | 0.3331 | 0.7930 | 3/128 |

Negative task correlations occur in LIBERO-10/task5 (−0.181), Goal/task6 (−0.048),
Goal/task9 (−0.115), and Spatial/task5 (−0.471). Object has seven positive tasks.
Within the five frozen s bins, eligible task-cell mean sigma* spreads are 0.325,
0.208, 0.243, 0.175, and 0.290 (at least three states per cell). Task dependence
therefore remains; a positive pooled correlation does not establish universal
calibration. Repeated states from the same tasks/episodes are not independent trials.

Full task results: [per_task.csv](phase_a_results/per_task.csv).

## Frozen task-held-out mapping g(s)

Leave one task out, fit on the other 27 only. Use five training-s quantile bins,
task-balanced empirical 90th-percentile sigma* per bin, then prefix maxima to
enforce a nondecreasing mapping. No held-out tuning was performed.

All 28 folds produce sigma values **[0.8, 0.9, 0.9, 0.9, 1.0]** from low to high
s bins. Their bin boundaries are separately learned from training tasks and are
not one global threshold set. Full boundaries/supports are frozen in
the private run's `FOLDS.json` (not distributed with this code-only release).

| Held-out metric | Result |
|---|---:|
| Selected sigma 0.8 / 0.9 / 1.0 | 156 / 462 / 154 states |
| Mean selected sigma | 0.899741 |
| Pooled sigma MAE against conservative oracle | 0.140285 |
| Macro-task sigma MAE | 0.134896 |
| Below conservative sigma* | 32/772 (4.1451%) |
| Actual selected-action threshold failure | 32/772 (4.1451%) |
| Macro-task threshold failure rate | 3.8305% |
| Actual selected-action threshold pass | 740/772 (95.8549%) |
| Selected direct reuse | 0/772 |

The last pass rate is action agreement, not LIBERO SR or a deployment safety
guarantee. The conservative quantile mapping intentionally overestimates sigma
on many states; it does not eliminate threshold failures. The 32 failures are
already infeasible at selected-sigma N=10 under the planned suffix-N criterion;
Phase B must report them rather than invent N*=10 labels.

Task-level mapping results: [heldout_g_per_task.csv](phase_a_results/heldout_g_per_task.csv).

## Execution and independent verification

Job **pt-su5qdf1o**: SUCCEEDED, 2026-09-28 01:49:06–02:10:04 Asia/Shanghai
(20m58s). One NVIDIA H100 80GB HBM3, PyTorch 2.7.1+cu128, CUDA 12.8, compiled BF16.
Measured Action-DiT NFE: 77,200; warmup: 30; total: 77,230. Zero environment steps.

The frozen reviewer verified all 772 raw bundles and original-input hashes,
prior/noise initialization, actual first network inputs, sigma-specific timesteps
and deltas, NFE counts, float32 errors with a float64 cross-check, and suffix labels.
A separate reference computation checked mappings, training-task exclusion, the
Phase B pair union, correlations, mapping metrics and task-bin spread.
All checks passed. This does not claim exact replay equality to archived actions;
the reference is the fresh original branch in this same run.

- [Portable summary](phase_a_results/summary.json) (local-path provenance omitted).
- [Source and original-summary hashes](PROVENANCE.json).
- Full raw bundles, state-level tables, fold mappings and the review marker remain
  in the private experiment archive; they are not included in this code release.

## Phase B boundary

Applying each outer fold's own g to both its training and test states produces
**828 distinct positive (state, sigma) pairs**, after deduplication across folds.
This is more than 772 because a state can receive different sigmas across folds.
The planned independent N=1..10 sweep shares the actual first NFE and therefore
needs 46 calls per pair: **38,088 nominal measured Action-DiT NFE**, excluding
preflight and warmup. This is audit cost, not an online-policy cost or saving.

The private run's `PHASE_B_PAIRS.json` remains gpu_authorized=false.
The first-forward GPU hidden adapter and baseline-equivalence checks remain to
be completed, costed and separately authorized before submission. No Phase B job
or final adaptive closed-loop was launched.
