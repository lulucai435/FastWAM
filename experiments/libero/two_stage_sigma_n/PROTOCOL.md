# LIBERO two-stage sigma → N audit

Prepared before new GPU results. Full objective remains Phase A AND Phase B;
no adaptive closed-loop or final deployed allocator is authorized.

## Frozen inputs and sampler

Reuse all 772 H32/E16 frozen states /28 tasks /4 suites, previous normalized
32×7 chunks and existing future-real reliability s. Prior p is previous[16:32].
Same image, instruction, normalized 8D proprio, checkpoint and seed42 Gaussian
epsilon across candidates. Only first16 action positions mix (1−sigma)p+sigma*eps;
remaining16 positions start at unscaled eps. No extra action clamping or padding.

Preserve the original continuous-flow scheduler, shift=5. Sigma and N are
independent: u0=sigma/(5−4sigma); use N equal intervals in u from u0 to0 and
phi(u)=5u/(1+4u), with the original float32 construction/BF16 casts and Euler step.
Sigma=1,N=10 is the original fresh branch with no prior argument. Record actual
runtime timesteps/deltas, not only ideal formulas. Sigma=0,N=0 directly executes
p; no Video prefill or Action-DiT forward is needed for this branch.

Error is float32 RMSE over first16×all7 normalized action values against the
same-state fresh sigma=1,N=10 reference. Primary threshold is strict error<0.01.

## Phase A

Sweep sigma grid0,.1,...,1. Positive sigma uses N=10. Sigma=1 is also the fresh
reference, not an extra run. Cost100 Action-DiT NFE/state,77,200 measured NFE for
772 states;30 standard warmup NFE. No new d or layer search is needed for Phase A.
Conservative sigma* is the lowest grid value whose candidate and ALL larger
sigmas pass. Report raw errors, monotonicity violations, distribution, direct
reuse fraction, pooled/raw-task-demeaned Spearman, per-task signs and dependence.
Also report actual direct-reuse action validity separately from sigma*=0.

## Frozen mapping g and honest task isolation

Outer leave-one-task-out (28 folds). Fit g using ONLY the other27 tasks:
5 quantile bins of training s; task-balanced empirical90th percentile of sigma*
in each bin (each state weight1/n_training_task); prefix maximum enforces
nondecreasing sigma. Empty bins use the global training weighted90th percentile.
Exact ties use right-inclusive bin assignment via searchsorted(side='right');
predictions stay on the11-value sigma grid. No heldout tuning of bin count,
coverage or monotonicity direction. Save all boundaries, supports and mappings.

For each outer fold, apply THAT fold's g both to its training states and heldout
states. Never train the N predictor using global-OOF g values for training states:
those mappings can include the outer heldout task's sigma labels. Deduplicate the
required (state,sigma) pairs across folds after Phase A to bound Phase B compute.

## Phase B

At every required positive (state,sigma), capture the frozen d from the actual
first denoising forward: layer29 POST-layer action hidden H, all32 action tokens,
mean_d sqrt(mean_token (H−mean_token H)^2), float32 as in the historical definition.
Do not reuse a sigma=1 d at another sigma; do not search another layer/statistic.
The captured velocity MUST update each candidate's first step using its own delta.
Reuse that exact first result across independent N=1..10 schedules only after
checking same first latent, timestep and conditioning. No N10 prefix truncation.
Cost per positive pair is1+sum(N−1,N=1..10)=46 network calls, not55+an extra probe.
Production chosen-N cost remains N, including the d-producing first forward.

Phase B implementation must compare the first-capture path with the ordinary
fresh sigma=1,N=10 path on explicit preflight inputs. Do not silently change the
scheduler/baseline or introduce an archived-action/KV replay-equality gate.
Any observable numerical baseline discrepancy must be reported and resolved or
explicitly scoped before claiming equivalence; do not silently relax a threshold.

Use conservative suffix N* over1..10 at selected sigma, same first16 reference
error threshold. If N10 fails, N* is INFEASIBLE, not10 or a fabricated11. Keep these
states in coverage/unsafe-selection reporting, but do not regress an invented label.
At mapped sigma=0, execute p and set actual cost0; d is absent, not a measured zero.
Report failed direct reuse explicitly. Do not let correct trivial N0 assignments
hide failure on the positive-sigma refinement branch.

Main models: fixed-alpha1 ridge with train-only scaling, s-only vs(s,d), using
training-task feasible positive-sigma labels under the same outer g. Primary N
predictions ceil(clip(continuous,1,10)); report integer macro-task MAE, pooled MAE,
under-allocation rate, mean predicted N, per-task results; continuous errors as
secondary. If no eligible training rows, fixedN10 fallback is explicit, not fitted.
Report feasible-positive coverage and unavailable targets; direct-reuse decisions
and action-criterion failures separately. For complete-policy metrics, consult
saved actions at predicted N (not only N<N*), since suffix labels are conservative.

Within fixed s bins, compare same-task-cell raw target spread and heldout residual
spread, eligible task cells>=3 states. d correlations are on feasible positive
sigma states and stratified by mapped sigma where support permits.

Oracle savings: 1−mean(N*)/10 on feasible selections, with their coverage. Also
report an explicitly charged N10 fresh fallback for infeasible sigma/reuse choices;
this is an offline oracle fallback accounting, not a demonstrated online policy.
All warmup, preflight, Phase A and Phase B audit-call costs are separate from ideal
allocation savings. No environment rerun, training of a final allocator or SR claim.

## Deliverable

Short English report: s→sigma* associations; frozen outer-fold g; d→N* associations;
s-only vs s+d heldout metrics; direct reuse; oracle NFE savings; task heterogeneity;
verdicts s→sigma PASS/PARTIAL/FAIL, d adds information YES/PARTIAL/NO, two-stage
allocator PROMISING/WEAK. Do not stop at Phase A and call the full objective complete.

Historical execution update: Phase A completed and passed independent review.
Phase B remains unexecuted; this code publication does not authorize GPU jobs.
See README.md and PHASE_A_RESULTS.md for current implementation boundaries.
