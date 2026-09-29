# LIBERO two-stage sigma → N audit

Research code for **s_t → sigma_0,t → d_t(sigma_0,t) → N_t**. This directory
organizes the existing audit; it does not claim a completed adaptive policy.

**Status:** Phase A completed and independently reviewed on 772 frozen states,
28 tasks, 84 episodes, four suites. Phase B has a CPU-tested continuation core,
but the GPU first-forward hidden adapter, baseline-equivalence preflight and
held-out N comparison are **not yet implemented/completed**. No closed-loop SR
claim or speedup claim is made.

## Code map

| Stage | Code | Contract |
|---|---|---|
| Frozen reliability s | `signals.py` | Layer19 V, future-vs-real relative L2, mean of four scalar distances |
| Prior / labels | `definitions.py` | First16 masked old tail; strict RMSE<.01; conservative suffix labels |
| Phase A sigma sweep | `sweep_a.py` | Sigma .1..1, independent N10; sigma0 direct reuse, 100 NFE/state |
| Training-only g(s) | `mapping.py` | Five bins, task-balanced 90th percentile, monotone, leave-one-task-out |
| Analysis / pair freeze | `analyze.py`, `analyze_a.py` | Statistics, fold mappings, deduplicated (state,sigma) requirements |
| Independent reviewer | `review_a.py` | Separate label/mapping/metric calculations and raw-state checks |
| Frozen d and N sweeps | `refinement.py` | Post-layer29 dispersion, all32 tokens; shared actual first NFE |

The repository model files `fastwam.py`, `mot.py`, and
`schedulers/scheduler_continuous.py` are copied from the reviewed Phase A source.
They provide explicit action noise, masked priors, independent start sigma,
sampling metadata and opt-in current/future cache instrumentation. Historical
pooled-feature hooks are retained unchanged, but **pooled hidden features are not
the layer29 token-dispersion d**. In particular, the existing
`action_feature_layers` path is restricted to sigma=1; it is not a Phase B adapter.
No training, RoboTwin or cluster-account configuration changes are included.

## Frozen definitions

`s` compares previous-plan native future slot 1 with the next E16 control state's
current frame, using zero-based Video layer19 V (98×3072). The future noise seeds
are 71,72,73,74. Use the mean of four relative-L2 distances, each normalized by
the real/current feature norm. Larger s means more mismatch / less reliable tail.
Never replace this with the distance of a mean feature. This audit consumes the
already-frozen s; it does not run a new feature/layer search. Cache acquisition
uses first_frame_causal, and Action-DiT attends to current-frame cache only.

For H32/E16, let p=previous_actions[16:32]. Initialize first16 with
(1−sigma)p+sigma*epsilon; last16 remain epsilon. Sigma0,N0 returns p directly.
Keep original scheduler shift=5: start_u=sigma/(5−4sigma), uniformly subdivide u
to zero, then sigma(u)=5u/(1+4u). N and sigma are independent. Sigma1,N10 retains
the original fresh branch. Every candidate integrates its own full start-to-zero
interval; none is an N10 prefix. Errors use first16×all7 normalized actions.

`d` is `mean_channel(sqrt(mean_token((H−mean_token(H))²)))`, evaluated in float32
from actual post-layer29 Action hidden H of shape (1,32,D). It must come from the
first denoising forward that is also used to update actions, not an extra probe.

## CPU checks (no weights, LIBERO data or GPU required)

Use Python 3.10+ with NumPy, SciPy and PyTorch (the audit used torch 2.7.1+cu128).
The rest of FastWAM's heavy dependencies are unnecessary for these CPU tests.
From the repository root:

```bash
python -m unittest discover -s experiments/libero/two_stage_sigma_n -p 'test_*.py' -v
```

Tests include independent proper-N vs shared-first equivalence, actual scheduler
endpoints/default equivalence, first-input invariance across N, threshold edges,
task-label leakage checks, and intentionally corrupted raw/analysis artifacts.
Synthetic test results are not GPU validation or scientific results.

## Phase A model API

Add this directory to PYTHONPATH when importing its script-style modules. Use
your normal configured/checkpoint-loaded `FastWAM` model and image preprocessor:

```python
from sweep_a import run_phase_a_state

result = run_phase_a_state(
    model,
    conditioning={"prompt": prompt, "input_image": image, "proprio": proprio},
    previous=previous_normalized_actions,  # float32 (32,7)
    epsilon=initial_gaussian_noise,        # float32 (32,7), same for every sigma
    compiled=True,
)
errors = result["errors"]  # float32 (11,), sigma0..1
```

Only model-ready conditioning is accepted; sampler controls cannot be overridden
through the dictionary. Persist the returned actions, initial states, actual first
inputs, timesteps/deltas and observed NFE for auditing. This API does not load a
checkpoint, preprocess observations, submit a job or run environment rollouts.
The CPU Toy fixture demonstrates its full contract.

## Portable offline analysis

Provide an NPZ (no pickled objects) containing `errors` float32[M,11], `signal`
float64[M], Unicode `keys`[M,4]=(suite,task,episode,planning_index), and integer
`s_bins`[M] (the five frozen signal-only bins). At least three tasks are required.
The original cohort's bins came from pooled signal quintiles; no action labels
were used to make these descriptive bins. Outer g boundaries are always refitted
on training tasks only.

```bash
python experiments/libero/two_stage_sigma_n/analyze.py \
  --samples /path/to/phase_a_samples.npz --output /path/to/new_analysis
```

The output directory must not exist. This writes summary, state/task CSVs, complete
outer-fold mappings, labels and Phase B pair requirements, then independently
reviews the derived analysis. Input error tables are trusted as measurements:
the portable entrypoint does **not** certify missing raw actions or GPU traces.
`review_a.verify_state` provides raw per-state verification. The historical
`analyze_a.py --raw --plan --output` route additionally verifies the original
772-state frozen run; it requires private raw artifacts and manifest, not bundled
here. Original paths/hashes stay in the local archive, not the portable interface.

## Phase B continuation API (not a GPU collector)

```python
from refinement import continue_all_n

result = continue_all_n(
    scheduler, x0, actual_first_timestep, actual_first_velocity,
    actual_first_post_layer29_hidden, denoise_rest, sigma=selected_sigma,
)
```

The caller must already have executed exactly one ordinary first Action forward.
`denoise_rest(x,t)` must preserve that first call's conditioning and KV cache.
Each N gets fresh x0 and its own first delta. Cost: 1 shared first +45 remaining
forwards =46 physical NFE; logical independent-candidate cost is 1+...+10=55.
For an online chosen N, the first forward is included in N, not N+1. Sigma0 is
handled outside this routine with zero action calls and no fabricated d value.

Still required before Phase B: actual post-layer29 hidden capture without an extra
probe; compile/baseline equivalence checks; GPU collector; feasible-label coverage;
held-out fixed-ridge s-only vs s+d comparison. If selected-sigma N10 fails,
`n_star` returns None (infeasible), never a fabricated N=10 or N=11 label.

## Evidence and boundaries

- [Phase A results](PHASE_A_RESULTS.md): rho=.442, task-demeaned=.391, 24/28
  positive tasks; held-out g action-threshold failures 32/772; zero direct reuse.
- [Protocol](PROTOCOL.md): frozen scientific definitions and planned evaluation.
- [Provenance](PROVENANCE.json): imported source hashes and original result hash.
- [Packaging validation](VALIDATION.md): 27 CPU tests passed; the real 772-state
  analysis and all 28 fold mappings exactly match the original reviewed results.
- `phase_a_results/`: compact result summary and per-task tables only.

No weights, observation/action bundles, cluster credentials, submission scripts or
automatic jobs are distributed. Those private inputs are needed to reproduce the
GPU experiment. The packaging/portable entrypoints have CPU validation only; no
new GPU equivalence claim is made by this refactor.
