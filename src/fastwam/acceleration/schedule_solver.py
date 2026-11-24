"""Calibration-only joint allocation over completely enumerated layer schedules.

True in a returned mask means skip/reuse. Each layer chooses one admissible
refresh pattern. HiGHS solves the resulting finite mixed-integer linear program;
its optimality certificate concerns the stated similarity objective and numerical
tolerances, never action error or success rate. No evaluation output is read.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import warnings

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


@dataclass(frozen=True)
class Pattern:
    bits: int
    updates: tuple[int, ...]
    count: int
    score: float


def normalize_step_weights(weights, steps):
    result = np.ones(steps, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if result.shape != (steps,) or not np.isfinite(result).all() or not (result > 0).all():
        raise ValueError('Step weights must be finite, strictly positive, and length steps')
    return result / result.mean()


def schedule_score(similarity, updates, weights=None):
    sim = np.asarray(similarity, dtype=np.float64)
    steps = len(sim)
    updates = tuple(map(int, updates))
    if sim.shape != (steps, steps) or not np.isfinite(sim).all():
        raise ValueError('Expected a finite square similarity matrix')
    if not updates or updates[0] != 0 or tuple(sorted(set(updates))) != updates or updates[-1] >= steps:
        raise ValueError('Refresh steps must be sorted, distinct, in range, and start at zero')
    weight = normalize_step_weights(weights, steps)
    return sum(float(np.dot(sim[a, a:b], weight[a:b]))
               for a, b in zip(updates, updates[1:] + (steps,)))


def mask_score(similarity, mask, weights=None):
    sim, mask = np.asarray(similarity, dtype=np.float64), np.asarray(mask)
    layers, steps, last = sim.shape
    if steps != last or mask.shape != (steps, layers) or mask.dtype != np.bool_:
        raise ValueError('Expected similarity[layer, anchor, target] and bool mask[step, layer]')
    return sum(schedule_score(sim[layer], np.flatnonzero(~mask[:, layer]), weights)
               for layer in range(layers))


def enumerate_patterns(similarity, mandatory=(0,), count=None, weights=None):
    """Enumerate all allowed patterns in ascending bit-mask order, without pruning."""
    sim = np.asarray(similarity, dtype=np.float64)
    steps = len(sim)
    required = frozenset(map(int, mandatory))
    if 0 not in required or any(t < 0 or t >= steps for t in required):
        raise ValueError('Mandatory refresh steps must include zero and be in range')
    if count is not None and (int(count) != count or not len(required) <= count <= steps):
        raise ValueError('Layer count is incompatible with mandatory refreshes')
    free = [t for t in range(steps) if t not in required]
    patterns = []
    sizes = range(len(free) + 1) if count is None else (int(count) - len(required),)
    for size in sizes:
        for subset in itertools.combinations(free, size):
            updates = tuple(sorted(required | set(subset)))
            patterns.append(Pattern(sum(1 << t for t in updates), updates, len(updates),
                                    schedule_score(sim, updates, weights)))
    return sorted(patterns, key=lambda pattern: pattern.bits)


def _highs_version():
    try:
        from scipy.optimize._highspy import _core
        return '.'.join(str(getattr(_core, f'HIGHS_VERSION_{part}')) for part in ('MAJOR', 'MINOR', 'PATCH'))
    except (ImportError, AttributeError):
        return 'unknown'


def solve_schedule(similarity, total_calls, *, layer_counts=None, step_weights=None,
                   prefix_layers=3, require_step_coverage=True, terminal_anchor=False,
                   time_limit_seconds=120., objective_scale=1e6, require_optimal=True):
    """Solve fixed-quota or free-quota schedules with the same joint constraints.

    Every layer refreshes at step0; the first ``prefix_layers`` also refresh at
    step1. With ``terminal_anchor`` every layer refreshes at the final step.
    Supplying ``layer_counts`` fixes each layer quota; otherwise only the exact
    total call count is fixed. Coverage prevents a wholly cached denoising step.

    Pattern order, random seed and the single HiGHS thread are fixed. Ties have
    deterministic solver selection for this pinned build, not a claimed portable
    lexicographic ordering. No objective perturbation is used to break ties.
    """
    sim = np.asarray(similarity, dtype=np.float64)
    if sim.ndim != 3 or sim.shape[1] != sim.shape[2] or not np.isfinite(sim).all():
        raise ValueError('Similarity must be finite with shape [layers, steps, steps]')
    layers, steps, _ = sim.shape
    if layers < 1 or steps < 1 or not 0 <= prefix_layers <= layers:
        raise ValueError('Invalid dimensions or prefix layer count')
    if prefix_layers and steps < 2:
        raise ValueError('A step1 prefix requires at least two denoising steps')
    if int(total_calls) != total_calls or not layers <= total_calls <= layers * steps:
        raise ValueError('Invalid total call count')
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError('A finite positive time limit is required')
    if not np.isfinite(objective_scale) or objective_scale <= 0:
        raise ValueError('Objective scale must be finite and positive')
    weights = normalize_step_weights(step_weights, steps)
    counts = None if layer_counts is None else np.asarray(layer_counts)
    if counts is not None:
        if counts.shape != (layers,) or not np.isfinite(counts).all() or not np.equal(counts, counts.astype(int)).all():
            raise ValueError('Layer counts must be an integer vector of length layers')
        if int(counts.sum()) != total_calls:
            raise ValueError('Fixed layer counts must sum to total_calls')
    patterns_by_layer = []
    for layer in range(layers):
        mandatory = {0}
        if layer < prefix_layers:
            mandatory.add(1)
        if terminal_anchor:
            mandatory.add(steps - 1)
        patterns_by_layer.append(enumerate_patterns(sim[layer], mandatory,
            None if counts is None else int(counts[layer]), weights))
    min_calls = sum(min(p.count for p in patterns) for patterns in patterns_by_layer)
    max_calls = sum(max(p.count for p in patterns) for patterns in patterns_by_layer)
    if not min_calls <= total_calls <= max_calls:
        raise ValueError('Total calls incompatible with mandatory refreshes')
    offsets = np.cumsum([0] + [len(patterns) for patterns in patterns_by_layer])
    variables = int(offsets[-1])
    row, col, data, objective = [], [], [], []
    layer_best = [max(p.score for p in patterns) for patterns in patterns_by_layer]
    for layer, patterns in enumerate(patterns_by_layer):
        for local, pattern in enumerate(patterns):
            index = int(offsets[layer]) + local
            row.extend((layer, layers)); col.extend((index, index)); data.extend((1., float(pattern.count)))
            if require_step_coverage:
                for step in pattern.updates:
                    row.append(layers + 1 + step); col.append(index); data.append(1.)
            # Subtract a layer-constant offset and scale; neither changes the
            # maximizing schedule, unlike rounding or epsilon tie perturbations.
            objective.append((layer_best[layer] - pattern.score) * objective_scale)
    constraint_count = layers + 1 + (steps if require_step_coverage else 0)
    matrix = csc_matrix((np.asarray(data), (np.asarray(row), np.asarray(col))),
                        shape=(constraint_count, variables))
    lower = np.concatenate((np.ones(layers), [total_calls], np.ones(steps) if require_step_coverage else []))
    upper = np.concatenate((np.ones(layers), [total_calls], np.full(steps, layers) if require_step_coverage else []))
    options = dict(time_limit=float(time_limit_seconds), mip_rel_gap=0., presolve=True,
                   threads=1, random_seed=0, parallel=False)
    with warnings.catch_warnings():
        # SciPy documents that unknown options are passed verbatim to HiGHS.
        # These three native options pin solver execution, and are recorded below.
        warnings.filterwarnings('ignore', message='Unrecognized options detected:', category=RuntimeWarning)
        result = milp(np.asarray(objective), integrality=np.ones(variables, dtype=np.uint8),
                      bounds=Bounds(np.zeros(variables), np.ones(variables)),
                      constraints=LinearConstraint(matrix, lower, upper), options=options)
    report = dict(solver='scipy.optimize.milp/HiGHS', scipy_version=scipy.__version__,
        highs_version=_highs_version(), solver_options=options, status=int(result.status),
        success=bool(result.success), message=str(result.message), variables=variables,
        constraints=constraint_count, patterns_per_layer=[len(p) for p in patterns_by_layer],
        quota_mode='fixed' if counts is not None else 'free', total_calls=int(total_calls),
        layer_counts=None if counts is None else counts.astype(int).tolist(),
        step_weights=weights.tolist(), prefix_layers=int(prefix_layers),
        require_step_coverage=bool(require_step_coverage), terminal_anchor=bool(terminal_anchor),
        objective_scale=float(objective_scale), objective='Sum of target-weighted cached-anchor similarities over all layers and targets',
        optimality_scope='Finite enumerated proxy objective, subject to stated constraints and HiGHS numerical tolerances; not action quality or SR',
        tie_rule='Ascending pattern bitset enumeration; deterministic single-thread HiGHS seed0; no objective perturbation',
        mip_node_count=None if getattr(result, 'mip_node_count', None) is None else int(result.mip_node_count),
        mip_gap=None if getattr(result, 'mip_gap', None) is None else float(result.mip_gap),
        loss_objective=None if getattr(result, 'fun', None) is None else float(result.fun),
        loss_dual_bound=None if getattr(result, 'mip_dual_bound', None) is None else float(result.mip_dual_bound),
        optimal_within_solver_tolerances=bool(result.success and result.status == 0))
    if result.x is None:
        report.update(mask=None, weighted_score=None, unweighted_score=None,
                      max_constraint_violation=None, max_integrality_violation=None,
                      max_feasibility_residual=None)
    else:
        values = np.asarray(result.x)
        activity = np.asarray(matrix @ values)
        row_violation = float(max(0., np.max(lower - activity), np.max(activity - upper)))
        bound_violation = float(max(0., -values.min(), values.max() - 1.))
        integrality_violation = float(np.max(np.abs(values - np.rint(values))))
        report.update(max_constraint_violation=row_violation,
                      max_bound_violation=bound_violation,
                      max_integrality_violation=integrality_violation,
                      max_feasibility_residual=max(row_violation, bound_violation, integrality_violation))
        if report['max_feasibility_residual'] > 1e-5:
            raise RuntimeError('Solver returned a nonintegral candidate')
        mask = np.ones((steps, layers), dtype=bool)
        selected = []
        for layer, patterns in enumerate(patterns_by_layer):
            indices = np.flatnonzero(values[offsets[layer]:offsets[layer + 1]] > .5)
            if len(indices) != 1:
                raise RuntimeError('Solver candidate did not choose one pattern per layer')
            pattern = patterns[int(indices[0])]
            selected.append(pattern.bits)
            mask[list(pattern.updates), layer] = False
        assert int((~mask).sum()) == total_calls and not mask[0].any()
        assert prefix_layers == 0 or not mask[1, :prefix_layers].any()
        assert not require_step_coverage or not mask.all(axis=1).any()
        assert not terminal_anchor or not mask[-1].any()
        assert counts is None or np.array_equal((~mask).sum(axis=0), counts)
        score = mask_score(sim, mask, weights)
        bound = None if report['loss_dual_bound'] is None else float(sum(layer_best) - report['loss_dual_bound'] / objective_scale)
        report.update(mask=mask.tolist(), selected_pattern_bits=selected,
                      calls_per_layer=(~mask).sum(axis=0).tolist(), calls_per_step=(~mask).sum(axis=1).tolist(),
                      weighted_score=score, unweighted_score=mask_score(sim, mask),
                      proxy_upper_bound=bound, proxy_absolute_gap=None if bound is None else max(0., bound - score))
    if require_optimal and not report['optimal_within_solver_tolerances']:
        raise RuntimeError(f'No certified optimum: status={report["status"]}, message={report["message"]}, gap={report["mip_gap"]}')
    return report
