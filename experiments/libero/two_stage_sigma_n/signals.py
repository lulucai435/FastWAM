"""Frozen future-real reliability statistic; no inference or layer search.

Use previous-plan native future slot 1 and current-plan current-frame V at
zero-based layer 19. The four noise seeds are 71, 72, 73, 74. Larger s means
greater mismatch (less reliable old tail), not greater confidence.
"""
import numpy as np

FUTURE_SEEDS = (71, 72, 73, 74)
VIDEO_LAYER = 19
FEATURE_SHAPE = (98, 3072)


def relative_l2(future, current):
    """||future-current|| / max(||current||, 1e-12), in float64."""
    a = np.asarray(future, dtype=np.float64)
    b = np.asarray(current, dtype=np.float64)
    if a.shape != FEATURE_SHAPE or b.shape != FEATURE_SHAPE:
        raise ValueError('Expected layer19 V features with shape (98, 3072)')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Features must be finite')
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def reliability_signal(previous_futures, current):
    """Average four scalar distances, NOT distance of the averaged features.

    previous_futures maps the four fixed integer seeds to NumPy feature arrays.
    For BF16 torch caches explicitly convert with tensor.float().cpu().numpy().
    Temporal pairing (same episode, next E16 control state) is the caller's duty.
    """
    if set(previous_futures) != set(FUTURE_SEEDS):
        raise ValueError('Exactly the frozen four future-noise seeds are required')
    distances = np.array([relative_l2(previous_futures[seed], current)
                          for seed in FUTURE_SEEDS], dtype=np.float64)
    return float(distances.mean()), distances
