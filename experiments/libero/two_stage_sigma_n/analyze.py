"""Portable offline Phase A analysis; never loads weights or submits jobs."""
import argparse
from pathlib import Path
import numpy as np
from analyze_a import write_analysis
from io_utils import sha
from review_a import review_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True, type=Path,
                        help='NPZ: errors[M,11], signal[M], keys[M,4], s_bins[M]')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.samples, allow_pickle=False) as a:
        errors, signal, keys, bins = (a[k] for k in ('errors', 'signal', 'keys', 's_bins'))
        size = len(signal)
        if errors.dtype != np.float32 or errors.shape != (size, 11):
            raise ValueError('errors must be float32[M,11] on sigma=0,.1,...,1')
        if keys.shape != (size, 4) or bins.shape != (size,) or signal.shape != (size,):
            raise ValueError('Invalid sample shapes')
        if not np.isfinite(bins).all() or not ((bins >= 0) & (bins < 5) & (bins == bins.astype(int))).all():
            raise ValueError('s_bins must contain the frozen signal-only bin IDs 0..4')
        if len({tuple(row) for row in keys.tolist()}) != size:
            raise ValueError('Duplicate control-state key')
        entries = [dict(id=i, key=[str(k[0]), int(k[1]), int(k[2]), int(k[3])],
                        signal=float(signal[i]), signal_bin=int(bins[i]))
                   for i, k in enumerate(keys)]
        provenance = dict(samples_sha256=sha(args.samples), input_kind='precomputed_errors',
                          raw_actions_independently_reviewed=False)
        result = write_analysis(args.output, errors, entries, provenance)
    review_analysis(args.output)
    print('ANALYSIS_REVIEW_PASSED', result['states'], 'states;',
          result['phase_b_required_positive_pairs'], 'Phase B pairs')


if __name__ == '__main__':
    main()
