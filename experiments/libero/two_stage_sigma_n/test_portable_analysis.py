from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import analyze
from io_utils import read
from test_analysis_a import example


class PortableAnalysisTests(unittest.TestCase):
    def test_cli_roundtrip_without_private_paths(self):
        errors, signal, tasks, bins = example()
        keys = np.array([['suite', str(int(task[-1])), '0', str(i)] for i, task in enumerate(tasks)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root/'samples.npz'; output = root/'analysis'
            np.savez(samples, errors=errors, signal=signal, keys=keys, s_bins=bins)
            with patch('sys.argv', ['analyze.py', '--samples', str(samples), '--output', str(output)]):
                analyze.main()
            self.assertTrue(read(output/'ANALYSIS_REVIEW.json')['passed'])
            self.assertFalse(read(output/'SUMMARY.json')['provenance']['raw_actions_independently_reviewed'])
            self.assertFalse(read(output/'PHASE_B_PAIRS.json')['gpu_authorized'])

    def test_cli_rejects_non_float32_errors(self):
        errors, signal, tasks, bins = example()
        keys = np.array([['suite', str(int(task[-1])), '0', str(i)] for i, task in enumerate(tasks)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); samples = root/'samples.npz'
            np.savez(samples, errors=errors.astype(np.float64), signal=signal, keys=keys, s_bins=bins)
            with patch('sys.argv', ['analyze.py', '--samples', str(samples), '--output', str(root/'out')]):
                with self.assertRaises(ValueError):
                    analyze.main()


if __name__ == '__main__':
    unittest.main()
