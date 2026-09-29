import unittest
import numpy as np
from signals import FEATURE_SHAPE, FUTURE_SEEDS, reliability_signal, relative_l2


class SignalTests(unittest.TestCase):
    def test_direction_and_denominator(self):
        x = np.ones(FEATURE_SHAPE)
        self.assertEqual(relative_l2(2*x, x), 1.)
        self.assertEqual(relative_l2(x, 2*x), .5)

    def test_average_distances_not_features(self):
        x = np.ones(FEATURE_SHAPE)
        futures = dict(zip(FUTURE_SEEDS, [0*x, 2*x, 0*x, 2*x]))
        value, distances = reliability_signal(futures, x)
        self.assertEqual(value, 1.)
        np.testing.assert_array_equal(distances, np.ones(4))
        self.assertEqual(relative_l2(np.mean(list(futures.values()), axis=0), x), 0.)

    def test_rejects_invalid_features_and_seed_set(self):
        x = np.ones(FEATURE_SHAPE)
        with self.assertRaises(ValueError):
            reliability_signal({71: x}, x)
        for bad in (x[:-1], x * np.nan):
            with self.assertRaises(ValueError):
                relative_l2(bad, x)


if __name__ == '__main__':
    unittest.main()
