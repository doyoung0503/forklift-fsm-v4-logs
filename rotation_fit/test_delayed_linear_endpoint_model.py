import unittest
import numpy as np
from rotation_fit.fit_delayed_linear_endpoint_model import fit_delayed_linear, predict, inverse


class DelayedLinearTests(unittest.TestCase):
    def test_exact_recovery(self):
        t = np.linspace(0, 3, 40)
        m = fit_delayed_linear(t, 10 * np.maximum(t - 1.2, 0))
        self.assertAlmostEqual(m['slope_deg_s'], 10)
        self.assertAlmostEqual(m['effective_delay_s'], 1.2)

    def test_global_solution_matches_dense_profile(self):
        rng = np.random.default_rng(10)
        t = rng.uniform(.1, 3, 25)
        y = np.maximum(0, 8 * (t - 1) + rng.normal(0, 2, 25))
        m = fit_delayed_linear(t, y)
        d = np.linspace(0, t.max() - .00001, 10000)
        q = np.maximum(t[:, None] - d, 0)
        k = np.sum(q * y[:, None], axis=0) / np.sum(q * q, axis=0)
        sse = np.sum((q * k - y[:, None])**2, axis=0)
        self.assertLessEqual(m['weighted_sse'], sse.min() + 1e-8)

    def test_inverse_and_support(self):
        m = fit_delayed_linear([0, 1, 2, 3], [0, 0, 10, 20])
        self.assertEqual(inverse(m, 0, 3), 0)
        self.assertAlmostEqual(float(predict(m, inverse(m, 15, 3))), 15)
        with self.assertRaises(ValueError): inverse(m, 25, 3)

    def test_zero_and_invalid_inputs(self):
        m = fit_delayed_linear([1, 2, 3], [0, 0, 0])
        self.assertEqual(m['slope_deg_s'], 0)
        with self.assertRaises(ValueError): inverse(m, 1, 3)
        with self.assertRaises(ValueError): fit_delayed_linear([1, 2], [1, np.nan])


if __name__ == '__main__':
    unittest.main()
