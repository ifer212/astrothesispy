"""Regression checks for package installation and numerical helpers."""

import unittest

import numpy as np
from astropy import units as u
from astropy.modeling.models import BlackBody

import astrothesispy
from astrothesispy.utiles import u_conversion, utiles


class CoreTests(unittest.TestCase):
    def test_package_version(self):
        self.assertEqual(astrothesispy.__version__, "1.0.1")

    def test_binnings_uses_y_values_and_accepts_arrays(self):
        centers, values = utiles.binnings(np.array([0, 1, 2]), [10, 20, 30], 1)
        np.testing.assert_array_equal(centers, [-1, 0, 1, 2, 3])
        self.assertEqual(values, [[], [10], [20], [30], []])

    def test_binnings_empty_and_invalid_inputs(self):
        centers, values = utiles.binnings([], [], 1)
        self.assertEqual(len(centers), 0)
        self.assertEqual(values, [])
        with self.assertRaises(ValueError):
            utiles.binnings([1], [2], 0)
        with self.assertRaises(ValueError):
            utiles.binnings([1], [], 1)

    def test_merge_ranges_empty_and_adjacent(self):
        self.assertEqual(list(utiles.merge_ranges([])), [])
        self.assertEqual(list(utiles.merge_ranges([(5, 7), (1, 3), (3, 5)])), [(1, 7)])

    def test_planck_matches_astropy_blackbody(self):
        frequencies = np.array([100, 230]) * u.GHz
        radiance = u_conversion.planck(frequencies, 20 * u.K)
        reference = BlackBody(temperature=20 * u.K)(frequencies).to(radiance.unit)
        np.testing.assert_allclose(radiance.value, reference.value, rtol=1e-12)
        self.assertTrue(radiance.unit.is_equivalent(u.W / (u.m**2 * u.Hz * u.sr)))

    def test_linear_size_with_distance_error(self):
        length, error = u_conversion.lin_size_pd(3.5, 1, 0.1)
        self.assertAlmostEqual(error / length, 0.1 / 3.5)


if __name__ == "__main__":
    unittest.main()
