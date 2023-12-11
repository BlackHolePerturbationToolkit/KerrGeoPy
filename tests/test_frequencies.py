import unittest
import numpy as np
from kerrgeopy.frequencies import *
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"


class TestFrequencies(unittest.TestCase):
    def test_extreme_kerr(self):
        """
        Test that a ValueError is raised when a = 1.
        """
        with self.assertRaises(ValueError):
            mino_frequencies(1, 12, 0.5, 0.5)

    def test_polar(self):
        """
        Test that a ValueError is raised when x = 0.
        """
        with self.assertRaises(ValueError):
            mino_frequencies(0.5, 12, 0.5, 0)

    def test_marginally_bound(self):
        """
        Test that a ValueError is raised when e = 1
        """
        with self.assertRaises(ValueError):
            mino_frequencies(0.5, 12, 1, 0.5)

    def test_invalid_arguments(self):
        """
        Test that a ValueError is raised for invalid arguments.
        """
        with self.assertRaises(ValueError):
            mino_frequencies(2, 5, 0.5, 0.5)

    def test_unstable(self):
        """
        Test that a ValueError is raised when p is < separatrix(a,e,x)
        """
        with self.assertRaises(ValueError):
            mino_frequencies(0.5, 5, 0.5, 0.5)

    def test_random(self):
        """
        Test mino_frequencies method against output from the KerrGeodesics 
        Mathematica library for a random set of orbits.
        """
        parameters = np.genfromtxt(DATA_DIR / "freq_parameters.txt", delimiter=",")
        mathematica_freq_output = np.genfromtxt(
            DATA_DIR / "mathematica_freq_output.txt"
        )
        python_freq_output = np.apply_along_axis(
            lambda x: mino_frequencies(*x), 1, parameters
        )
        for i, params in enumerate(parameters):
            with self.subTest(i=i, params=params):
                self.assertTrue(
                    np.allclose(
                        abs(mathematica_freq_output[i]), abs(python_freq_output[i])
                    )
                )
