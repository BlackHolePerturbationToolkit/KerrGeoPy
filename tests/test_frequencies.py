import unittest
import numpy as np
from kerrgeopy.frequencies import *
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

class TestFrequencies(unittest.TestCase):
    def test_extreme_kerr(self):
        with self.assertRaises(ValueError): orbital_frequencies(1,12,0.5,0.5)
        with self.assertRaises(ValueError): r_frequency(1,0.5,12,0.5)
        with self.assertRaises(ValueError): theta_frequency(1,0.5,12,0.5)
        with self.assertRaises(ValueError): phi_frequency(1,0.5,12,0.5)

    def test_polar(self):
        with self.assertRaises(ValueError): orbital_frequencies(0.5,12,0.5,0)
        with self.assertRaises(ValueError): r_frequency(0.5,0.5,12,0)
        with self.assertRaises(ValueError): theta_frequency(0.5,0.5,12,0)
        with self.assertRaises(ValueError): phi_frequency(0.5,0.5,12,0)

    def test_marginally_bound(self):
        with self.assertRaises(ValueError): orbital_frequencies(0.5,12,1,0.5)
        with self.assertRaises(ValueError): r_frequency(0.5,12,1,0.5)
        with self.assertRaises(ValueError): theta_frequency(0.5,12,1,0.5)
        with self.assertRaises(ValueError): phi_frequency(0.5,12,1,0.5)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError): orbital_frequencies(2,5,0.5,0.5)
        with self.assertRaises(ValueError): r_frequency(0.5,5,-0.5,0.5)
        with self.assertRaises(ValueError): theta_frequency(0.5,5,0.5,-2)
        with self.assertRaises(ValueError): phi_frequency(0.5,5,2,0.5)

    def test_unstable(self):
        with self.assertRaises(ValueError): orbital_frequencies(0.5,5,0.5,0.5)
        with self.assertRaises(ValueError): r_frequency(0.5,5,0.5,0.5)
        with self.assertRaises(ValueError): theta_frequency(0.5,5,0.5,0.5)
        with self.assertRaises(ValueError): phi_frequency(0.5,5,0.5,0.5)

    def test_random(self):
        values = np.genfromtxt(DATA_DIR / "freq_values.txt",delimiter=",")
        mathematica_freq_output = np.genfromtxt(DATA_DIR / "mathematica_freq_output.txt")
        python_freq_output = np.apply_along_axis(lambda x: orbital_frequencies(*x),1,values)

        self.assertTrue(np.allclose(python_freq_output,mathematica_freq_output))