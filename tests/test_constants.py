import unittest
import numpy as np
from kerrgeopy.constants_of_motion import *
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

class TestConstants(unittest.TestCase):
    def test_extreme_kerr(self):
        with self.assertRaises(ValueError): constants_of_motion(1,12,0.5,0.5)
        with self.assertRaises(ValueError): energy(1,0.5,12,0.5)
        with self.assertRaises(ValueError): angular_momentum(1,0.5,12,0.5)
        with self.assertRaises(ValueError): carter_constant(1,0.5,12,0.5)

    def test_unstable(self):
        with self.assertRaises(ValueError): constants_of_motion(0.5,5,0.5,0.5)
        with self.assertRaises(ValueError): energy(0.5,5,0.5,0.5)
        with self.assertRaises(ValueError): angular_momentum(0.5,5,0.5,0.5)
        with self.assertRaises(ValueError): carter_constant(0.5,5,0.5,0.5)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError): constants_of_motion(2,5,0.5,0.5)
        with self.assertRaises(ValueError): energy(0.5,5,2,0.5)
        with self.assertRaises(ValueError): angular_momentum(0.5,5,0.5,2)
        with self.assertRaises(ValueError): carter_constant(-0.5,5,-0.5,0.5)

    def test_constants_random(self):
        values = np.genfromtxt(DATA_DIR / "const_values.txt",delimiter=",")
        mathematica_const_output = np.genfromtxt(DATA_DIR / "mathematica_const_output.txt")
        python_const_output = np.apply_along_axis(lambda x: constants_of_motion(*x),1,values)

        for i, params in enumerate(values):
            with self.subTest(i=i,params=params):
                self.assertTrue(np.allclose(mathematica_const_output[i],python_const_output[i]))


class TestSeparatrix(unittest.TestCase):
    def test_extreme_kerr(self):
        with self.assertRaises(ValueError): separatrix(1,0.5,0.5)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError): separatrix(2,-0.5,-0.5)

    def test_separatrix_random(self):
        sep_values = np.genfromtxt(DATA_DIR / "sep_values.txt",delimiter=",")
        mathematica_separatrix_output = np.genfromtxt(DATA_DIR / "mathematica_sep_output.txt")
        python_separatrix_output = np.apply_along_axis(lambda x: separatrix(*x),1,sep_values)
        for i, params in enumerate(sep_values):
            with self.subTest(i=i,params=params):
                self.assertTrue(np.allclose(python_separatrix_output[i],mathematica_separatrix_output[i]))