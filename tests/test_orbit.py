import unittest
import numpy as np
from kerrgeopy.orbit import *
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

class TestOrbit(unittest.TestCase):
    def test_extreme_kerr(self):
        with self.assertRaises(ValueError): Orbit(1,12,0.5,0.5)

    def test_polar(self):
        with self.assertRaises(ValueError): Orbit(0.5,12,0.5,0)

    def test_marginally_bound(self):
        with self.assertRaises(ValueError): Orbit(0.5,12,1,0.5)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError): Orbit(2,5,0.5,0.5)
        with self.assertRaises(ValueError): Orbit(0.5,5,-0.5,0.5)

    def test_unstable(self):
        with self.assertRaises(ValueError): Orbit(0.5,5,0.5,0.5)

    def test_random(self):
        pass