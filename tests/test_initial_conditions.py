import unittest
import numpy as np
from kerrgeopy.initial_conditions import *
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

stable_orbit_parameters = np.genfromtxt(
    DATA_DIR / "stable_orbit_parameters.txt", delimiter=","
)
complex_plunging_orbit_parameters = np.genfromtxt(
    DATA_DIR / "plunging_orbit_parameters_complex.txt", delimiter=","
)
real_plunging_orbit_parameters = np.genfromtxt(
    DATA_DIR / "plunging_orbit_parameters_real.txt", delimiter=","
)


class TestInitialConditions(unittest.TestCase):
    def test_is_stable(self):
        pass

    def test_constants(self):
        pass

    def test_phases(self):
        pass

    def test_stable_orbit_phases(self):
        pass

    def test_plunging_orbit_phases(self):
        pass
