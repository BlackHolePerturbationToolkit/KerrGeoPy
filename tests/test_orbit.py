import unittest
import numpy as np
from kerrgeopy.bound_orbit import *
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

class TestOrbit(unittest.TestCase):
    def test_extreme_kerr(self):
        with self.assertRaises(ValueError): BoundOrbit(1,12,0.5,0.5)

    def test_polar(self):
        with self.assertRaises(ValueError): BoundOrbit(0.5,12,0.5,0)

    def test_marginally_bound(self):
        with self.assertRaises(ValueError): BoundOrbit(0.5,12,1,0.5)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError): BoundOrbit(2,5,0.5,0.5)
        with self.assertRaises(ValueError): BoundOrbit(0.5,5,-0.5,0.5)

    def test_unstable(self):
        with self.assertRaises(ValueError): BoundOrbit(0.5,5,0.5,0.5)

    def test_random(self):
        components = ["t","r","theta","phi"]
        orbit_values = np.genfromtxt(DATA_DIR / "orbit_values.txt", delimiter=",")
        orbit_times = np.genfromtxt(DATA_DIR / "orbit_times.txt", delimiter=",")

        for i, orbit in enumerate(orbit_values):
            mathematica_trajectory = np.genfromtxt(DATA_DIR / f"orbits/trajectory{i}.txt", delimiter=",")
            test_orbit = BoundOrbit(*orbit)
            t, r , theta, phi = test_orbit.trajectory()
            python_trajectory = np.transpose(
                np.apply_along_axis(lambda x: np.array([t(x),r(x),theta(x),phi(x)]),0,orbit_times)
                )
            
            for j, component in enumerate(components):
                with self.subTest(i=i,
                                  component=component,
                                  params="a = {}, p = {}, e = {}, x = {}".format(*orbit),
                                  diff=np.max(np.abs(mathematica_trajectory[:,j]-python_trajectory[:,j]))
                                  ):
                    self.assertTrue(np.allclose(mathematica_trajectory[:,j],python_trajectory[:,j]))