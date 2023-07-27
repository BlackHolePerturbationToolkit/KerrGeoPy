import unittest
import numpy as np
from kerrgeopy.bound_solutions import *
from kerrgeopy.frequencies_from_constants import _radial_roots, _polar_roots
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

class TestBoundSolutions(unittest.TestCase):
    def test_random(self):
        components = ["t_r","t_theta","phi_r","phi_theta"]
        orbit_values = np.genfromtxt(DATA_DIR / "orbit_values.txt", delimiter=",")
        orbit_times = np.genfromtxt(DATA_DIR / "orbit_times.txt", delimiter=",")

        for i, orbit in enumerate(orbit_values):
            mathematica_trajectory = np.genfromtxt(DATA_DIR / f"geodesics/trajectory{i}.txt", delimiter=",")
            
            a,p,e,x = orbit
            constants = constants_of_motion(*orbit)
            radial_roots = _radial_roots(a,p,e,constants)
            polar_roots = _polar_roots(a,x,constants)
            r, t_r, phi_r = radial_solutions(a,constants,radial_roots)
            theta, t_theta, phi_theta = polar_solutions(a,constants,polar_roots)

            python_trajectory = np.transpose(
                np.apply_along_axis(lambda x: np.array([t_r(x),t_theta(x),phi_r(x),phi_theta(x)]),0,orbit_times)
                )
            
            for j, component in enumerate(components):
                with self.subTest(i=i,
                                  component=component,
                                  params="a = {}, p = {}, e = {}, x = {}".format(*orbit),
                                  diff=np.max(np.abs(mathematica_trajectory[:,j]-python_trajectory[:,j]))
                                  ):
                    self.assertTrue(np.allclose(mathematica_trajectory[:,j],python_trajectory[:,j]))