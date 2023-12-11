import unittest
import numpy as np
from kerrgeopy.stable import *
from kerrgeopy.constants import stable_radial_roots, stable_polar_roots
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"


orbit_parameters = np.genfromtxt(DATA_DIR / "stable_orbit_parameters.txt", delimiter=",")
orbit_times = np.genfromtxt(DATA_DIR / "stable_orbit_times.txt", delimiter=",")


class TestStableSolutions(unittest.TestCase):
    def test_random(self):
        """
        Test trajectory deltas against Mathematica output for a random set of stable orbits.
        """
        components = ["t_r", "t_theta", "phi_r", "phi_theta"]

        for i, orbit in enumerate(orbit_parameters):
            mathematica_trajectory = np.genfromtxt(
                DATA_DIR / f"stable_solutions/trajectory{i}.txt", delimiter=","
            )

            a, p, e, x = orbit
            constants = constants_of_motion(*orbit)
            radial_roots = stable_radial_roots(a, p, e, x, constants)
            polar_roots = stable_polar_roots(a, p, e, x, constants)
            r, t_r, phi_r = radial_solutions(a, constants, radial_roots)
            theta, t_theta, phi_theta = polar_solutions(a, constants, polar_roots)

            python_trajectory = np.transpose(
                np.apply_along_axis(
                    lambda x: np.array([t_r(x), t_theta(x), phi_r(x), phi_theta(x)]),
                    0,
                    orbit_times,
                )
            )

            for j, component in enumerate(components):
                with self.subTest(
                    i=i,
                    component=component,
                    params="a = {}, p = {}, e = {}, x = {}".format(*orbit),
                    diff=np.max(
                        np.abs(mathematica_trajectory[:, j] - python_trajectory[:, j])
                    ),
                ):
                    self.assertTrue(
                        np.allclose(
                            mathematica_trajectory[:, j], python_trajectory[:, j]
                        )
                    )


class TestStableOrbit(unittest.TestCase):
    def test_extreme_kerr(self):
        """
        Test that a ValueError is raised when a = 1.
        """
        with self.assertRaises(ValueError):
            StableOrbit(1, 12, 0.5, 0.5)

    def test_polar(self):
        """
        Test that a ValueError is raised when x = 0.
        """
        with self.assertRaises(ValueError):
            StableOrbit(0.5, 12, 0.5, 0)

    def test_marginally_bound(self):
        """
        Test that a ValueError is raised when e = 1
        """
        with self.assertRaises(ValueError):
            StableOrbit(0.5, 12, 1, 0.5)

    def test_invalid_arguments(self):
        """
        Test that a ValueError is raised for invalid arguments.
        """
        with self.assertRaises(ValueError):
            StableOrbit(2, 5, 0.5, 0.5)
        with self.assertRaises(ValueError):
            StableOrbit(0.5, 5, -0.5, 0.5)

    def test_unstable(self):
        """
        Test that a ValueError is raised when p is < separatrix(a,e,x)
        """
        with self.assertRaises(ValueError):
            StableOrbit(0.5, 5, 0.5, 0.5)

    def test_random(self):
        """
        Test the trajectory method against Mathematica output for a random set of stable orbits.
        """
        components = ["t", "r", "theta", "phi"]

        for i, orbit in enumerate(orbit_parameters):
            mathematica_trajectory = np.genfromtxt(
                DATA_DIR / f"stable_orbits/trajectory{i}.txt", delimiter=","
            )
            test_orbit = StableOrbit(*orbit)
            t, r, theta, phi = test_orbit.trajectory()
            python_trajectory = np.transpose(
                np.apply_along_axis(
                    lambda x: np.array([t(x), r(x), theta(x), phi(x)]), 0, orbit_times
                )
            )

            for j, component in enumerate(components):
                with self.subTest(
                    i=i,
                    component=component,
                    params="a = {}, p = {}, e = {}, x = {}".format(*orbit),
                    diff=np.max(
                        np.abs(mathematica_trajectory[:, j] - python_trajectory[:, j])
                    ),
                ):
                    self.assertTrue(
                        np.allclose(
                            mathematica_trajectory[:, j], python_trajectory[:, j]
                        )
                    )
