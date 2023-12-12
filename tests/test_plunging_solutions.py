import unittest
import numpy as np
from kerrgeopy.plunge import *
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

complex_orbit_parameters = np.genfromtxt(
    DATA_DIR / "plunging_orbit_parameters_complex.txt", delimiter=","
)
real_orbit_parameters = np.genfromtxt(
    DATA_DIR / "plunging_orbit_parameters_real.txt", delimiter=","
)
times = np.genfromtxt(DATA_DIR / "plunging_orbit_times.txt", delimiter=",")


class TestPlungingSolutions(unittest.TestCase):
    def test_integrals(self):
        """
        Test that the plunging radial integrals match the Mathematica output for a random set of orbits.
        """
        components = ["I_r", "I_r2", "I_r_plus", "I_r_minus"]
        for i, orbit in enumerate(complex_orbit_parameters):
            mathematica_trajectory = np.genfromtxt(
                DATA_DIR / f"plunging_integrals/trajectory{i}.txt", delimiter=","
            )

            a, E, L, Q = orbit
            I_r, I_r2, I_r_plus, I_r_minus = plunging_radial_integrals(a, E, L, Q)
            upsilon_r, upsilon_theta = plunging_mino_frequencies(a, E, L, Q)

            python_trajectory = np.transpose(
                np.apply_along_axis(
                    lambda x: np.array(
                        [
                            I_r(upsilon_r * x),
                            I_r2(upsilon_r * x),
                            I_r_plus(upsilon_r * x),
                            I_r_minus(upsilon_r * x),
                        ]
                    ),
                    0,
                    times,
                )
            )

            for j, component in enumerate(components):
                with self.subTest(
                    i=i,
                    component=component,
                    params="a = {}, E = {}, L = {}, Q = {}".format(*orbit),
                    diff=np.max(
                        np.abs(mathematica_trajectory[:, j] - python_trajectory[:, j])
                    ),
                ):
                    self.assertTrue(
                        np.allclose(
                            mathematica_trajectory[:, j], python_trajectory[:, j]
                        )
                    )

    def test_solutions(self):
        """
        Test that the plunging trajectory deltas match the Mathematica output for a random set of orbits.
        """
        components = ["t_r", "phi_r", "t_theta", "t_phi"]
        for i, orbit in enumerate(complex_orbit_parameters):
            mathematica_trajectory = np.genfromtxt(
                DATA_DIR / f"plunging_solutions/trajectory{i}.txt", delimiter=","
            )

            a, E, L, Q = orbit
            r, t_r, phi_r = plunging_radial_solutions_complex(a, E, L, Q)
            theta, t_theta, phi_theta = plunging_polar_solutions(a, E, L, Q)
            upsilon_r, upsilon_theta = plunging_mino_frequencies(a, E, L, Q)

            python_trajectory = np.transpose(
                np.apply_along_axis(
                    lambda x: np.array(
                        [
                            t_r(upsilon_r * x),
                            phi_r(upsilon_r * x),
                            t_theta(upsilon_theta * x),
                            phi_theta(upsilon_theta * x),
                        ]
                    ),
                    0,
                    times,
                )
            )

            for j, component in enumerate(components):
                with self.subTest(
                    i=i,
                    component=component,
                    params="a = {}, E = {}, L = {}, Q = {}".format(*orbit),
                    diff=np.max(
                        np.abs(mathematica_trajectory[:, j] - python_trajectory[:, j])
                    ),
                ):
                    self.assertTrue(
                        np.allclose(
                            mathematica_trajectory[:, j], python_trajectory[:, j]
                        )
                    )

    def test_orbit_complex(self):
        """
        Test the trajectory method against Mathematica output for a random set of plunging orbits
        where the radial polynomial has complex roots.
        """
        components = ["t", "r", "theta", "phi"]
        for i, orbit in enumerate(complex_orbit_parameters):
            mathematica_trajectory = np.genfromtxt(
                DATA_DIR / f"plunging_orbits_complex/trajectory{i}.txt", delimiter=","
            )

            a, E, L, Q = orbit
            plunging_orbit = PlungingOrbit(a, E, L, Q)
            # set initial phases to match the convention used in Mathematica
            t, r, theta, phi = plunging_orbit.trajectory(
                initial_phases=(0, 0, -pi / 2, 0)
            )
            python_trajectory = np.transpose(
                np.apply_along_axis(
                    lambda x: np.array([t(x), r(x), theta(x), phi(x)]), 0, times
                )
            )

            for j, component in enumerate(components):
                with self.subTest(
                    i=i,
                    component=component,
                    params="a = {}, E = {}, L = {}, Q = {}".format(*orbit),
                    diff=np.max(
                        np.abs(mathematica_trajectory[:, j] - python_trajectory[:, j])
                    ),
                ):
                    self.assertTrue(
                        np.allclose(
                            mathematica_trajectory[:, j], python_trajectory[:, j]
                        )
                    )

    def test_orbit_real(self):
        """
        Test the trajectory method against Mathematica output for a random set of plunging orbits
        where the radial polynomial has all real roots.
        """
        components = ["t", "r", "theta", "phi"]
        for i, orbit in enumerate(real_orbit_parameters):
            mathematica_trajectory = np.genfromtxt(
                DATA_DIR / f"plunging_orbits_real/trajectory{i}.txt", delimiter=","
            )

            a, E, L, Q = orbit
            plunging_orbit = PlungingOrbit(a, E, L, Q)
            # set initial phases to match the convention used in Mathematica
            t, r, theta, phi = plunging_orbit.trajectory(
                initial_phases=(0, 0, -pi / 2, 0)
            )
            python_trajectory = np.transpose(
                np.apply_along_axis(
                    lambda x: np.array([t(x), r(x), theta(x), phi(x)]), 0, times
                )
            )

            for j, component in enumerate(components):
                with self.subTest(
                    i=i,
                    component=component,
                    params="a = {}, E = {}, L = {}, Q = {}".format(*orbit),
                    diff=np.max(
                        np.abs(mathematica_trajectory[:, j] - python_trajectory[:, j])
                    ),
                ):
                    self.assertTrue(
                        np.allclose(
                            mathematica_trajectory[:, j], python_trajectory[:, j]
                        )
                    )
