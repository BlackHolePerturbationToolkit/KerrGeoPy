import unittest
import numpy as np
from kerrgeopy.light import LightOrbit
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

orbit_parameters = np.genfromtxt(
    DATA_DIR / "light_orbit_parameters.txt", delimiter=","
)


class TestLightSolutions(unittest.TestCase):
    def test_constants(self):
        """
        Test that the constants of motion of light trajectories match the Mathematica output for a random set of trajectories.
        """
        for i, orbit in enumerate(orbit_parameters):
            a = orbit[0]
            x = orbit[1:5]
            p = orbit[5:9]
            eta = orbit[10]
            ell = orbit[11]

            orbit = LightOrbit(a, x, p)

            with self.subTest(
                i=i,
                params=f"{a=}, {x=}, {p=}",
                constant="eta",
                diff=np.abs(eta - orbit.eta),
            ):
                self.assertTrue(np.isclose(eta, orbit.eta))

            with self.subTest(
                i=i,
                params=f"{a=}, {x=}, {p=}",
                constant="ell",
                diff=np.abs(ell - orbit.ell),
            ):
                self.assertTrue(np.isclose(ell, orbit.ell))

    def test_lambda_x(self):
        """
        Test that the Mino time of escape of light trajectories matches the Mathematica output for a random set of trajectories.
        """
        for i, orbit in enumerate(orbit_parameters):
            a = orbit[0]
            x = orbit[1:5]
            p = orbit[5:9]
            lambda_x = orbit[9]

            orbit = LightOrbit(a, x, p)
            orbit.trajectory()

            with self.subTest(
                i=i,
                params=f"{a=}, {x=}, {p=}",
                diff=np.abs(lambda_x - orbit.lambda_x),
            ):
                self.assertTrue(np.isclose(lambda_x, orbit.lambda_x))

    def test_solutions(self):
        """
        Test that the light trajectories match the Mathematica output for a random set of trajectories.
        """
        components = ["t", "r", "theta", "phi"]
        for i, orbit in enumerate(orbit_parameters):
            mathematica_trajectory = np.genfromtxt(
                DATA_DIR / f"light_solutions/trajectory{i}.txt", delimiter=","
            )

            a = orbit[0]
            x = orbit[1:5]
            p = orbit[5:9]

            orbit = LightOrbit(a, x, p)
            t, r, theta, phi = orbit.trajectory()
            lambda_x = orbit.lambda_x

            times = np.linspace(0, lambda_x, mathematica_trajectory.shape[0] + 1)[:-1]
            python_trajectory = np.transpose(
                np.apply_along_axis(
                    lambda x: np.array(
                        [
                            t(x),
                            r(x),
                            theta(x),
                            phi(x),
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
                    params=f"{a=}, {x=}, {p=}",
                    diff=np.max(
                        np.abs(mathematica_trajectory[:, j] - python_trajectory[:, j])
                    ),
                ):
                    self.assertTrue(
                        np.allclose(
                            mathematica_trajectory[:, j], python_trajectory[:, j]
                        ),
                        python_trajectory[:, j],
                    )
