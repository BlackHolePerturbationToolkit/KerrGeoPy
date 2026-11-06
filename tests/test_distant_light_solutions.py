import unittest
import numpy as np
from kerrgeopy.light import DistantLightOrbit
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

orbit_parameters = np.genfromtxt(
    DATA_DIR / "distant_light_orbit_parameters.txt", delimiter=","
)


class TestDistantLightSolutions(unittest.TestCase):
    def test_constants(self):
        """
        Test that the constants of motion of distant light trajectories match the Mathematica output for a random set of trajectories.
        """
        for i, orbit in enumerate(orbit_parameters):
            a, theta0, alpha, beta, _, _, eta, ell = orbit

            orbit = DistantLightOrbit(a, theta0, 0, alpha, beta)

            with self.subTest(
                i=i,
                params=f"{a=}, {theta0=}, {alpha=}, {beta=}",
                constant="eta",
                diff=np.abs(eta - orbit.eta),
            ):
                self.assertTrue(np.isclose(eta, orbit.eta))

            with self.subTest(
                i=i,
                params=f"{a=}, {theta0=}, {alpha=}, {beta=}",
                constant="ell",
                diff=np.abs(ell - orbit.ell),
            ):
                self.assertTrue(np.isclose(ell, orbit.ell))

    def test_lambda_x(self):
        """
        Test that the Mino time of escape of distant light trajectories matches the Mathematica output for a random set of trajectories.
        """
        for i, orbit in enumerate(orbit_parameters):
            a, theta0, alpha, beta, lambda_x, _, _, _ = orbit

            orbit = DistantLightOrbit(a, theta0, 0, alpha, beta)
            orbit.trajectory()

            with self.subTest(
                i=i,
                params=f"{a=}, {theta0=}, {alpha=}, {beta=}",
                diff=np.abs(lambda_x - orbit.lambda_x),
            ):
                self.assertTrue(np.isclose(lambda_x, orbit.lambda_x))

    def test_lambda_shell(self):
        """
        Test that the shell intersection Mino time of distant light trajectories matches the Mathematica output for a random set of trajectories.
        """
        for i, orbit in enumerate(orbit_parameters):
            a, theta0, alpha, beta, _, lambda_shell, _, _ = orbit

            orbit = DistantLightOrbit(a, theta0, 0, alpha, beta)
            orbit.trajectory()

            with self.subTest(
                i=i,
                params=f"{a=}, {theta0=}, {alpha=}, {beta=}",
                diff=np.abs(lambda_shell - orbit.lambda_shell),
            ):
                self.assertTrue(np.isclose(lambda_shell, orbit.lambda_shell) or (np.isnan(orbit.lambda_shell) and lambda_shell < 0))

    def test_solutions(self):
        """
        Test that the distant light trajectories match the Mathematica output for a random set of trajectories.
        """
        components = ["delta_v", "r", "theta", "phi"]
        for i, orbit in enumerate(orbit_parameters):
            mathematica_trajectory = np.genfromtxt(
                DATA_DIR / f"distant_light_solutions/trajectory{i}.txt", delimiter=","
            )

            a, theta0, alpha, beta, _, _, _, _ = orbit

            orbit = DistantLightOrbit(a, theta0, 0, alpha, beta)
            orbit.trajectory()
            delta_v, r, theta, phi = orbit.trajectory()
            lambda_x = orbit.lambda_x

            times = np.linspace(0, lambda_x, mathematica_trajectory.shape[0] + 2)[1:-1]
            python_trajectory = np.transpose(
                np.apply_along_axis(
                    lambda x: np.array(
                        [
                            delta_v(x),
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
                    params=f"{a=}, {theta0=}, {alpha=}, {beta=}",
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
