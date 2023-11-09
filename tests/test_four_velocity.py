import unittest
import numpy as np
from kerrgeopy.stable import *
from kerrgeopy.plunge import *
from pathlib import Path

THIS_DIR = Path(__file__).parent

DATA_DIR = THIS_DIR.parent / "tests/data"

stable_orbit_values = np.genfromtxt(DATA_DIR / "stable_orbit_parameters.txt", delimiter=",")
complex_plunging_orbit_values = np.genfromtxt(DATA_DIR / "plunging_orbit_parameters_complex.txt", delimiter=",")
real_plunging_orbit_values = np.genfromtxt(DATA_DIR / "plunging_orbit_parameters_real.txt", delimiter=",")
plunging_orbit_values = np.concatenate((complex_plunging_orbit_values,real_plunging_orbit_values),axis=0)
times = np.genfromtxt(DATA_DIR / "stable_orbit_times.txt", delimiter=",")

class TestFourVelocity(unittest.TestCase):
    def test_stable_orbit_four_velocity(self):
        components = ["ut","ur","utheta","uphi"]

        for i, orbit in enumerate(stable_orbit_values):
            mathematica_trajectory = np.genfromtxt(DATA_DIR / f"four_velocity/trajectory{i}.txt", delimiter=",")
            
            a,p,e,x = orbit
            stable_orbit = StableOrbit(a,p,e,x)
            u_t, u_r, u_theta, u_phi = stable_orbit.four_velocity()

            python_trajectory = np.transpose(
                np.apply_along_axis(lambda x: np.array([u_t(x),u_r(x),u_theta(x),u_phi(x)]),0,times)
                )
            
            for j, component in enumerate(components):
                with self.subTest(i=i,
                                  component=component,
                                  params="a = {}, p = {}, e = {}, x = {}".format(*orbit),
                                  diff=np.max(np.abs(mathematica_trajectory[:,j]-python_trajectory[:,j]))
                                  ):
                    self.assertTrue(np.allclose(mathematica_trajectory[:,j],python_trajectory[:,j],atol=1e-6))
    
    def test_norm(self):
        times = np.linspace(0,10,10)
        # stable orbits
        for i, orbit in enumerate(stable_orbit_values):
            a,p,e,x = orbit
            stable_orbit = StableOrbit(a,p,e,x)
            norm = stable_orbit._four_velocity_norm()
            for j, time in enumerate(times):
                with self.subTest(i=i,
                                params="a = {}, p = {}, e = {}, x = {}".format(*orbit),
                                norm = norm(time)
                                ):
                    self.assertTrue(abs(norm(time)+1)<1e-8)

        # plunging orbits
        for i, orbit in enumerate(plunging_orbit_values):
            a,E,L,Q = orbit
            plunging_orbit = PlungingOrbit(a,E,L,Q)
            norm = plunging_orbit._four_velocity_norm()
            for j, time in enumerate(times):
                with self.subTest(i=i,
                                params="a = {}, E = {}, L = {}, Q = {}".format(*orbit),
                                norm = norm(time)
                                ):
                    self.assertTrue(abs(norm(time)+1)<1e-1)

    def test_using_numerical_differentiation(self):
        components = ["ut","ur","utheta","uphi"]

        #stable orbits
        for i, orbit in enumerate(stable_orbit_values):
            a,p,e,x = orbit
            stable_orbit = StableOrbit(a,p,e,x)
            u_t, u_r, u_theta, u_phi = stable_orbit.four_velocity()
            delta_t, delta_r, delta_theta, delta_phi = stable_orbit.numerical_four_velocity()

            analytic_four_velocity = np.transpose(
                np.apply_along_axis(lambda x: np.array([u_t(x),u_r(x),u_theta(x),u_phi(x)]),0,times)
                )
            
            numerical_four_velocity = np.transpose(
                np.apply_along_axis(lambda x: np.array([delta_t(x),delta_r(x),delta_theta(x),delta_phi(x)]),0,times)
                )
            
            for j, component in enumerate(components):
                with self.subTest(i=i,
                                  component=component,
                                  params="a = {}, p = {}, e = {}, x = {}".format(*orbit),
                                  diff=np.max(np.abs(analytic_four_velocity[:,j]-numerical_four_velocity[:,j]))
                                  ):
                    self.assertTrue(np.allclose(analytic_four_velocity[:,j],numerical_four_velocity[:,j],atol=1e-3))

        #plunging orbits

        # for i, orbit in enumerate(plunging_orbit_values):
        #     a,E,L,Q = orbit
        #     plunging_orbit = PlungingOrbit(a,E,L,Q)
        #     u_t, u_r, u_theta, u_phi = plunging_orbit.four_velocity()
        #     delta_t, delta_r, delta_theta, delta_phi = plunging_orbit.numerical_four_velocity()

        #     analytic_four_velocity = np.transpose(
        #         np.apply_along_axis(lambda x: np.array([u_t(x),u_r(x),u_theta(x),u_phi(x)]),0,times)
        #         )
            
        #     numerical_four_velocity = np.transpose(
        #         np.apply_along_axis(lambda x: np.array([delta_t(x),delta_r(x),delta_theta(x),delta_phi(x)]),0,times)
        #         )
            
        #     for j, component in enumerate(components):
        #         with self.subTest(i=i,
        #                           component=component,
        #                           params="a = {}, E = {}, L = {}, Q = {}".format(*orbit),
        #                           diff=np.max(np.abs(analytic_four_velocity[:,j]-numerical_four_velocity[:,j]))
        #                           ):
        #             self.assertTrue(np.allclose(analytic_four_velocity[:,j],numerical_four_velocity[:,j],rtol=1e-3,atol=1e-1))
    