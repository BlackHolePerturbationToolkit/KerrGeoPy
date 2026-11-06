Many tests use data files containing a list of orbital parameters to test along with files containing output from the KerrGeodesics mathematica library. 
The jupyter notebook data/Unit Test Generation.ipynb contains the code that was used to generate these files.
Below is a list of data files used by each test suite:

test_constants.py:

const_parameters.txt - orbital parameters (a,p,e,x)
mathematica_const_output.txt - (E,L,Q) computed using KerrGeoConstantsOfMotion[a,p,e,x] for each orbit in const_parameters.txt
separatrix_parameters.txt - orbital parameters (a,p,e,x)
mathematica_separatrix_output.txt - output of KerrGeoSeparatrix[a,p,e,x] for each orbit in separatrix_parameters.txt

test_frequencies.py:

freq_parameters.txt - orbital parameters (a,p,e,x)
mathematica_freq_output.txt - (upsilon_r, upsilon_theta, upsilon_phi, gamma) computed using KerrGeoFrequencies[a,p,e,x] for each orbit in freq_parameters.txt

test_stable_solutions.py:

stable_orbit_parameters.txt - orbital parameters (a,p,e,x)
stable_orbit_times.txt - list of mino time values to test
stable_orbits/trajectory{i}.txt - (t, r, theta, phi) evaluated at each time from stable_orbit_times.txt for the i-th orbit defined in stable_orbit_parameters.txt
stable_solutions/trajectory{i}.txt - (t_r, t_theta, phi_r, phi_theta) evaluated at each time from stable_orbit_times.txt for the i-th orbit defined in stable_orbit_parameters.txt

test_plunging_solutions.py:

plunging_orbit_parameters_real.txt - list of orbital parameters (a,E,L,Q) for which the radial polynomial has all real roots
plunging_orbit_parameters_complex.txt - list of orbital parameters (a,E,L,Q) for which the radial polynomial has complex roots
plunging_orbit_times.txt - list of mino time values to test
plunging_integrals/trajectory{i}.txt - (I_r, I_r2, I_r_plus, I_r_minus) evaluated at each time from plunging_orbit_times.txt for the i-th orbit defined in plunging_orbit_parameters_complex.txt
plunging_solutions/trajectory{i}.txt - (t_r, phi_r, t_theta, phi_theta) evaluated at each time from plunging_orbit_times.txt for the i-th orbit defined in plunging_orbit_parameters_complex.txt
plunging_orbits_real/trajectory{i}.txt - (t, r, theta, phi) evaluated at each time from plunging_orbit_times.txt for the i-th orbit defined in plunging_orbit_parameters_real.txt
plunging_orbits_complex/trajectory{i}.txt - (t, r, theta, phi) evaluated at each time from plunging_orbit_times.txt for the i-th orbit defined in plunging_orbit_parameters_complex.txt

test_four_velocity.py:

stable_orbit_parameters.txt - orbital parameters (a,p,e,x)
stable_orbit_times.txt - list of mino time values to test
plunging_orbit_parameters_real.txt - list of orbital parameters (a,E,L,Q) for which the radial polynomial has all real roots
plunging_orbit_parameters_complex.txt - list of orbital parameters (a,E,L,Q) for which the radial polynomial has complex roots
four_velocity/trajectory{i}.txt - (u_t, u_r, u_theta, u_phi) evaluated at each time from stable_orbit_times.txt for the i-th orbit in stable_orbit_parameters.txt

test_numerical_integration.py:

This suite uses no data files. It is a self-contained, independent cross-check
that does not rely on the closed-form solutions: it integrates the second-order
geodesic equation directly with scipy.integrate.solve_ivp, and
verifies that each KerrGeoPy trajectory reproduces the numerically integrated
path starting from KerrGeoPy's own state at lambda = 0. Orbits are drawn from a
seeded (reproducible) random generator spanning the parameter space:

- stable bound orbits (a, p, e, x), including circular, equatorial and high spin;
- plunging orbits (a, E, L, Q), compared along their exterior excursion;
- null geodesics covering all four radial cases and both polar cases (ordinary
  eta > 0 and vortical eta < 0).

Each numerical integration is first checked for its own accuracy (conservation
of E, L, Q and the four-velocity norm), so a mismatch is attributable to the
closed-form solution rather than the integrator. For plunging and null orbits
the comparison is restricted to the exterior region r > r_+, where a
Boyer-Lindquist integrator is valid.