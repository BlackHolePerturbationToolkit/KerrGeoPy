Many tests use data files containing a list of orbital parameters to test along with files containing output from the KerrGeodesics mathematica library.
Below is a list of data files used by each test suite:

test_constants.py:

const_values.txt - orbital parameters (a,p,e,x)
mathematica_const_output.txt - (E,L,Q) computed using KerrGeoConstantsOfMotion[a,p,e,x] for each orbit in const_values.txt
separatrix_values.txt - orbital parameters (a,p,e,x)
mathematica_separatrix_output.txt - output of KerrGeoSeparatrix[a,p,e,x] for each orbit in separatrix_values.txt

test_frequencies.py:

freq_values.txt - orbital parameters (a,p,e,x)
mathematica_freq_output.txt - (upsilon_r, upsilon_theta, upsilon_phi, gamma) computed using KerrGeoFrequencies[a,p,e,x] for each orbit in freq_values.txt

test_stable_solutions.py:

stable_orbit_values.txt - orbital parameters (a,p,e,x)
stable_orbit_times.txt - list of mino time values to test
stable_orbits/trajectory{i}.txt - (t, r, theta, phi) evaluated at each time from stable_orbit_times.txt for the i-th orbit defined in stable_orbit_values.txt
stable_solutions/trajectory{i}.txt - (t_r, t_theta, phi_r, phi_theta) evaluated at each time from stable_orbit_times.txt for the i-th orbit defined in stable_orbit_values.txt

test_plunging_solutions.py:

plunging_orbit_parameters_real.txt - list of orbital parameters (a,E,L,Q) for which the radial polynomial has all real roots
plunging_orbit_parameters_complex.txt - list of orbital parameters (a,E,L,Q) for which the radial polynomial has complex roots
plunging_orbit_times.txt - list of mino time values to test
plunging_integrals/trajectory{i}.txt - (I_r, I_r2, I_r_plus, I_r_minus) evaluated at each time from plunging_orbit_times.txt for the i-th orbit defined in plunging_orbit_parameters_complex.txt
plunging_solutions/trajectory{i}.txt - (t_r, phi_r, t_theta, phi_theta) evaluated at each time from plunging_orbit_times.txt for the i-th orbit defined in plunging_orbit_parameters_complex.txt
plunging_orbits_real/trajectory{i}.txt - (t, r, theta, phi) evaluated at each time from plunging_orbit_times.txt for the i-th orbit defined in plunging_orbit_parameters_real.txt
plunging_orbits_complex/trajectory{i}.txt - (t, r, theta, phi) evaluated at each time from plunging_orbit_times.txt for the i-th orbit defined in plunging_orbit_parameters_complex.txt