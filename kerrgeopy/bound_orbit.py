from .constants_of_motion import *
from .frequencies import *
from .bound_solutions import *
from .units import *
from .frequencies_from_constants import _radial_roots, _polar_roots
from .orbit import Orbit
import numpy as np
import matplotlib.pyplot as plt

class BoundOrbit(Orbit):
    """
    Class representing a bound geodesic orbit in Kerr spacetime.

    :param a: dimensionless angular momentum (must satisfy 0 <= a < 1)
    :type a: double
    :param p: semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e < 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    :type x: double
    :param M: mass of the primary in solar masses, optional
    :type M: double
    :param mu: mass of the smaller body in solar masses, optional
    :type mu: double

    :ivar a: dimensionless angular momentum
    :ivar p: semi-latus rectum
    :ivar e: orbital eccentricity
    :ivar x: cosine of the orbital inclination
    :ivar M: mass of the primary in solar masses
    :ivar mu: mass of the smaller body in solar masses
    :ivar E: dimensionless energy
    :ivar L: dimensionless angular momentum
    :ivar Q: dimensionless carter constant
    :ivar upsilon_r: dimensionless radial orbital frequency in Mino time
    :ivar upsilon_theta: dimensionless polar orbital frequency in Mino time
    :ivar upsilon_phi: dimensionless azimuthal orbital frequency in Mino time
    :ivar gamma: dimensionless time dilation factor
    :ivar omega_r: dimensionless radial orbital frequency in Boyer-Lindquist time
    :ivar omega_theta: dimensionless polar orbital frequency in Boyer-Lindquist time
    :ivar omega_phi: dimensionless azimuthal orbital frequency in Boyer-Lindquist time
    """
    def __init__(self,a,p,e,x,M = None,mu=None):
        """
        Constructor method
        """
        self.a, self.p, self.e, self.x, self.M, self.mu = a, p, e, x, M, mu
        constants = constants_of_motion(a,p,e,x)

        self.E, self.L, self.Q = constants
        self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma = mino_frequencies(a,p,e,x)
        self.omega_r, self.omega_theta, self.omega_phi = observer_frequencies(a,p,e,x)

    def constants_of_motion(self, units="natural"):
        """
        Computes the energy, angular momentum, and carter constant for the orbit. Computes dimensionless constants in geometried units by default.
        M and mu must be defined in order to convert to physical units.

        :param units: units to return the constants of motion in (options are "natural", "mks" and "cgs"), defaults to "natural"
        :type units: str, optional

        :return: tuple of the form (E, L, Q)
        :rtype: tuple(double, double, double)
        """
        constants = self.E, self.L, self.Q
        if units == "natural":
            return constants
        
        if self.M is None or self.mu is None: raise ValueError("M and mu must be specified to convert constants of motion to physical units")
        
        if units == "mks":
            E, L, Q = scale_constants(constants,1,self.mu/self.M)
            return energy_in_joules(E,self.M), angular_momentum_in_mks(L,self.M), carter_constant_in_mks(Q,self.M)
        
        if units == "cgs":
            E, L, Q = scale_constants(constants,1,self.mu/self.M)
            return energy_in_ergs(E,self.M), angular_momentum_in_cgs(L,self.M), carter_constant_in_cgs(Q,self.M)
        
        raise ValueError("units must be one of 'natural', 'mks', or 'cgs'")
        
    def mino_frequencies(self, units="natural"):
        r"""
        Computes orbital frequencies in Mino time. Returns dimensionless frequencies in geometrized units by default.
        M and mu must be defined in order to convert to physical units.

        :param units: units to return the frequencies in (options are "natural", "mks" and "cgs"), defaults to "natural"
        :type units: str, optional

        :return: tuple of orbital frequencies in the form :math:`(\Upsilon_r, \Upsilon_\theta, \Upsilon_\phi, \Gamma)`
        :rtype: tuple(double, double, double, double)
        """
        upsilon_r, upsilon_theta, upsilon_phi, gamma = self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma
        if units == "natural":
            return upsilon_r, upsilon_theta, upsilon_phi, gamma
        
        if self.M is None: raise ValueError("M must be specified to convert frequencies to physical units")
        
        if units == "mks" or units == "cgs":
            return time_in_seconds(upsilon_r,self.M), time_in_seconds(upsilon_theta,self.M), time_in_seconds(upsilon_phi,self.M), time2_in_seconds2(gamma,self.M)
        
        raise ValueError("units must be one of 'natural', 'mks', or 'cgs'")
    
    def observer_frequencies(self, units="natural"):
        r"""
        Computes orbital frequencies in Boyer-Lindquist time. Returns dimensionless frequencies in geometrized units by default.
        M and mu must be defined in order to convert to physical units.

        :param units: units to return the frequencies in (options are "natural", "mks", "cgs" and "mHz"), defaults to "natural"
        :type units: str, optional
        :return: tuple of orbital frequencies in the form :math:`(\Omega_r, \Omega_\theta, \Omega_\phi)`
        :rtype: tuple(double, double, double)
        """
        upsilon_r, upsilon_theta, upsilon_phi, gamma = self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma
        if units == "natural":
            return upsilon_r/gamma, upsilon_theta/gamma, upsilon_phi/gamma
        
        if self.M is None: raise ValueError("M must be specified to convert frequencies to physical units")

        if units == "mks" or units == "cgs":
            return frequency_in_Hz(upsilon_r/gamma,self.M), frequency_in_Hz(upsilon_theta/gamma,self.M), frequency_in_Hz(upsilon_phi/gamma,self.M)
        if units == "mHz":
            return frequency_in_mHz(upsilon_r/gamma,self.M), frequency_in_mHz(upsilon_theta/gamma,self.M), frequency_in_mHz(upsilon_phi/gamma,self.M)
        
        raise ValueError("units must be one of 'natural', 'mks', 'cgs', or 'mHz'")
        
    def trajectory(self,initial_phases=(0,0,0,0),distance_units="natural",time_units="natural"):
        r"""
        Computes the time, radial, polar, and azimuthal coordinates of the orbit as a function of mino time.

        :param initial_phases: tuple of initial phases for the time, radial, polar, and azimuthal coordinates, defaults to (0,0,0,0)
        :type initial_phases: tuple, optional
        :param distance_units: units to compute the radial component of the trajectory in (options are "natural", "mks", "cgs", "au" and "km"), defaults to "natural"
        :type distance_units: str, optional
        :param time_units: units to compute the time component of the trajectory in (options are "natural", "mks", "cgs", and "days"), defaults to "natural"
        :type time_units: str, optional

        :return: tuple of functions in the form :math:`(t(\lambda), r(\lambda), \theta(\lambda), \phi(\lambda))`
        :rtype: tuple(function, function, function, function)
        """
        if distance_units != "natural" and (self.M is None or self.mu is None): raise ValueError("M and mu must be specified to convert r to physical units")
        if time_units != "natural" and (self.M is None or self.mu is None): raise ValueError("M and mu must be specified to convert t to physical units")
        
        a, p, e, x = self.a, self.p, self.e, self.x
        upsilon_r, upsilon_theta, upsilon_phi, gamma = self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma
        constants = self.constants_of_motion()
        radial_roots = _radial_roots(a,p,e,constants)
        polar_roots = _polar_roots(a,x,constants)

        r_phases, t_r, phi_r = radial_solutions(a,constants,radial_roots)
        theta_phases, t_theta, phi_theta = polar_solutions(a,constants,polar_roots)
        q_t0, q_r0, q_theta0, q_phi0 = initial_phases

        distance_conversion_func = {"natural": lambda x,M: x, "mks": distance_in_meters, "cgs": distance_in_cm, "au": distance_in_au,"km": distance_in_km}
        time_conversion_func = {"natural": lambda x,M: x, "mks": time_in_seconds, "cgs": time_in_seconds, "days": time_in_days}

        # Calculate normalization constants so that t = 0 and phi = 0 at lambda = 0 when q_t0 = 0 and q_phi0 = 0 
        C_t = t_r(q_r0)+t_theta(q_theta0)
        C_phi= phi_r(q_r0)+phi_theta(q_theta0)

        def t(mino_time):
            # equation 6
            return time_conversion_func[time_units](q_t0 + gamma*mino_time + t_r(upsilon_r*mino_time+q_r0) + t_theta(upsilon_theta*mino_time+q_theta0) - C_t, self.M)
        
        def r(mino_time):
            return distance_conversion_func[distance_units](r_phases(upsilon_r*mino_time+q_r0),self.M)
        
        def theta(mino_time):
            return theta_phases(upsilon_theta*mino_time+q_theta0)
        
        def phi(mino_time):
            # equation 6
            return q_phi0 + upsilon_phi*mino_time + phi_r(upsilon_r*mino_time+q_r0) + phi_theta(upsilon_theta*mino_time+q_theta0) - C_phi
        
        return t, r, theta, phi

    