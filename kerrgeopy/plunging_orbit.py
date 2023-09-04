"""
Module containing the PlungingOrbit class
"""
from .plunging_solutions import *
from .plunging_solutions import _plunging_radial_roots
from .bound_solutions import *
from .orbit import Orbit
import matplotlib.pyplot as plt

class PlungingOrbit(Orbit):
    """
    Class representing a plunging orbit in Kerr spacetime.

    :param a: dimensionaless spin parameter
    :type a: double
    :param E: dimensionless energy
    :type E: double
    :param L: dimensionless angular momentum
    :type L: double
    :param Q: dimensionless Carter constant
    :type Q: double
    :param M: mass of the primary in solar masses, optional
    :type M: double
    :param mu: mass of the smaller body in solar masses, optional
    :type mu: double

    :ivar a: dimensionaless spin parameter
    :ivar E: dimensionless energy
    :ivar L: dimensionless angular momentum
    :ivar Q: dimensionless Carter constant
    :ivar upsilon_r: radial Mino frequency
    :ivar upsilon_theta: polar Mino frequency
    """
    def __init__(self,a,E,L,Q,M=None,mu=None):
        self.a, self.E, self.L, self.Q, self.M, self.mu = a, E, L, Q, M, mu
        self.upsilon_r, self.upsilon_theta = plunging_mino_frequencies(a,E,L,Q)

    def trajectory(self,initial_phases=(0,0,0,0),distance_units="natural",time_units="natural"):
        r"""
        Computes the components of the trajectory as a function of mino time

        :param initial_phases: tuple of initial phases, defaults to (0,0,0,0)
        :type initial_phases: tuple, optional
        :param distance_units: units to compute the radial component of the trajectory in (options are "natural", "mks", "cgs", "au" and "km"), defaults to "natural"
        :type distance_units: str, optional
        :param time_units: units to compute the time component of the trajectory in (options are "natural", "mks", "cgs", and "days"), defaults to "natural"
        :type time_units: str, optional
        
        :return: tuple of functions :math:`t(\lambda), r(\lambda), \theta(\lambda), \phi(\lambda)`
        :rtype: tuple(function,function,function,function)
        """
        
        if distance_units != "natural" and (self.M is None or self.mu is None): raise ValueError("M and mu must be specified to convert r to physical units")
        if time_units != "natural" and (self.M is None or self.mu is None): raise ValueError("M and mu must be specified to convert t to physical units")

        a, E, L, Q = self.a, self.E, self.L, self.Q
        radial_roots = _plunging_radial_roots(a,E,L,Q)

        if np.iscomplex(radial_roots[3]):
            return self._complex_trajectory(initial_phases,distance_units,time_units)
        else:
            return self._real_trajectory(initial_phases,distance_units,time_units)
    
    def _complex_trajectory(self,initial_phases=(0,0,0,0), distance_units="natural",time_units="natural"):
        """
        Computes the components of the trajectory in the case of two complex and two real radial roots

        :param initial_phases: tuple of initial phases, defaults to (0,0,0,0)
        :type initial_phases: tuple, optional

        :rtype: tuple(function,function,function,function)
        """
        a, E, L, Q = self.a, self.E, self.L, self.Q
        upsilon_r, upsilon_theta = self.upsilon_r, self.upsilon_theta
        r_phases, t_r, phi_r = plunging_radial_solutions_complex(a,E,L,Q)
        theta_phases, t_theta, phi_theta = plunging_polar_solutions(a,E,L,Q)
        q_t0, q_r0, q_theta0, q_phi0 = initial_phases

        distance_conversion_func = {"natural": lambda x,M: x, "mks": distance_in_meters, "cgs": distance_in_cm, "au": distance_in_au,"km": distance_in_km}
        time_conversion_func = {"natural": lambda x,M: x, "mks": time_in_seconds, "cgs": time_in_seconds, "days": time_in_days}

        # Calculate normalization constants so that t = 0 and phi = 0 at lambda = 0 when q_t0 = 0 and q_phi0 = 0 
        C_t = t_r(q_r0)+t_theta(q_theta0)
        C_phi= phi_r(q_r0)+phi_theta(q_theta0)

        def t(mino_time):
            return time_conversion_func[time_units](t_r(upsilon_r*mino_time+q_r0) + t_theta(upsilon_theta*mino_time+q_theta0) + a*L*mino_time - C_t + q_t0, self.M)
        
        def r(mino_time):
            return distance_conversion_func[distance_units](r_phases(upsilon_r*mino_time+q_r0), self.M)
        
        def theta(mino_time):
            return theta_phases(upsilon_theta*mino_time+q_theta0)
        
        def phi(mino_time):
            return phi_r(upsilon_r*mino_time+q_r0) + phi_theta(upsilon_theta*mino_time+q_theta0) - a*E*mino_time - C_phi + q_phi0
        
        return t, r, theta, phi
    
    def _real_trajectory(self,initial_phases=(0,0,0,0),distance_units="natural",time_units="natural"):
        """
        Computes the components of the trajectory in the case of four real radial roots

        :param initial_phases: tuple of initial phases, defaults to (0,0,0,0)
        :type initial_phases: tuple, optional

        :rtype: tuple(function,function,function,function)
        """
        a, E, L, Q = self.a, self.E, self.L, self.Q
        constants = (E,L,Q)
        Z = Polynomial([Q,-(Q+a**2*(1-E**2)+L**2),a**2*(1-E**2)])
        radial_roots = _plunging_radial_roots(a,E,L,Q)
        polar_roots = Z.roots()

        upsilon_r, upsilon_theta, upsilon_phi, gamma = mino_frequencies_from_constants(a,constants,radial_roots,polar_roots)
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
            return distance_conversion_func[distance_units](r_phases(upsilon_r*mino_time+q_r0), self.M)
        
        def theta(mino_time):
            return theta_phases(upsilon_theta*mino_time+q_theta0)
        
        def phi(mino_time):
            # equation 6
            return q_phi0 + upsilon_phi*mino_time + phi_r(upsilon_r*mino_time+q_r0) + phi_theta(upsilon_theta*mino_time+q_theta0) - C_phi
        
        return t, r, theta, phi