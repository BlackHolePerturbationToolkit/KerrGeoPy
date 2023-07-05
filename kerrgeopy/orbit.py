from .constants import *
from .frequencies import *
from .geodesics import *

class Orbit:
    def __init__(self,a,p,e,x,M = None,mu=None):
        """
        Initializes an orbit with the given orbital parameters

        :param a: dimensionless angular momentum
        :type a: double
        :param p: semi-latus rectum
        :type p: double
        :param e: orbital eccentricity
        :type e: double
        :param x: cosine of the orbital inclination
        :type x: double
        :param mu: mass ratio
        :type mu: double
        """
        self.a, self.p, self.e, self.x, self.M, self.mu = a, p, e, x, M, mu
        self.E, self.L, self.Q = constants_of_motion(a,p,e,x)
        self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma = orbital_frequencies(a,p,e,x)
    

    def trajectory(self,initial_phases=(0,0,0,0)):
        a, p, e, x = self.a, self.p, self.e, self.x
        E, L, Q = self.E, self.L, self.Q
        upsilon_r, upsilon_theta, upsilon_phi, gamma = self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma
        r_phase, t_r, phi_r = radial_solutions(a,p,e,x)
        theta_phase, t_theta, phi_theta = polar_solutions(a,p,e,x)
        q_t0, q_r0, q_theta0, q_phi0 = initial_phases

        C_t = t_r(q_r0)+t_theta(q_theta0)
        C_phi= phi_r(q_r0)+phi_theta(q_theta0)

        def t(mino_time):
            return q_t0 + gamma*mino_time + t_r(upsilon_r*mino_time+q_r0) + t_theta(upsilon_theta*mino_time+q_theta0) - C_t
        
        def r(mino_time):
            return r_phase(upsilon_r*mino_time+q_r0)
        
        def theta(mino_time):
            return theta_phase(upsilon_theta*mino_time+q_theta0)
        
        def phi(mino_time):
            return q_phi0 + upsilon_phi*mino_time + phi_r(upsilon_r*mino_time+q_r0) + phi_theta(upsilon_theta*mino_time+q_theta0) - C_phi
        
        return t, r, theta, phi

class BlackHole:
    def __init__(self,a,M):
        """
        Initializes a black hole with mass M and spin parameter a

        :param a: dimensionless angular momentum
        :type a: double
        :param M: mass of the black hole
        :type M: double
        """
        self.a = a
        self.M = M

    def separatrix(self,e,x):
        """
        Computes the value of p at the separatrix for a given value of e and x

        :param e: orbital eccentricity
        :type e: double
        :param x: cosine of the orbital inclination
        :type x: double

        :rtype: double
        """
        return separatrix(self.a,e,x)
    
    def is_stable(self,orbit):
        """
        Determines whether a given orbit is stable or not.

        :param orbit: orbit to be tested
        :type orbit: Orbit

        :rtype: bool
        """
        return is_stable(self.a,orbit.p,orbit.e,orbit.x)
    
    def constants_of_motion(self,orbit,dimensionless=True):
        """
        Computes the constants of motion for a given orbit
        
        :param orbit: orbit to compute the constants of motion for
        :type orbit: Orbit

        :rtype: tuple
        """
        constants = constants_of_motion(self.a, orbit.p, orbit.e, orbit.x)
        return constants if dimensionless else scale_constants(constants, self.M, orbit.mu)
    
    def orbital_frequencies(self,orbit):
        """
        Computes the orbital frequencies for a given orbit

        :param orbit: orbit to compute the orbital frequencies for
        :type orbit: Orbit

        :rtype: tuple
        """
        return orbital_frequencies(self.a, orbit.p, orbit.e, orbit.x)
