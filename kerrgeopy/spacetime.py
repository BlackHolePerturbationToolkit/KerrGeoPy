from .constants import *
from .frequencies import *

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
