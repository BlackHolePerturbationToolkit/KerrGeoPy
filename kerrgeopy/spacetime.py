from .constants_of_motion import *
from numpy import cos, sin

class KerrSpacetime:
    """
        Class representing a black hole with mass M and spin parameter a

        :param a: dimensionless angular momentum
        :type a: double
        :param M: mass of the black hole
        :type M: double

        :ivar a: dimensionless angular momentum
        :ivar M: mass of the black hole
    """
    def __init__(self,a,M):
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
    
    def inner_horizon(self):
        """
        Computes the radius of the inner event horizon

        :return: dimensionless radius of the inner event horizon
        :rtype: double
        """
        return 1-sqrt(1-self.a**2)
    
    def outer_horizon(self):
        """
        Computes the radius of the outer event horizon

        :return: dimensionless radius of the outer event horizon
        :rtype: double
        """
        return 1+sqrt(1-self.a**2)
    
    def metric(self,t,r,theta,phi):
        """
        Returns the matrix representation of the metric at a given point expressed in Boyer-Lindquist coordinates.

        :param t: time coordinate
        :type t: double
        :param r: radial coordinate
        :type r: double
        :param theta: polar coordinate
        :type theta: double
        :param phi: azimuthal coordinate
        :type phi: double

        :rtype: numpy.ndarray
        """
        a = self.a
        sigma = r**2+a**2*cos(theta)**2
        delta = r**2-2*r+a**2
        return np.array(
            [[-(1-2*r/sigma),               0,              0,      -2*a*r*sin(theta)**2/sigma],
             [0,                            sigma/delta,    0,      0],
             [0,                            0,              sigma,  0],
             [-2*a*r*sin(theta)**2/sigma,   0,              0,      sin(theta)**2*(r**2+a**2+2*a**2*r*sin(theta)**2/sigma)]
             ]
        )
    
    def norm(self,t,r,theta,phi,v):
        """
        Computes the norm of a vector at a given point in spacetime expressed in Boyer-Lindquist coordinates

        :param t: time coordinate
        :type t: double
        :param r: radial coordinate
        :type r: double
        :param theta: polar coordinate
        :type theta: double
        :param phi: azimuthal coordinate
        :type phi: double
        :param v: vector to compute the norm of
        :type v: array_like

        :return: norm of v
        :rtype: double
        """
        return np.dot(v,np.dot(self.metric(t,r,theta,phi),v))
    
    def four_velocity(self,t,r,theta,phi,constants):
        r"""
        Computes the four velocity of a given trajectory

        :param t: time component of trajectory
        :type t: function
        :param r: radial component of trajectory
        :type r: function
        :param theta: polar component of trajectory
        :type theta: function
        :param phi: azimuthal component of trajectory
        :type phi: function
        :param constants: tuple of constants of motion in the form :math:`(E,L,Q)`
        :type constants: tuple(double, double, double)

        :return: components of the four velocity (i.e. :math:`(\frac{dt}{d\tau}, \frac{dr}{d\tau},\frac{d\theta}{d\tau},\frac{d\phi}{d\tau})`)
        :rtype: tuple(function, function, function, function)
        """
        a = self.a
        E, L, Q = constants
        # radial polynomial
        R = lambda r: (E*(r**2+a**2)-a*L)**2-(r**2-2*r+a**2)*(r**2+(a*E-L)**2+Q)
        # polar polynomial
        Z = lambda z: Q-(Q+a**2*(1-E**2)+L**2)*z**2+a**2*(1-E**2)*z**4

        def t_prime(time):
            delta = r(time)**2-2*r(time)+a**2
            sigma = r(time)**2+a**2*cos(theta(time))**2
            return 1/sigma*((r(time)**2+a**2)/delta*(E*(r(time)**2+a**2)-a*L)-a**2*E*(1-cos(theta(time))**2)+a*L)

        def r_prime(time):
            sigma = r(time)**2+a**2*cos(theta(time))**2
            return sqrt(R(r(time)))/sigma

        def theta_prime(time):
            sigma = r(time)**2+a**2*cos(theta(time))**2
            return sqrt(Z(cos(theta(time))))/sigma
        
        def phi_prime(time):
            sigma = r(time)**2+a**2*cos(theta(time))**2
            delta = r(time)**2-2*r(time)+a**2
            return 1/sigma*(a/delta*(E*(r(time)**2+a**2)-a*L)+L/(1-cos(theta(time))**2)-a*E)
        
        return t_prime, r_prime, theta_prime, phi_prime
