"""Module containing the KerrSpacetime class"""
from .constants import *
from numpy import cos, sin
import numpy as np


class KerrSpacetime:
    """Class representing spacetime around a black hole with mass :math:`M` 
    and spin parameter :math:`a`

    Parameters
    ----------
    a : double
        dimensionless angular momentum
    M : double
        mass of the black hole

    Attributes
    ----------
    a
        dimensionless angular momentum
    M
        mass of the black hole
    inner_horizon
        radius of the inner horizon
    outer_horizon
        radius of the outer horizon
    """

    def __init__(self, a, M=None):
        self.a, self.M = a, M
        self.inner_horizon = 1 - sqrt(1 - self.a**2)
        self.outer_horizon = 1 + sqrt(1 - self.a**2)

    def separatrix(self, e, x):
        """Computes the value of p at the separatrix for a given value of e and x

        Parameters
        ----------
        e : double
            orbital eccentricity
        x : double
            cosine of the orbital inclination

        Returns
        -------
        double
        """
        return separatrix(self.a, e, x)

    def is_stable(self, p, e, x):
        """Determines whether a given orbit is stable or not

        Parameters
        ----------
        p : double
            dimensionless semi-latus rectum
        e : double
            orbital eccentricity
        x : double
            cosine of the orbital inclination

        Returns
        -------
        bool
        """
        return is_stable(self.a, p, e, x)

    def metric(self, t, r, theta, phi):
        """Returns the matrix representation of the metric at a given point 
        expressed in Boyer-Lindquist coordinates.

        Parameters
        ----------
        t : double
            time coordinate
        r : double
            radial coordinate
        theta : double
            polar coordinate
        phi : double
            azimuthal coordinate

        Returns
        -------
        numpy.ndarray
        """
        a = self.a
        sigma = r**2 + a**2 * cos(theta) ** 2
        delta = r**2 - 2 * r + a**2
        # fmt: off
        return np.array(
            [[-(1-2*r/sigma),               0,              0,      -2*a*r*sin(theta)**2/sigma],
             [0,                            sigma/delta,    0,      0],
             [0,                            0,              sigma,  0],
             [-2*a*r*sin(theta)**2/sigma,   0,              0,      sin(theta)**2*(r**2+a**2+2*a**2*r*sin(theta)**2/sigma)]
             ]
        )
        # fmt: on

    def norm(self, t, r, theta, phi, v):
        """Computes the norm of a vector at a given point in spacetime 
        expressed in Boyer-Lindquist coordinates

        Parameters
        ----------
        t : double
            time coordinate
        r : double
            radial coordinate
        theta : double
            polar coordinate
        phi : double
            azimuthal coordinate
        v : array_like
            vector to compute the norm of

        Returns
        -------
        double
            norm of v
        """
        return np.dot(v, np.dot(self.metric(t, r, theta, phi), v))

    def four_velocity(self, t, r, theta, phi, constants, upsilon_r, upsilon_theta, initial_phases):
        r"""Computes the four velocity of a given trajectory

        Parameters
        ----------
        t : function
            time component of trajectory
        r : function
            radial component of trajectory
        theta : function
            polar component of trajectory
        phi : function
            azimuthal component of trajectory
        constants : tuple(double, double, double)
            tuple of constants of motion in the form 
            :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})`
        upsilon_r : double
            radial frequency
        upsilon_theta : double
            polar frequency
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`

        Returns
        -------
        tuple(function, function, function, function)
            components of the four velocity (i.e.
            :math:`(u^t,u^r,u^\theta,u^\phi)`)
        """
        a = self.a
        E, L, Q = constants
        q_t0, q_r0, q_theta0, q_phi0 = initial_phases

        # radial polynomial
        R = lambda r: (E * (r**2 + a**2) - a * L) ** 2 - (r**2 - 2 * r + a**2) * (
            r**2 + (a * E - L) ** 2 + Q
        )
        # polar polynomial
        Z = (
            lambda z: Q
            - (Q + a**2 * (1 - E**2) + L**2) * z**2
            + a**2 * (1 - E**2) * z**4
        )
        
        def u_t(time):
            delta = r(time) ** 2 - 2 * r(time) + a**2
            sigma = r(time) ** 2 + a**2 * cos(theta(time)) ** 2
            return (
                1
                / sigma
                * (
                    (r(time) ** 2 + a**2) / delta * (E * (r(time) ** 2 + a**2) - a * L)
                    - a**2 * E * (1 - cos(theta(time)) ** 2)
                    + a * L
                )
            )

        def u_r(time):
            q_r = upsilon_r * time + q_r0
            sigma = r(time) ** 2 + a**2 * cos(theta(time)) ** 2
            return np.copysign(1, sin(q_r)) * sqrt(abs(R(r(time)))) / sigma

        def u_theta(time):
            q_theta = upsilon_theta * time + q_theta0
            sigma = r(time) ** 2 + a**2 * cos(theta(time)) ** 2
            return (
                np.copysign(1, sin(q_theta))
                * sqrt(abs(Z(cos(theta(time)))))
                / (sigma * sin(theta(time)))
            )

        def u_phi(time):
            sigma = r(time) ** 2 + a**2 * cos(theta(time)) ** 2
            delta = r(time) ** 2 - 2 * r(time) + a**2
            return (
                1
                / sigma
                * (
                    a / delta * (E * (r(time) ** 2 + a**2) - a * L)
                    + L / (1 - cos(theta(time)) ** 2)
                    - a * E
                )
            )

        return u_t, u_r, u_theta, u_phi
