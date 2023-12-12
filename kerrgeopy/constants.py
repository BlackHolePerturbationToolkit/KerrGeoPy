"""Module containing functions for computing the constants of motion for an orbit in Kerr Spacetime.
Constants of motion are computed using the method described in Appendix B of `Schmidt <https://doi.org/10.48550/arXiv.gr-qc/0202090>`_.
"""
from numpy import sign, sqrt, copysign, pi, nan, inf
from scipy.optimize import root_scalar
from .units import *
from scipy.interpolate import RectBivariateSpline
import numpy as np
from numpy.polynomial import Polynomial


def stable_radial_roots(a, p, e, x, constants=None):
    """Computes the radial roots for a stable bound orbit as defined in 
    equation 10 of `Fujita and Hikida <https://doi.org/10.48550/arXiv.0906.1420>`_. 
    Roots are given in decreasing order.

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    x : tuple(double, double, double, double)
        cosine of the orbital inclination
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit

    Returns
    -------
    tuple(double, double, double, double)
        tuple containing the four roots of the radial equation
    """
    if constants is None:
        constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants

    r1 = p / (1 - e)
    r2 = p / (1 + e)

    A_plus_B = 2 / (1 - E**2) - r1 - r2
    AB = a**2 * Q / (r1 * r2 * (1 - E**2))

    r3 = (A_plus_B + sqrt(A_plus_B**2 - 4 * AB)) / 2
    r4 = AB / r3

    return r1, r2, r3, r4


def plunging_radial_roots(a, E, L, Q):
    """Computes the radial roots for a plunging orbit. If all roots are real, 
    roots are sorted such that the motion is between r1 and r2 and roots are 
    otherwise in decreasing order. If there are two complex roots, 
    r1 < r2 are real and r3/r4 are complex conjugates.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    E : double
        dimensionless energy
    L : double
        dimensionless angular momentum
    Q : double
        dimensionless Carter constant

    Returns
    -------
    tuple(double,double,double,double)
        tuple of radial roots
    """
    # standard form of the radial polynomial R(r)
    R = Polynomial(
        [
            -(a**2) * Q,
            2 * L**2 + 2 * Q + 2 * a**2 * E**2 - 4 * a * E * L,
            a**2 * E**2 - L**2 - Q - a**2,
            2,
            E**2 - 1,
        ]
    )
    roots = R.roots()
    # get the real roots and the complex roots
    real_roots = np.sort(np.real(roots[np.isreal(roots)]))
    complex_roots = roots[np.iscomplex(roots)]

    r_minus = 1 - sqrt(1 - a**2)

    # if there are 4 real roots, by convention r4 < r3 < r2 < r1 
    # (consistent with stable orbits)
    if len(real_roots) == 4:
        # if there are three roots outside the event horizon swap r1/r3 and r2/r4
        if real_roots[1] > r_minus:
            r1 = real_roots[1]
            r2 = real_roots[0]
            r3 = real_roots[3]
            r4 = real_roots[2]
        else:
            r4, r3, r2, r1 = real_roots

    # in the case of two complex roots, r1 < r2 are real and r3/r4 are complex conjugates
    elif len(real_roots) == 2:
        r1, r2 = real_roots
        r3, r4 = complex_roots

    return r1, r2, r3, r4


def stable_polar_roots(a, p, e, x, constants=None):
    r"""Computes the polar roots for a stable bound orbit as defined in equation 
    10 of `Fujita and Hikida <https://doi.org/10.48550/arXiv.0906.1420>`_. 
    Roots are given in increasing order.

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    x : tuple(double, double, double)
        cosine of the orbital inclination
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit

    Returns
    -------
    tuple(double, double, double, double)
        tuple of roots :math:`(z_-, z_+)`
    """
    if constants is None:
        constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants
    epsilon0 = a**2 * (1 - E**2) / L**2
    z_minus = 1 - x**2
    # z_plus = a**2*(1-E**2)/(L**2*epsilon0)+1/(epsilon0*(1-z_minus))
    # simplified using definition of carter constant
    z_plus = nan if a == 0 else 1 + 1 / (epsilon0 * (1 - z_minus))

    return z_minus, z_plus


def _coefficients(r, a, x):
    """Computes the coefficients f, g, h and d from equation B.5 in 
    `Schmidt <https://doi.org/10.48550/arXiv.gr-qc/0202090>`_

    Parameters
    ----------
    r : double
        dimensionless distance from the black hole
    a : double
        dimensionless spin of the black hole
    x : double
        cosine of the orbital inclination

    Returns
    -------
    tuple(double, double, double, double)
    """
    z = sqrt(1 - x**2)
    delta = r**2 - 2 * r + a**2
    f = lambda r: r**4 + a**2 * (r * (r + 2) + z**2 * delta)
    g = lambda r: 2 * a * r
    h = lambda r: r * (r - 2) + z**2 / (1 - z**2) * delta
    d = lambda r: (r**2 + a**2 * z**2) * delta

    return f(r), g(r), h(r), d(r)


def _coefficients_derivative(r, a, x):
    """Computes the derivatives f', g', h' and d' of the coefficients from 
    equation B.5 in `Schmidt <https://doi.org/10.48550/arXiv.gr-qc/0202090>`_

    Parameters
    ----------
    r : double
        dimensionless distance from the black hole
    a : double
        dimensionless spin of the black hole
    x : double
        cosine of the orbital inclination

    Returns
    -------
    tuple(double, double, double, double)
    """
    z = sqrt(1 - x**2)
    f_prime = lambda r: 4 * r**3 + 2 * a**2 * ((1 + z**2) * r + (1 - z**2))
    g_prime = lambda r: 2 * a
    h_prime = lambda r: 2 * (r - 1) / (1 - z**2)
    d_prime = lambda r: 2 * (2 * r - 3) * r**2 + 2 * a**2 * ((1 + z**2) * r - z**2)

    return f_prime(r), g_prime(r), h_prime(r), d_prime(r)


def _standardize_params(a, x):
    """Changes signs of a and x so that a is positive 
    and x encodes the direction of the orbit.

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    x : double
        cosine of the orbital inclination

    Returns
    -------
    tuple(double, double)
    """
    return abs(a), x * copysign(1, a)


def energy(a, p, e, x):
    """Computes the dimensionless energy of a bound orbit with the given parameters

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity (must satisfy 0 <= e <= 1)
    x : double
        cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)

    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)

    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a, p, e, x):
        raise ValueError("Not a stable orbit")

    # marginally bound case
    if e == 1:
        return 1

    # polar case
    if x == 0:
        # expression from ConstantsOfMotion.m in the KerrGeodesics mathematica library
        return sqrt(
            -(
                (
                    p
                    * (
                        a**4 * (-1 + e**2) ** 2
                        + (-4 * e**2 + (-2 + p) ** 2) * p**2
                        + 2 * a**2 * p * (-2 + p + e**2 * (2 + p))
                    )
                )
                / (
                    a**4 * (-1 + e**2) ** 2 * (-1 + e**2 - p)
                    + (3 + e**2 - p) * p**4
                    - 2 * a**2 * p**2 * (-1 - e**4 + p + e**2 * (2 + p))
                )
            )
        )

    # spherical case
    if e == 0:
        r0 = p
        f1, g1, h1, d1 = _coefficients(r0, a, x)
        f2, g2, h2, d2 = _coefficients_derivative(r0, a, x)
    # generic case
    else:
        r1 = p / (1 - e)
        r2 = p / (1 + e)
        f1, g1, h1, d1 = _coefficients(r1, a, x)
        f2, g2, h2, d2 = _coefficients(r2, a, x)

    # equation B.19 - B.21
    kappa = d1 * h2 - h1 * d2
    rho = f1 * h2 - h1 * f2
    sigma = g1 * h2 - h1 * g2
    epsilon = d1 * g2 - g1 * d2
    eta = f1 * g2 - g1 * f2

    # equation B.22
    return sqrt(
        (
            kappa * rho
            + 2 * epsilon * sigma
            - sign(x)
            * 2
            * sqrt(sigma * (sigma * epsilon**2 + rho * epsilon * kappa - eta * kappa**2))
        )
        / (rho**2 + 4 * eta * sigma)
    )


def angular_momentum(a, p, e, x, E=None):
    """Computes the dimensionless angular momentum 
    of a bound orbit with the given parameters

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity (must satisfy 0 <= e <= 1)
    x : double
        cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)
    E : double, optional
        dimensionless energy of the orbit can be passed in to speed
        computation if it is already known

    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)

    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a, p, e, x):
        raise ValueError("Not a stable orbit")

    # compute energy if not given
    if E is None:
        E = energy(a, p, e, x)

    # polar case
    if x == 0:
        return 0

    # marginally bound case
    if e == 1:
        r2 = p / (1 + e)
        f2, g2, h2, d2 = _coefficients(r2, a, x)
        # obtained by solving equation B.17 for L
        return (-E * g2 + sign(x) * sqrt(-d2 * h2 + E**2 * (g2**2 + f2 * h2))) / h2

    # generic case
    else:
        r1 = p / (1 - e)
        f1, g1, h1, d1 = _coefficients(r1, a, x)
        # obtained by solving equation B.17 for L
        return (-E * g1 + sign(x) * sqrt(-d1 * h1 + E**2 * (g1**2 + f1 * h1))) / h1


def carter_constant(a, p, e, x, E=None, L=None):
    """Computes the dimensionless carter constant 
    of a bound orbit with the given parameters

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity (must satisfy 0 <= e <= 1)
    x : double
        cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)
    E : double, optional
        dimensionless energy of the orbit can be passed in to speed
        computation if it is already known
    L : double, optional
        dimensionless angular momentum of the orbit can be passed in to
        speed computation if it is already known

    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)

    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a, p, e, x):
        raise ValueError("Not a stable orbit")

    # polar case
    if x == 0:
        # expression from ConstantsOfMotion.m in the KerrGeodesics mathematica library
        return -(
            (
                p**2
                * (
                    a**4 * (-1 + e**2) ** 2
                    + p**4
                    + 2 * a**2 * p * (-2 + p + e**2 * (2 + p))
                )
            )
            / (
                a**4 * (-1 + e**2) ** 2 * (-1 + e**2 - p)
                + (3 + e**2 - p) * p**4
                - 2 * a**2 * p**2 * (-1 - e**4 + p + e**2 * (2 + p))
            )
        )

    z = sqrt(1 - x**2)
    # compute energy and angular momentum if not given
    if E is None:
        E = energy(a, p, e, x)
    if L is None:
        L = angular_momentum(a, p, e, x, E)
    #  equation B.4
    return z**2 * (a**2 * (1 - E**2) + L**2 / (1 - z**2))


def constants_of_motion(a, p, e, x):
    """Computes the dimensionless energy, angular momentum, and 
    Carter constant of a bound orbit with the given parameters

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity (must satisfy 0 <= e <= 1)
    x : double
        cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)

    Returns
    -------
    tuple(double, double, double)
        tuple of constants of motion :math:`(E, L, Q)`
    """
    E = energy(a, p, e, x)
    L = angular_momentum(a, p, e, x, E)
    Q = carter_constant(a, p, e, x, E, L)
    return E, L, Q


def apex_from_constants(a, E, L, Q):
    r"""Computes the orbital parameters :math:`(a,p,e,x)` for a 
    stable bound orbit with the given constants of motion

    Parameters
    ----------
    a : double
        spin parameter
    E : double
        dimensionless energy
    L : double
        dimensionless angular momentum
    Q : double
        dimensionless Carter constant

    Returns
    -------
    tuple(double,double,double,double)
        tuple of orbital parameters :math:`(a,p,e,x)`
    """
    # Radial polynomial
    R = Polynomial(
        [
            -(a**2) * Q,
            2 * L**2 + 2 * Q + 2 * a**2 * E**2 - 4 * a * E * L,
            a**2 * E**2 - L**2 - Q - a**2,
            2,
            E**2 - 1,
        ]
    )
    radial_roots = R.roots()
    # numpy returns roots in increasing order
    r4, r3, r2, r1 = radial_roots

    # polar polynomial written in terms of z = cos^2(theta)
    Z = Polynomial([Q, -(Q + a**2 * (1 - E**2) + L**2), a**2 * (1 - E**2)])
    polar_roots = Z.roots()
    if a == 0:
        polar_roots = [polar_roots[0], polar_roots[0]]
    z_minus, z_plus = polar_roots

    p = 2 * r1 * r2 / (r1 + r2)
    e = (r1 - r2) / (r1 + r2)
    x = np.sign(L) * sqrt(1 - z_minus)

    return a, p, e, x


def _S_polar(p, a, e):
    """Separatrix polynomial for a polar orbit from equation 37 in
    `Stein and Warburton <https://doi.org/10.48550/arXiv.1912.07609>`_

    Parameters
    ----------
    p : double
        orbital semi-latus rectum
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity

    Returns
    -------
    double
    """
    return (
        p**5 * (-6 - 2 * e + p)
        + a**2 * p**3 * (-4 * (-1 + e) * (1 + e) ** 2 + (3 + e * (2 + 3 * e)) * p)
        - a**4
        * (1 + e) ** 2
        * p
        * (6 + 2 * e**3 + 2 * e * (-1 + p) - 3 * p - 3 * e**2 * (2 + p))
        + a**6 * (-1 + e) ** 2 * (1 + e) ** 4
    )


def _S_equatorial(p, a, e):
    """Separatrix polynomial for an equatorial orbit from equation 23 in
    `Stein and Warburton <https://doi.org/10.48550/arXiv.1912.07609>`_

    Parameters
    ----------
    p : double
        orbital semi-latus rectum
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity

    Returns
    -------
    double
    """
    return (
        a**4 * (-3 - 2 * e + e**2) ** 2
        + p**2 * (-6 - 2 * e + p) ** 2
        - 2 * a**2 * (1 + e) * p * (14 + 2 * e**2 + 3 * p - e * p)
    )


def _S(p, a, e, x):
    """Full separatrix polynomial from equation A1 in
    `Stein and Warburton <https://doi.org/10.48550/arXiv.1912.07609>`_

    Parameters
    ----------
    p : double
        orbital semi-latus rectum
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity

    Returns
    -------
    double
    """
    # fmt: off
    return -4*(3 + e)*p**11 + p**12 + \
       a**12*(-1 + e)**4*(1 + e)**8*(-1 + x)**4*(1 + x)**4 - \
       4*a**10*(-3 + e)*(-1 + e)**3*(1 + e)**7*p*(-1 + x**2)**4 - \
       4*a**8*(-1 + e)*(1 + e)**5*p**3*(-1 + x)**3*(1 + x)**3* \
        (7 - 7*x**2 - e**2*(-13 + x**2) + e**3*(-5 + x**2) + 7*e*(-1 + x**2)) + \
       8*a**6*(-1 + e)*(1 + e)**3*p**5*(-1 + x**2)**2* \
        (3 + e + 12*x**2 + 4*e*x**2 + e**3*(-5 + 2*x**2) + e**2*(1 + 2*x**2)) - \
       8*a**4*(1 + e)**2*p**7*(-1 + x)*(1 + x)* \
        (-3 + e + 15*x**2 - 5*e*x**2 + e**3*(-5 + 3*x**2) + e**2*(-1 + 3*x**2))\
        + 4*a**2*p**9*(-7 - 7*e + e**3*(-5 + 4*x**2) + e**2*(-13 + 12*x**2)) + \
       2*a**8*(-1 + e)**2*(1 + e)**6*p**2*(-1 + x**2)**3* \
        (2*(-3 + e)**2*(-1 + x**2) + \
          a**2*(e**2*(-3 + x**2) - 3*(1 + x**2) + 2*e*(1 + x**2))) - \
       2*p**10*(-2*(3 + e)**2 + a**2* \
           (-3 + 6*x**2 + e**2*(-3 + 2*x**2) + e*(-2 + 4*x**2))) + \
       a**6*(1 + e)**4*p**4*(-1 + x**2)**2* \
        (-16*(-1 + e)**2*(-3 - 2*e + e**2)*(-1 + x**2) + \
          a**2*(15 + 6*x**2 + 9*x**4 + e**2*(26 + 20*x**2 - 2*x**4) + \
             e**4*(15 - 10*x**2 + x**4) + 4*e**3*(-5 - 2*x**2 + x**4) - \
             4*e*(5 + 2*x**2 + 3*x**4))) - \
       4*a**4*(1 + e)**2*p**6*(-1 + x)*(1 + x)* \
        (-2*(11 - 14*e**2 + 3*e**4)*(-1 + x**2) + \
          a**2*(5 - 5*x**2 - 9*x**4 + 4*e**3*x**2*(-2 + x**2) + \
             e**4*(5 - 5*x**2 + x**4) + e**2*(6 - 6*x**2 + 4*x**4))) + \
       a**2*p**8*(-16*(1 + e)**2*(-3 + 2*e + e**2)*(-1 + x**2) + \
          a**2*(15 - 36*x**2 + 30*x**4 + e**4*(15 - 20*x**2 + 6*x**4) + \
             4*e**3*(5 - 12*x**2 + 6*x**4) + 4*e*(5 - 12*x**2 + 10*x**4) + \
             e**2*(26 - 72*x**2 + 44*x**4)))
    # fmt: on


def separatrix(a, e, x):
    """Returns the value of p at the separatrix for the given orbital parameters 
    computed using the bracked root finding method described in 
    `Stein and Warburton <https://doi.org/10.48550/arXiv.1912.07609>`_

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    e : double
        orbital eccentricity (must satisfy 0 <= e <= 1)
    x : double
        cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)

    Returns
    -------
    double
    """

    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    if a == 0:
        return 6 + 2 * e

    polar_bracket = [1 + sqrt(3) + sqrt(3 + 2 * sqrt(3)), 8]
    p_polar = root_scalar(_S_polar, args=(a, e), bracket=polar_bracket)

    if x == 0:
        return p_polar.root

    equatorial_prograde_bracket = [1 + e, 6 + 2 * e]
    p_equatorial_prograde = root_scalar(
        _S_equatorial, args=(a, e), bracket=equatorial_prograde_bracket
    )

    if x == 1:
        return p_equatorial_prograde.root

    if x == -1:
        equatorial_retrograde_bracket = [6 + 2 * e, 5 + e + 4 * sqrt(1 + e)]
        p_equatorial_retrograde = root_scalar(
            _S_equatorial, args=(a, e), bracket=equatorial_retrograde_bracket
        )
        return p_equatorial_retrograde.root

    if x > 0:
        p = root_scalar(_S, args=(a, e, x), bracket=[p_equatorial_prograde.root, p_polar.root])
        return p.root

    if x < 0:
        p = root_scalar(_S, args=(a, e, x), bracket=[p_polar.root, 12])
        return p.root


def fast_separatrix(a, grid_spacing=0.01):
    """Constructs a faster separatrix function for a given value of :math:`a` 
    by interpolating over a grid of :math:`e` and :math:`x` values.

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    grid_spacing : double, optional
        spacing of the grid over which to interpolate, defaults to 0.01

    Returns
    -------
    scipy.interpolate.RectBivariateSpline
        interpolated function of e and x
    """

    # create grid of e and x values to interpolate over
    num_e_pts = int(1 / grid_spacing)
    num_x_pts = int(2 / grid_spacing)
    e = np.linspace(0, 1, num_e_pts)
    x = np.linspace(-1, 1, num_x_pts)
    E, X = np.meshgrid(e, x)

    # compute separatrix values on grid
    P = np.zeros((num_e_pts, num_x_pts))
    for i in range(num_e_pts):
        for j in range(num_x_pts):
            P[i, j] = separatrix(a, E[j, i], X[j, i])

    # create interpolator
    interpolator = RectBivariateSpline(e, x, P)

    return interpolator


def is_stable(a, p, e, x):
    """Tests whether or not the given orbital parameters define a stable bound orbit

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    boolean
    """
    if p > separatrix(a, e, x):
        return True
    return False


def valid_params(a, e, x):
    """Tests whether the given parameters fall into the allowed ranges

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    boolean
    """
    if (0 <= a <= 1) and (0 <= e <= 1) and (-1 <= x <= 1):
        return True
    return False


def scale_constants(constants, M, mu):
    """Scales the dimensionless constants of motion to the given mass parameters

    Parameters
    ----------
    constants : tuple
        dimensionless constants of motion in the form (E, L, Q)
    M : double
        mass of the black hole
    mu : double
        mass ratio

    Returns
    -------
    tuple(double, double, double)
    """
    M = mass_in_kg(M)
    return constants[0] * mu, constants[1] * mu * M, constants[2] * mu**2 * M**2
