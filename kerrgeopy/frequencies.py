"""Module containing functions for computing frequencies of motion for orbits in Kerr spacetime.
Frequencies are computed using the method derived in `Fujita and Hikida <https://doi.org/10.48550/arXiv.0906.1420>`_
"""
from .constants import _standardize_params
from .constants import *
from scipy.special import ellipk, ellipe, elliprj, elliprf, elliprd
from numpy import sin, cos, sqrt, pi, arcsin, floor, where


def _ellipeinc(phi, m):
    r"""Incomplete elliptic integral of the second kind defined as 
    :math:`E(\phi,m) = \int_0^{\phi} \sqrt{1-m\sin^2\theta}d\theta`.

    Parameters
    ----------
    phi : double
    m : double

    Returns
    -------
    double
    """
    # count the number of half periods

    num_cycles = floor(phi / (pi / 2))
    # map phi to [0,pi/2]
    phi = abs(arcsin(sin(phi)))

    # formula from https://en.wikipedia.org/wiki/Carlson_symmetric_form
    integral = sin(phi) * elliprf(cos(phi) ** 2, 1 - m * sin(phi) ** 2, 1) - 1 / 3 * m * sin(
        phi
    ) ** 3 * elliprd(cos(phi) ** 2, 1 - m * sin(phi) ** 2, 1)
    result = where(
        num_cycles % 2 == 0,
        num_cycles * ellipe(m) + integral,
        (num_cycles + 1) * ellipe(m) - integral,
    )

    # return scalar for scalar input
    return result.item() if np.isscalar(phi) else result


def _ellippi(n, m):
    r"""Complete elliptic integral of the third kind defined as 
    :math:`\Pi(n,m) = \int_0^{\frac{\pi}{2}} \frac{d\theta}{(1-n\sin^2{\theta})\sqrt{1-m\sin^2{\theta}}}`

    Parameters
    ----------
    n : double
    m : double

    Returns
    -------
    double
    """
    # Note: sign of n is reversed from the definition in Fujita and Hikida

    # formula from https://en.wikipedia.org/wiki/Carlson_symmetric_form
    return elliprf(0, 1 - m, 1) + 1 / 3 * n * elliprj(0, 1 - m, 1, 1 - n)


def _ellippiinc(phi, n, m):
    r"""Incomplete elliptic integral of the third kind defined as 
    :math:`\Pi(\phi,n,m) = \int_0^{\phi} \frac{1}{1-n\sin^2\theta}\frac{1}{\sqrt{1-m\sin^2\theta}}d\theta`.

    Parameters
    ----------
    phi : double
    n : double
    m : double

    Returns
    -------
    double
    """
    # Note: sign of n is reversed from the definition in Fujita and Hikida

    # count the number of half periods
    num_cycles = floor(phi / (pi / 2))
    # map phi to [0,pi/2]
    phi = abs(arcsin(sin(phi)))
    # formula from https://en.wikipedia.org/wiki/Carlson_symmetric_form
    integral = sin(phi) * elliprf(cos(phi) ** 2, 1 - m * sin(phi) ** 2, 1) + 1 / 3 * n * sin(
        phi
    ) ** 3 * elliprj(cos(phi) ** 2, 1 - m * sin(phi) ** 2, 1, 1 - n * sin(phi) ** 2)

    result = where(
        num_cycles % 2 == 0,
        num_cycles * _ellippi(n, m) + integral,
        (num_cycles + 1) * _ellippi(n, m) - integral,
    )

    # return scalar for scalar input
    return result.item() if np.isscalar(phi) else result


def r_frequency(a, p, e, x, constants=None):
    """Computes the frequency of motion in r in Mino time

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity (must satisfy 0 <= e < 1)
    x : double
        cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    constants : tuple(double, double, double), optional
        dimensionless constants of motion for the orbit can be passed in
        to speed computation if they are already known

    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)

    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if x == 0:
        raise ValueError("Polar orbits not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a, p, e, x):
        raise ValueError("Not a stable orbit")

    # compute constants if not passed in
    if constants is None:
        constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants

    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x, constants)
    # equation 13
    k_r = sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))
    # equation 15
    return pi * sqrt((1 - E**2) * (r1 - r3) * (r2 - r4)) / (2 * ellipk(k_r**2))


def theta_frequency(a, p, e, x, constants=None):
    """Computes the frequency of motion in theta in Mino time

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity (must satisfy 0 <= e < 1)
    x : double
        cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    constants : tuple(double, double, double), optional
        dimensionless constants of motion for the orbit can be passed in
        to speed computation if they are already known

    Returns
    -------
    double
    """

    a, x = _standardize_params(a, x)

    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if x == 0:
        raise ValueError("Polar orbits not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a, p, e, x):
        raise ValueError("Not a stable orbit")

    # Schwarzschild case
    if a == 0:
        return p / sqrt(-3 - e**2 + p) * (sign(x) if e == 0 else 1)

    # compute constants if not provided
    if constants is None:
        constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants
    z_minus, z_plus = stable_polar_roots(a, p, e, x, constants)

    # equation 13
    k_theta = sqrt(z_minus / z_plus)
    # simplified form of epsilon0*z_plus
    e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))

    # equation 15
    return pi * L * sqrt(e0zp) / (2 * ellipk(k_theta**2))


def phi_frequency(a, p, e, x, constants=None, upsilon_r=None, upsilon_theta=None):
    """Computes the frequency of motion in phi in Mino time

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity (must satisfy 0 <= e < 1)
    x : double
        cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    constants : tuple(double, double, double), optional
        dimensionless constants of motion for the orbit can be passed in
        to speed computation if they are already known
    upsilon_r : double, optional
        Mino frequency of motion in r can be passed in to speed
        computation if it is already known
    upsilon_theta : double, optional
        Mino frequency of motion in theta can be passed in to speed
        computation if it is already known

    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)

    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if x == 0:
        raise ValueError("Polar orbits not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a, p, e, x):
        raise ValueError("Not a stable orbit")

    # Schwarzschild case
    if a == 0:
        return sign(x) * p / sqrt(-3 - e**2 + p)

    # compute constants if they are not passed in
    if constants is None:
        constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants
    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x, constants)
    z_minus, z_plus = stable_polar_roots(a, p, e, x, constants)

    # compute frequencies if they are not passed in
    if upsilon_r is None:
        upsilon_r = r_frequency(a, p, e, x, constants)
    if upsilon_theta is None:
        upsilon_theta = theta_frequency(a, p, e, x, constants)

    # simplified form of epsilon0*z_plus
    e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))

    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)

    h_plus = (r1 - r2) * (r3 - r_plus) / ((r1 - r3) * (r2 - r_plus))
    h_minus = (r1 - r2) * (r3 - r_minus) / ((r1 - r3) * (r2 - r_minus))

    # equation 13
    k_theta = sqrt(z_minus / z_plus)
    k_r = sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))

    # equation 21
    return 2 * upsilon_theta / (pi * sqrt(e0zp)) * _ellippi(
        z_minus, k_theta**2
    ) + 2 * a * upsilon_r / (
        pi * (r_plus - r_minus) * sqrt((1 - E**2) * (r1 - r3) * (r2 - r4))
    ) * (
        (2 * E * r_plus - a * L)
        / (r3 - r_plus)
        * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_plus) * _ellippi(h_plus, k_r**2))
        - (2 * E * r_minus - a * L)
        / (r3 - r_minus)
        * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_minus) * _ellippi(h_minus, k_r**2))
    )


def gamma(a, p, e, x, constants=None, upsilon_r=None, upsilon_theta=None):
    """Computes the average rate at which observer time accumulates in Mino time

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity (must satisfy 0 <= e < 1)
    x : double
        cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    constants : tuple(double, double, double), optional
        dimensionless constants of motion for the orbit can be passed in
        to speed computation if they are already known
    upsilon_r : double, optional
        Mino frequency of motion in r can be passed in to speed
        computation if it is already known
    upsilon_theta : double, optional
        Mino frequency of motion in theta can be passed in to speed
        computation if it is already known

    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)

    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if x == 0:
        raise ValueError("Polar orbits not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a, p, e, x):
        raise ValueError("Not a stable orbit")

    # marginally bound case
    if e == 1:
        return inf

    # compute constants if they are not passed in
    if constants is None:
        constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants
    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x, constants)
    z_minus, z_plus = stable_polar_roots(a, p, e, x, constants)
    epsilon0 = a**2 * (1 - E**2) / L**2
    # simplified form of a**2*sqrt(z_plus/epsilon0)
    a2sqrt_zp_over_e0 = (
        L**2 / ((1 - E**2) * sqrt(1 - z_minus))
        if a == 0
        else a**2 * z_plus / sqrt(epsilon0 * z_plus)
    )

    # compute frequencies if they are not passed in
    if upsilon_r is None:
        upsilon_r = r_frequency(a, p, e, x, constants)
    if upsilon_theta is None:
        upsilon_theta = theta_frequency(a, p, e, x, constants)

    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)

    h_r = (r1 - r2) / (r1 - r3)
    h_plus = (r1 - r2) * (r3 - r_plus) / ((r1 - r3) * (r2 - r_plus))
    h_minus = (r1 - r2) * (r3 - r_minus) / ((r1 - r3) * (r2 - r_minus))

    # equation 13
    k_theta = 0 if a == 0 else sqrt(z_minus / z_plus)
    k_r = sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))

    # equation 21
    return (
        4 * E
        + 2
        * a2sqrt_zp_over_e0
        * E
        * upsilon_theta
        * (ellipk(k_theta**2) - ellipe(k_theta**2))
        / (pi * L)
        + 2
        * upsilon_r
        / (pi * sqrt((1 - E**2) * (r1 - r3) * (r2 - r4)))
        * (
            E
            / 2
            * (
                (r3 * (r1 + r2 + r3) - r1 * r2) * ellipk(k_r**2)
                + (r2 - r3) * (r1 + r2 + r3 + r4) * _ellippi(h_r, k_r**2)
                + (r1 - r3) * (r2 - r4) * ellipe(k_r**2)
            )
            + 2 * E * (r3 * ellipk(k_r**2) + (r2 - r3) * _ellippi(h_r, k_r**2))
            + 2
            / (r_plus - r_minus)
            * (
                ((4 * E - a * L) * r_plus - 2 * a**2 * E)
                / (r3 - r_plus)
                * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_plus) * _ellippi(h_plus, k_r**2))
                - ((4 * E - a * L) * r_minus - 2 * a**2 * E)
                / (r3 - r_minus)
                * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_minus) * _ellippi(h_minus, k_r**2))
            )
        )
    )


def mino_frequencies(a, p, e, x):
    r"""Computes frequencies of orbital motion in Mino time

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity (must satisfy 0 <= e < 1)
    x : double
        cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)

    Returns
    -------
    tuple
        tuple of the form :math:`(\Upsilon_r, \Upsilon_\theta,
        \Upsilon_\phi, \Gamma)`
    """
    constants = constants_of_motion(a, p, e, x)
    upsilon_r = r_frequency(a, p, e, x, constants)
    upsilon_theta = theta_frequency(a, p, e, x, constants)
    upsilon_phi = phi_frequency(a, p, e, x, constants, upsilon_r, upsilon_theta)
    Gamma = gamma(a, p, e, x, constants, upsilon_r, upsilon_theta)

    return upsilon_r, abs(upsilon_theta), upsilon_phi, Gamma


def fundamental_frequencies(a, p, e, x):
    r"""Computes frequencies of orbital motion in Boyer-Lindquist time

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity (must satisfy 0 <= e < 1)
    x : double
        cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)

    Returns
    -------
    tuple
        tuple of the form :math:`(\Omega_r, \Omega_\theta, \Omega_\phi)`
    """
    constants = constants_of_motion(a, p, e, x)
    upsilon_r = r_frequency(a, p, e, x, constants)
    upsilon_theta = theta_frequency(a, p, e, x, constants)
    upsilon_phi = phi_frequency(a, p, e, x, constants, upsilon_r, upsilon_theta)
    Gamma = gamma(a, p, e, x, constants, upsilon_r, upsilon_theta)

    return upsilon_r / Gamma, abs(upsilon_theta) / Gamma, upsilon_phi / Gamma


def plunging_mino_frequencies(a, E, L, Q):
    r"""Computes the radial and polar mino frequencies for a plunging orbit.

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
    tuple(double,double)
        radial and polar mino frequencies
        :math:`(\Upsilon_r,\Upsilon_\theta)`
    """

    radial_roots = plunging_radial_roots(a, E, L, Q)
    r1, r2, r3, r4 = radial_roots
    rho_r = np.real(r3)
    rho_i = np.imag(r4)
    if np.iscomplex(radial_roots[3]):
        # equation 42
        A = sqrt((r2 - rho_r) ** 2 + rho_i**2)
        B = sqrt((r1 - rho_r) ** 2 + rho_i**2)
        k_r = sqrt(abs(((r2 - r1) ** 2 - (A - B) ** 2) / (4 * A * B)))

        # equation 52
        upsilon_r = pi * sqrt(A * B * (1 - E**2)) / (2 * ellipk(k_r**2))

        # equation 14
        z1 = sqrt(
            1
            / 2
            * (
                1
                + (L**2 + Q) / (a**2 * (1 - E**2))
                - sqrt(
                    (1 + (L**2 + Q) / (a**2 * (1 - E**2))) ** 2
                    - 4 * Q / (a**2 * (1 - E**2))
                )
            )
        )
        z2 = sqrt(
            a**2
            * (1 - E**2)
            / 2
            * (
                1
                + (L**2 + Q) / (a**2 * (1 - E**2))
                + sqrt(
                    (1 + (L**2 + Q) / (a**2 * (1 - E**2))) ** 2
                    - 4 * Q / (a**2 * (1 - E**2))
                )
            )
        )
        k_theta = a * sqrt(1 - E**2) * z1 / z2

        # equation 53
        upsilon_theta = pi / 2 * z2 / ellipk(k_theta**2)
    else:
        # polar polynomial written in terms of z = cos^2(theta)
        Z = Polynomial([Q, -(Q + a**2 * (1 - E**2) + L**2), a**2 * (1 - E**2)])
        polar_roots = Z.roots()
        if a == 0:
            polar_roots = [polar_roots[0], polar_roots[0]]
        constants = (E, L, Q)
        upsilon_r, upsilon_theta, upsilon_phi, gamma = _mino_frequencies_from_constants(
            a, constants, radial_roots, polar_roots
        )

    return upsilon_r, upsilon_theta


def _r_frequency_from_constants(constants, radial_roots):
    """Computes the frequency of motion in r in Mino time.

    Parameters
    ----------
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit in the form
        :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})`
    radial_roots : tuple(double, double, double, double)
        tuple containing the four roots of the radial equation
        :math:`(r_1, r_2, r_3, r_4)`.

    Returns
    -------
    double
    """
    E, L, Q = constants

    r1, r2, r3, r4 = radial_roots
    # equation 13
    k_r = sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))
    # equation 15
    return pi * sqrt((1 - E**2) * (r1 - r3) * (r2 - r4)) / (2 * ellipk(k_r**2))


def _theta_frequency_from_constants(a, constants, radial_roots, polar_roots):
    """Computes the frequency of motion in theta in Mino time.

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit in the form
        :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})`
    radial_roots : tuple(double, double, double, double)
        tuple containing the four roots of the radial polynomial
        :math:`(r_1, r_2, r_3, r_4)`.
    polar_roots : tuple(double, double)
        tuple containing the roots of the polar equation :math:`(z_-,
        z_+)`

    Returns
    -------
    double
    """
    r1, r2, r3, r4 = radial_roots
    z_minus, z_plus = polar_roots
    E, L, Q = constants

    # Schwarzschild case
    if a == 0:
        p = 2 * r1 * r2 / (r1 + r2)
        e = (r1 - r2) / (r1 + r2)
        return p / sqrt(-3 - e**2 + p) * (sign(L) if e == 0 else 1)

    # equation 13
    k_theta = sqrt(z_minus / z_plus)
    # simplified form of epsilon0*z_plus
    e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))

    # equation 15
    return pi * L * sqrt(e0zp) / (2 * ellipk(k_theta**2))


def _phi_frequency_from_constants(
    a, constants, radial_roots, polar_roots, upsilon_r=None, upsilon_theta=None
):
    """Computes the frequency of motion in phi in Mino time.

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit in the form
        :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})`
    radial_roots : tuple(double, double, double, double)
        tuple containing the four roots of the radial polynomial
        :math:`(r_1, r_2, r_3, r_4)`.
    polar_roots : tuple(double, double)
        tuple containing the roots of the polar equation :math:`(z_-, z_+)`
    upsilon_r : double, optional
        Mino frequency of motion in r can be passed in to speed
        computation if it is already known
    upsilon_theta : double, optional
        Mino frequency of motion in theta can be passed in to speed
        computation if it is already known

    Returns
    -------
    double
    """

    E, L, Q = constants
    r1, r2, r3, r4 = radial_roots
    z_minus, z_plus = polar_roots

    # Schwarzschild case
    if a == 0:
        p = 2 * r1 * r2 / (r1 + r2)
        e = (r1 - r2) / (r1 + r2)
        return sign(L) * p / sqrt(-3 - e**2 + p)

    # compute frequencies if they are not passed in
    if upsilon_r is None:
        upsilon_r = _r_frequency_from_constants(a, constants, radial_roots)
    if upsilon_theta is None:
        upsilon_theta = _theta_frequency_from_constants(a, constants, radial_roots, polar_roots)

    # simplified form of epsilon0*z_plus
    e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))

    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)

    h_plus = (r1 - r2) * (r3 - r_plus) / ((r1 - r3) * (r2 - r_plus))
    h_minus = (r1 - r2) * (r3 - r_minus) / ((r1 - r3) * (r2 - r_minus))

    # equation 13
    k_theta = sqrt(z_minus / z_plus)
    k_r = sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))

    # equation 21
    return 2 * upsilon_theta / (pi * sqrt(e0zp)) * _ellippi(
        z_minus, k_theta**2
    ) + 2 * a * upsilon_r / (
        pi * (r_plus - r_minus) * sqrt((1 - E**2) * (r1 - r3) * (r2 - r4))
    ) * (
        (2 * E * r_plus - a * L)
        / (r3 - r_plus)
        * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_plus) * _ellippi(h_plus, k_r**2))
        - (2 * E * r_minus - a * L)
        / (r3 - r_minus)
        * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_minus) * _ellippi(h_minus, k_r**2))
    )


def _gamma_from_constants(
    a, constants, radial_roots, polar_roots, upsilon_r=None, upsilon_theta=None
):
    """Computes the average rate at which observer time accumulates in Mino time.

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit in the form
        :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})`
    radial_roots : tuple(double, double, double, double)
        tuple containing the four roots of the radial polynomial
        :math:`(r_1, r_2, r_3, r_4)`.
    polar_roots : tuple(double, double)
        tuple containing the roots of the polar equation :math:`(z_-, z_+)`
    upsilon_r : double, optional
        Mino frequency of motion in r can be passed in to speed
        computation if it is already known
    upsilon_theta : double, optional
        Mino frequency of motion in theta can be passed in to speed
        computation if it is already known

    Returns
    -------
    double
    """
    r1, r2, r3, r4 = radial_roots
    z_minus, z_plus = polar_roots

    e = (r1 - r2) / (r1 + r2)
    # marginally bound case
    if e == 1:
        return inf

    E, L, Q = constants

    epsilon0 = a**2 * (1 - E**2) / L**2
    # simplified form of a**2*sqrt(z_plus/epsilon0)
    a2sqrt_zp_over_e0 = (
        L**2 / ((1 - E**2) * sqrt(1 - z_minus))
        if a == 0
        else a**2 * z_plus / sqrt(epsilon0 * z_plus)
    )

    # compute frequencies if they are not passed in
    if upsilon_r is None:
        upsilon_r = _r_frequency_from_constants(a, constants, radial_roots)
    if upsilon_theta is None:
        upsilon_theta = _theta_frequency_from_constants(a, constants, radial_roots, polar_roots)

    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)

    h_r = (r1 - r2) / (r1 - r3)
    h_plus = (r1 - r2) * (r3 - r_plus) / ((r1 - r3) * (r2 - r_plus))
    h_minus = (r1 - r2) * (r3 - r_minus) / ((r1 - r3) * (r2 - r_minus))

    # equation 13
    k_theta = 0 if a == 0 else sqrt(z_minus / z_plus)
    k_r = sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))

    # equation 21
    return (
        4 * E
        + 2
        * a2sqrt_zp_over_e0
        * E
        * upsilon_theta
        * (ellipk(k_theta**2) - ellipe(k_theta**2))
        / (pi * L)
        + 2
        * upsilon_r
        / (pi * sqrt((1 - E**2) * (r1 - r3) * (r2 - r4)))
        * (
            E
            / 2
            * (
                (r3 * (r1 + r2 + r3) - r1 * r2) * ellipk(k_r**2)
                + (r2 - r3) * (r1 + r2 + r3 + r4) * _ellippi(h_r, k_r**2)
                + (r1 - r3) * (r2 - r4) * ellipe(k_r**2)
            )
            + 2 * E * (r3 * ellipk(k_r**2) + (r2 - r3) * _ellippi(h_r, k_r**2))
            + 2
            / (r_plus - r_minus)
            * (
                ((4 * E - a * L) * r_plus - 2 * a**2 * E)
                / (r3 - r_plus)
                * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_plus) * _ellippi(h_plus, k_r**2))
                - ((4 * E - a * L) * r_minus - 2 * a**2 * E)
                / (r3 - r_minus)
                * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_minus) * _ellippi(h_minus, k_r**2))
            )
        )
    )


def _mino_frequencies_from_constants(a, constants, radial_roots, polar_roots):
    r"""Computes frequencies of orbital motion in Mino time.

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit in the form
        :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})`
    radial_roots : tuple(double, double, double, double)
        tuple containing the four roots of the radial polynomial
        :math:`(r_1, r_2, r_3, r_4)`.
    polar_roots : tuple(double, double)
        tuple containing the roots of the polar equation :math:`(z_-, z_+)`

    Returns
    -------
    tuple(double, double, double, double)
        tuple of frequencies in the form :math:`(\Upsilon_r,
        \Upsilon_\theta, \Upsilon_\phi, \Gamma)`
    """
    upsilon_r = _r_frequency_from_constants(constants, radial_roots)
    upsilon_theta = _theta_frequency_from_constants(a, constants, radial_roots, polar_roots)
    upsilon_phi = _phi_frequency_from_constants(
        a, constants, radial_roots, polar_roots, upsilon_r, upsilon_theta
    )
    Gamma = _gamma_from_constants(
        a, constants, radial_roots, polar_roots, upsilon_r, upsilon_theta
    )

    return upsilon_r, abs(upsilon_theta), upsilon_phi, Gamma


def _fundamental_frequencies_from_constants(a, constants, radial_roots, polar_roots):
    r"""Computes frequencies of orbital motion in Boyer-Lindquist time.

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole (must satisfy -1 < a < 1)
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit in the form
        :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})`
    radial_roots : tuple(double, double, double, double)
        tuple containing the four roots of the radial polynomial
        :math:`(r_1, r_2, r_3, r_4)`.
    polar_roots : tuple(double, double)
        tuple containing the roots of the polar equation :math:`(z_-, z_+)`

    Returns
    -------
    tuple(double, double, double)
        tuple of frequencies in the form :math:`(\Omega_r,
        \Omega_\theta, \Omega_\phi)`
    """
    upsilon_r = _r_frequency_from_constants(constants, radial_roots)
    upsilon_theta = _theta_frequency_from_constants(a, constants, radial_roots, polar_roots)
    upsilon_phi = _phi_frequency_from_constants(
        a, constants, radial_roots, polar_roots, upsilon_r, upsilon_theta
    )
    Gamma = _gamma_from_constants(
        a, constants, radial_roots, polar_roots, upsilon_r, upsilon_theta
    )

    return upsilon_r / Gamma, abs(upsilon_theta) / Gamma, upsilon_phi / Gamma
