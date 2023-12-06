"""Module containing functions to compute orbital properties from initial conditions"""
from .frequencies import (
    _r_frequency_from_constants,
    _theta_frequency_from_constants,
    plunging_mino_frequencies,
)
from .constants import plunging_radial_roots
import numpy as np
from numpy import sin, cos, sqrt, pi, arcsin, arccos
from numpy.polynomial import Polynomial
from scipy.special import ellipkinc


def constants_from_initial_conditions(a, initial_position, initial_velocity):
    r"""Computes the constants of motion :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})` 
    of the orbit with the given initial conditions.

    Parameters
    ----------
    a : double
        spin parameter
    initial_position : tuple(double,double,double,double)
        initial position :math:`(t_0,r_0,\theta_0,\phi_0)`
    initial_velocity : tuple(double,double,double,double)
        initial four-velocity :math:`(u^t_0,u^r_0,u^\theta_0,u^\phi_0)`

    Returns
    -------
    tuple(double,double,double)
        tuple of constants of motion :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})`
    """

    t0, r0, theta0, phi0 = initial_position
    dt0, dr0, dtheta0, dphi0 = initial_velocity

    sigma = r0**2 + a**2 * cos(theta0) ** 2
    delta = r0**2 - 2 * r0 + a**2

    # solve for E and L by writing the t and phi geodesic equations as a matrix equation
    A = np.array(
        [
            [
                (r0**2 + a**2) ** 2 / delta - a**2 * (1 - cos(theta0) ** 2),
                a - a * (r0**2 + a**2) / delta,
            ],
            [a * (r0**2 + a**2) / delta - a, 1 / (1 - cos(theta0) ** 2) - a**2 / delta],
        ]
    )
    b = np.array([sigma * dt0, sigma * dphi0])
    E, L = np.linalg.solve(A, b)

    # solve for Q by substituting E and L back into the theta equation
    Q = (
        (sigma * dtheta0 * sin(theta0)) ** 2
        + cos(theta0) ** 2 * (a**2 * (1 - E**2) * (1 - cos(theta0) ** 2) + L**2)
    ) / (1 - cos(theta0) ** 2)

    return E, L, Q


def is_stable(a, initial_position, initial_velocity, constants=None, tol=1e-4):
    r"""Tests whether the orbit with the given initial conditions is stable

    Parameters
    ----------
    a : double
        spin parameter
    initial_position : tuple(double,double,double,double)
        initial position :math:`(t_0,r_0,\theta_0,\phi_0)`
    initial_velocity : tuple(double,double,double,double)
        initial four-velocity :math:`(u^t_0,u^r_0,u^\theta_0,u^\phi_0)`
    constants : tuple(double,double,double), optional
        tuple of constants of motion :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})` 
        can be passed in to avoid recomputing them
    tol : double, optional
        numerical tolerance used when comparing :math:`r_0` to radial
        roots, defaults to 1e-4

    Returns
    -------
    bool
        True if the orbit is stable, False otherwise
    """

    t0, r0, theta0, phi0 = initial_position
    # recompute constants if they are not passed in
    if constants is None:
        constants = constants_from_initial_conditions(a, initial_position, initial_velocity)
    E, L, Q = constants

    # radial polynomial
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

    # get the real roots and the complex roots
    real_roots = np.sort(np.real(radial_roots[np.isreal(radial_roots)]))
    complex_roots = radial_roots[np.iscomplex(radial_roots)]

    # event horizon radius
    r_minus = 1 - sqrt(1 - a**2)

    if len(complex_roots) == 4:
        raise ValueError("Not a physical orbit")

    if len(complex_roots) == 2:
        return False

    r4, r3, r2, r1 = real_roots
    # r4 < r3 < [r2 < r_minus < r_plus < r1]
    if (r2 < r_minus) & (r0 <= r1 + tol) & (r0 >= r2 - tol):
        return False
    # r4 < r_minus < r_plus < r3 < [r2 < r1]
    if (r2 > r_minus) & (r0 <= r1 + tol) & (r0 >= r2 - tol):
        return True
    # [r4 < r_minus < r_plus < r3] < r2 < r1
    if (r2 > r_minus) & (r0 <= r3 + tol) & (r0 >= r4 - tol):
        return False

    raise ValueError("Not a physical orbit")


def stable_orbit_initial_phases(
    a,
    initial_position,
    initial_velocity,
    constants=None,
    radial_roots=None,
    upsilon_r=None,
    upsilon_theta=None,
    tol=1e-4,
):
    r"""Computes the initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})` 
    of a stable bound orbit with the given initial conditions. Computes phases 
    with respect to the starting point :math:`(t_0,r_0,\theta_0,\phi_0) = 
    (0,r_\text{min},\theta_\text{min},0)`.

    Parameters
    ----------
    a : double
        spin parameter
    initial_position : tuple(double,double,double,double)
        initial position :math:`(t_0,r_0,\theta_0,\phi_0)`
    initial_velocity : tuple(double,double,double,double)
        initial four-velocity :math:`(u^t_0,u^r_0,u^\theta_0,u^\phi_0)`
    constants : tuple(double,double,double), optional
        constants of motion :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})` 
        can be passed in to avoid recomputing them
    radial_roots : tuple(double,double,double,double), optional
        radial roots :math:`(r_1,r_2,r_3,r_4)` can be passed in to avoid
        recomputing them
    upsilon_r : double, optional
        radial frequency :math:`\Upsilon_r` can be passed in to avoid
        recomputing it
    upsilon_theta : double, optional
        polar frequency :math:`\Upsilon_\theta` can be passed in to
        avoid recomputing it
    tol : double, optional
        numerical tolerance used when checking for turning points,
        defaults to 1e-4

    Returns
    -------
    tuple(double,double,double,double)
        tuple of initial phases
        :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
    """
    t0, r0, theta0, phi0 = initial_position
    dt0, dr0, dtheta0, dphi0 = initial_velocity

    # recompute constants if they are not passed in
    if constants is None:
        constants = constants_from_initial_conditions(a, initial_position, initial_velocity)
    E, L, Q = constants

    # polar polynomial written in terms of z = cos^2(theta)
    Z = Polynomial([Q, -(Q + a**2 * (1 - E**2) + L**2), a**2 * (1 - E**2)])
    polar_roots = Z.roots()
    if a == 0:
        polar_roots = [polar_roots[0], polar_roots[0]]
    z_minus, z_plus = polar_roots

    # recompute radial roots if they are not passed in
    if radial_roots is None:
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
    else:
        r1, r2, r3, r4 = radial_roots

    # check for turning points in r
    if abs(r0 - r2) < tol:
        q_r0 = 0
    elif abs(r0 - r1) < tol:
        q_r0 = pi
    else:
        # compute r frequency
        if upsilon_r is None:
            upsilon_r = _r_frequency_from_constants(constants, radial_roots)

        # Fujita and Hikida equation 13
        k_r = sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))
        y_r = sqrt((r1 - r3) * (r0 - r2) / ((r1 - r2) * (r0 - r3)))
        # Fujita and Hikida equation 26
        lambda_r0 = (
            1
            / sqrt(1 - E**2)
            * 2
            / sqrt((r1 - r3) * (r2 - r4))
            * ellipkinc(arcsin(y_r), k_r**2)
        )

        # Fujita and Hikida equation 24
        q_r0 = lambda_r0 * upsilon_r if dr0 > 0 else 2 * pi - lambda_r0 * upsilon_r

    # Fujita and Hikida equation 11
    theta_min = arccos(sqrt(z_minus))
    theta_max = pi - theta_min

    # check for turning points in theta
    if abs(theta0 - theta_min) < tol:
        q_theta0 = 0
    elif abs(theta0 - theta_max) < tol:
        q_theta0 = pi
    else:
        # compute theta frequency
        if upsilon_theta is None:
            upsilon_theta = _theta_frequency_from_constants(
                a, constants, radial_roots, polar_roots
            )

        k_theta = sqrt(z_minus / z_plus)
        y_theta = cos(theta0) / sqrt(z_minus)
        e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))

        # Fujita and Hikida equation 37
        lambda_theta0 = 1 / (L * sqrt(e0zp)) * ellipkinc(arcsin(y_theta), k_theta**2)

        # Fujita and Hikida equation 36
        if dtheta0 < 0:
            q_theta0 = lambda_theta0 * upsilon_theta + 3 * pi / 2
        if (dtheta0 == 0) & (theta0 == pi / 2):
            q_theta0 = 0  # special case for equatorial orbits
        if dtheta0 >= 0:
            q_theta0 = pi / 2 - lambda_theta0 * upsilon_theta

    q_t0 = t0
    q_phi0 = phi0

    return q_t0, q_r0, q_theta0, q_phi0


def plunging_orbit_initial_phases(
    a, initial_position, initial_velocity, constants=None, radial_roots=None, tol=1e-4
):
    r"""Computes the initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})` 
    of a plunging orbit with the given initial conditions. Computes phases with 
    respect to the starting point :math:`(t_0,r_0,\theta_0,\phi_0) = 
    (0,r_\text{min},\theta_{\text{min}},0)`.

    Parameters
    ----------
    a : double
        spin parameter
    initial_position : tuple(double,double,double,double)
        initial position :math:`(t_0,r_0,\theta_0,\phi_0)`
    initial_velocity : tuple(double,double,double,double)
        initial four-velocity :math:`(u^t_0,u^r_0,u^\theta_0,u^\phi_0)`
    constants : tuple(double,double,double), optional
        constants of motion :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})` 
        can be passed in to avoid recomputing them
    radial_roots : tuple(double,double,double,double), optional
        radial roots :math:`(r_1,r_2,r_3,r_4)` can be passed in to avoid
        recomputing them
    tol : double, optional
        numerical tolerance used when checking for turning points,
        defaults to 1e-4

    Returns
    -------
    tuple(double,double,double,double)
        tuple of initial phases
        :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
    """
    t0, r0, theta0, phi0 = initial_position
    dt0, dr0, dtheta0, dphi0 = initial_velocity

    # recompute constants if they are not passed in
    if constants is None:
        constants = constants_from_initial_conditions(a, initial_position, initial_velocity)
    E, L, Q = constants

    # recompute radial roots if they are not passed in
    if radial_roots is None:
        radial_roots = plunging_radial_roots(a, E, L, Q)
    r1, r2, r3, r4 = radial_roots

    if np.iscomplex(r3):
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
        rho_r = np.real(r3)
        rho_i = np.imag(r4)

        # equation 47
        A = sqrt((r2 - rho_r) ** 2 + rho_i**2)
        B = sqrt((r1 - rho_r) ** 2 + rho_i**2)
        k_r = sqrt(((r2 - r1) ** 2 - (A - B) ** 2) / (4 * A * B))
        # equation 49
        lambda_r0 = (
            -1
            / sqrt((1 - E**2) * A * B)
            * ellipkinc(
                pi / 2 - arcsin((B * (r2 - r0) - A * (r0 - r1)) / (B * (r2 - r0) + A * (r0 - r1))),
                k_r**2,
            )
        )
        # derived by inverting equation 29
        lambda_theta0 = 1 / z2 * ellipkinc(arcsin(cos(theta0) / z1), k_theta**2)

        upsilon_r, upsilon_theta = plunging_mino_frequencies(a, E, L, Q)

        q_t0 = t0
        q_r0 = -lambda_r0 * upsilon_r if dr0 > 0 else 2 * pi + lambda_r0 * upsilon_r

        if dtheta0 < 0:
            q_theta0 = lambda_theta0 * upsilon_theta + 3 * pi / 2
        if (dtheta0 == 0) & (theta0 == pi / 2):
            q_theta0 = 0  # special case for equatorial orbits
        if dtheta0 >= 0:
            q_theta0 = pi / 2 - lambda_theta0 * upsilon_theta

        q_phi0 = phi0

        return q_t0, q_r0, q_theta0, q_phi0
    else:
        return stable_orbit_initial_phases(
            a,
            initial_position,
            initial_velocity,
            constants=constants,
            radial_roots=radial_roots,
            tol=tol,
        )
