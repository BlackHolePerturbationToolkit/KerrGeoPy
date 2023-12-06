"""Module implementing the plunging orbit solutions of `Dyson and van de Meent <https://doi.org/10.48550/arXiv.2302.03704>`_"""
from numpy import sqrt, arctan, arctan2, arccos, log, pi
from numpy.polynomial import Polynomial
import numpy as np
from scipy.special import ellipj, ellipeinc, ellipk
from .frequencies import _mino_frequencies_from_constants, _ellippiinc
from .stable import *
from .orbit import Orbit
from .units import *


class PlungingOrbit(Orbit):
    r"""Class representing a plunging orbit in Kerr spacetime.

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
    initial_phases : tuple, optional
        tuple of initial phases :math:`(q^t_0, q^r_0, q^\theta_0,
        q^\phi_0)`, defaults to (0,0,0,0)
    M : double
        mass of the primary in solar masses, optional
    mu : double
        mass of the smaller body in solar masses, optional

    Attributes
    ----------
    a
        dimensionless spin parameter
    E
        dimensionless energy
    L
        dimensionless angular momentum
    Q
        dimensionless Carter constant
    initial_phases
        tuple of initial phases :math:`(q^t_0, q^r_0, q^\theta_0, q^\phi_0)`
    stable
        boolean indicating whether the orbit is stable
    initial_position
        tuple of initial position coordinates :math:`(t_0, r_0, \theta_0, \phi_0)`
    initial_velocity
        tuple of initial four-velocity components :math:`(u^t_0, u^r_0, u^\theta_0, u^\phi_0)`
    M
        mass of the primary in solar masses
    mu
        mass of the smaller body in solar masses
    upsilon_r
        radial Mino frequency
    upsilon_theta
        polar Mino frequency
    """

    def __init__(self, a, E, L, Q, initial_phases=(0, 0, 0, 0), M=None, mu=None):
        self.a, self.E, self.L, self.Q, self.initial_phases, self.M, self.mu = (
            a,
            E,
            L,
            Q,
            initial_phases,
            M,
            mu,
        )

        if a == 0:
            raise ValueError("Schwarzschild plunges are not currently supported")

        self.stable = False
        self.upsilon_r, self.upsilon_theta = plunging_mino_frequencies(a, E, L, Q)

        u_t, u_r, u_theta, u_phi = self.four_velocity()
        t, r, theta, phi = self.trajectory()
        self.initial_position = t(0), r(0), theta(0), phi(0)
        self.initial_velocity = u_t(0), u_r(0), u_theta(0), u_phi(0)

    def trajectory_deltas(self, initial_phases=None):
        r"""Computes the trajectory deltas :math:`t_r(q_r)`, :math:`t_\theta(q_\theta)`, 
        :math:`\phi_r(q_r)` and :math:`\phi_\theta(q_\theta)`

        Parameters
        ----------
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`

        Returns
        -------
        tuple(function, function, function, function)
            tuple of trajectory deltas :math:`(t_r(q_r),
            t_\theta(q_\theta), \phi_r(q_r),\phi_\theta(q_\theta))`
        """
        if initial_phases is None:
            initial_phases = self.initial_phases
        q_t0, q_r0, q_theta0, q_phi0 = initial_phases

        radial_roots = plunging_radial_roots(self.a, self.E, self.L, self.Q)
        if np.iscomplex(radial_roots[3]):
            # adjust q_theta0 so that initial conditions are consistent with stable orbits
            q_theta0 = q_theta0 + pi / 2
            r, t_r, phi_r = plunging_radial_solutions_complex(self.a, self.E, self.L, self.Q)
            theta, t_theta, phi_theta = plunging_polar_solutions(self.a, self.E, self.L, self.Q)
        else:
            constants = (self.E, self.L, self.Q)
            r, t_r, phi_r = radial_solutions(self.a, constants, radial_roots)
            theta, t_theta, phi_theta = polar_solutions(self.a, constants, radial_roots)

        return (
            lambda q_r: t_r(q_r + q_r0),
            lambda q_theta: t_theta(q_theta + q_theta0),
            lambda q_r: phi_r(q_r + q_r0),
            lambda q_theta: phi_theta(q_theta + q_theta0),
        )

    def trajectory(self, initial_phases=None, distance_units="natural", time_units="natural"):
        r"""Computes the components of the trajectory as a function of mino time

        Parameters
        ----------
        initial_phases : tuple, optional
            tuple of initial phases :math:`(q^t_0, q^r_0, q^\theta_0, q^\phi_0)`
        distance_units : str, optional
            units to compute the radial component of the trajectory in
            (options are "natural", "mks", "cgs", "au" and "km"),
            defaults to "natural"
        time_units : str, optional
            units to compute the time component of the trajectory in
            (options are "natural", "mks", "cgs", and "days"), defaults
            to "natural"

        Returns
        -------
        tuple(function,function,function,function)
            tuple of functions :math:`t(\lambda), r(\lambda),
            \theta(\lambda), \phi(\lambda)`
        """
        if initial_phases is None:
            initial_phases = self.initial_phases
        return plunging_trajectory(
            self.a, self.E, self.L, self.Q, initial_phases, self.M, distance_units, time_units
        )


def plunging_radial_integrals(a, E, L, Q):
    r"""Computes the radial integrals :math:`I_r`, :math:`I_{r^2}` and 
    :math:`I_{r_\pm}` defined in equation 39 of `Dyson and van de Meent 
    <https://doi.org/10.48550/arXiv.2302.03704>`_ as a function of the 
    radial phase. Used to compute the radial solutions for the case of 
    two complex roots.

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
    tuple(function,function,function,function)
        radial integrals :math:`(I_r,I_{r^2},I_{r_+},I_{r_-})`
    """
    r1, r2, r3, r4 = plunging_radial_roots(a, E, L, Q)
    rho_r = np.real(r3)
    rho_i = np.imag(r4)

    # inner and outer horizons
    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)

    # equation 42
    A = sqrt((r2 - rho_r) ** 2 + rho_i**2)
    B = sqrt((r1 - rho_r) ** 2 + rho_i**2)
    f = 4 * A * B / (A - B) ** 2
    k_r = sqrt(((r2 - r1) ** 2 - (A - B) ** 2) / (4 * A * B))

    upsilon_r, upsilon_theta = plunging_mino_frequencies(a, E, L, Q)

    # equation 48
    D_plus = sqrt(4 * A * B * (r2 - r_plus) * (r_plus - r1)) / (
        A * (r_plus - r1) + B * (r2 - r_plus)
    )
    D_minus = sqrt(4 * A * B * (r2 - r_minus) * (r_minus - r1)) / (
        A * (r_minus - r1) + B * (r2 - r_minus)
    )

    def I_r(q_r):
        # equation 43
        sn, cn, dn, xi_r = ellipj(2 * ellipk(k_r**2) * q_r / pi, k_r**2)
        mino_time = q_r / upsilon_r
        # equation 46
        return (
            (A * r1 - B * r2) / (A - B) * mino_time
            - 1 / sqrt(1 - E**2) * arctan((r2 - r1) * sn / (2 * sqrt(A * B) * dn))
            + (A + B)
            * (r2 - r1)
            / (2 * (A - B) * sqrt(A * B * (1 - E**2)))
            * _ellippiinc(xi_r, -1 / f, k_r**2)
        )

    def I_r2(q_r):
        # equation 43
        sn, cn, dn, xi_r = ellipj(2 * ellipk(k_r**2) * q_r / pi, k_r**2)
        mino_time = q_r / upsilon_r
        # equation 47
        return (
            (A * r1**2 - B * r2**2) / (A - B) * mino_time
            + sqrt(A * B) / sqrt(1 - E**2) * ellipeinc(xi_r, k_r**2)
            - (A + B)
            * (A**2 + 2 * r1**2 - B**2 - 2 * r2**2)
            / (4 * (A - B) * sqrt((1 - E**2) * A * B))
            * _ellippiinc(xi_r, -1 / f, k_r**2)
            - sqrt(A * B)
            * (A + B - (A - B) * cn)
            * sn
            * dn
            / ((A - B) * sqrt(1 - E**2) * (f + sn**2))
            + (A**2 + 2 * r1**2 - B**2 - 2 * r2**2)
            / (4 * (r2 - r1) * sqrt(1 - E**2))
            * arctan2(
                2 * sn * dn * sqrt(f * (1 + f * k_r**2)), f - (1 + 2 * f * k_r**2) * sn**2
            )
        )

    def I_r_plus(q_r):
        # equation 43
        sn, cn, dn, xi_r = ellipj(2 * ellipk(k_r**2) * q_r / pi, k_r**2)
        mino_time = q_r / upsilon_r
        # equation 48
        return (
            (A - B) * mino_time / (A * (r1 - r_plus) - B * (r2 - r_plus))
            + (r2 - r1)
            * (A * (r1 - r_plus) + B * (r2 - r_plus))
            / (
                2
                * sqrt(A * B * (1 - E**2))
                * (r_plus - r1)
                * (r2 - r_plus)
                * (A * (r1 - r_plus) - B * (r2 - r_plus))
            )
            * _ellippiinc(xi_r, 1 / D_plus**2, k_r**2)
            - (
                sqrt(r2 - r1)
                * log(
                    (
                        (D_plus * sqrt(1 - D_plus**2 * k_r**2) + dn * sn) ** 2
                        + (k_r * (D_plus**2 - sn**2)) ** 2
                    )
                    / (
                        (D_plus * sqrt(1 - D_plus**2 * k_r**2) - dn * sn) ** 2
                        + (k_r * (D_plus**2 - sn**2)) ** 2
                    )
                )
                / (
                    4
                    * sqrt((1 - E**2) * (r2 - r_plus) * (r_plus - r1))
                    * sqrt(
                        A**2 * (r_plus - r1)
                        - (r2 - r_plus) * (r1**2 - B**2 + r2 * r_plus - r1 * (r2 + r_plus))
                    )
                )
            )
        )

    def I_r_minus(q_r):
        # equation 43
        sn, cn, dn, xi_r = ellipj(2 * ellipk(k_r**2) * q_r / pi, k_r**2)
        mino_time = q_r / upsilon_r
        # equation 48
        return (
            (A - B) * mino_time / (A * (r1 - r_minus) - B * (r2 - r_minus))
            + (r2 - r1)
            * (A * (r1 - r_minus) + B * (r2 - r_minus))
            / (
                2
                * sqrt(A * B * (1 - E**2))
                * (r_minus - r1)
                * (r2 - r_minus)
                * (A * (r1 - r_minus) - B * (r2 - r_minus))
            )
            * _ellippiinc(xi_r, 1 / D_minus**2, k_r**2)
            - (
                sqrt(r2 - r1)
                * log(
                    (
                        (D_minus * sqrt(1 - D_minus**2 * k_r**2) + dn * sn) ** 2
                        + (k_r * (D_minus**2 - sn**2)) ** 2
                    )
                    / (
                        (D_minus * sqrt(1 - D_minus**2 * k_r**2) - dn * sn) ** 2
                        + (k_r * (D_minus**2 - sn**2)) ** 2
                    )
                )
                / (
                    4
                    * sqrt((1 - E**2) * (r2 - r_minus) * (r_minus - r1))
                    * sqrt(
                        A**2 * (r_minus - r1)
                        - (r2 - r_minus) * (r1**2 - B**2 + r2 * r_minus - r1 * (r2 + r_minus))
                    )
                )
            )
        )

    return I_r, I_r2, I_r_plus, I_r_minus


def plunging_radial_solutions_complex(a, E, L, Q):
    r"""Computes the radial solutions :math:`r(q_r), t_r(q_r), \phi_r(q_r)` 
    from equation 50 and 51 of `Dyson and van de Meent <https://doi.org/10.48550
    /arXiv.2302.03704>`_ for the case of two complex radial roots.

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
    tuple(function, function, function)
        tuple of radial solutions :math:`(r(q_r), t_r(q_r), \phi_r(q_r))`
    """
    roots = plunging_radial_roots(a, E, L, Q)
    if np.iscomplex(roots).sum() != 2:
        raise ValueError("There should be two complex roots")

    r1, r2, r3, r4 = roots  # r1 < r2 are real and r3/r4 are complex conjugates
    rho_r = np.real(r3)
    rho_i = np.imag(r4)

    # inner and outer horizons
    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)

    # equation 42
    A = sqrt((r2 - rho_r) ** 2 + rho_i**2)
    B = sqrt((r1 - rho_r) ** 2 + rho_i**2)
    k_r = sqrt(((r2 - r1) ** 2 - (A - B) ** 2) / (4 * A * B))

    upsilon_r, upsilon_theta = plunging_mino_frequencies(a, E, L, Q)

    I_r, I_r2, I_r_plus, I_r_minus = plunging_radial_integrals(a, E, L, Q)

    def r(q_r):
        # equation 43
        sn, cn, dn, xi_r = ellipj(2 * ellipk(k_r**2) * q_r / pi, k_r**2)
        # equation 49
        return (
            (A - B) * (A * r1 - B * r2) * sn**2
            + 2 * A * B * (r1 + r2)
            - 2 * A * B * (r2 - r1) * cn
        ) / (4 * A * B + (A - B) ** 2 * sn**2)

    def t_r(q_r):
        mino_time = q_r / upsilon_r
        # equation 41
        return (
            E * (r_plus**2 + r_minus**2 + r_plus * r_minus + 2 * a**2) * mino_time
            + E * (I_r2(q_r) + I_r(q_r) * (r_minus + r_plus))
            + (
                (r_minus**2 + a**2)
                * (E * (r_minus**2 + a**2) - a * L)
                / (r_minus - r_plus)
                * I_r_minus(q_r)
                + (r_plus**2 + a**2)
                * (E * (r_plus**2 + a**2) - a * L)
                / (r_plus - r_minus)
                * I_r_plus(q_r)
            )
        )

    def phi_r(q_r):
        mino_time = q_r / upsilon_r
        # equation 40
        return a * (
            (E * (r_minus**2 + a**2) - a * L) / (r_minus - r_plus) * I_r_minus(q_r)
            + (E * (r_plus**2 + a**2) - a * L) / (r_plus - r_minus) * I_r_plus(q_r)
        )

    return r, t_r, phi_r


def plunging_polar_solutions(a, E, L, Q):
    r"""Computes the polar solutions :math:`\theta(q_\theta), t_\theta(q_\theta), 
    \phi_\theta(q_\theta)` from equation 33 and 37 of `Dyson and van de Meent 
    <https://doi.org/10.48550/arXiv.2302.03704>`_

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
    tuple(function, function, function)
        tuple of polar solutions :math:`(\theta(q_\theta),
        t_\theta(q_\theta), \phi_\theta(q_\theta))`
    """
    z1 = sqrt(
        1
        / 2
        * (
            1
            + (L**2 + Q) / (a**2 * (1 - E**2))
            - sqrt(
                (1 + (L**2 + Q) / (a**2 * (1 - E**2))) ** 2 - 4 * Q / (a**2 * (1 - E**2))
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
                (1 + (L**2 + Q) / (a**2 * (1 - E**2))) ** 2 - 4 * Q / (a**2 * (1 - E**2))
            )
        )
    )
    k_theta = a * sqrt(1 - E**2) * z1 / z2

    upsilon_r, upsilon_theta = plunging_mino_frequencies(a, E, L, Q)

    def theta(q_theta):
        mino_time = q_theta / upsilon_theta
        # equation 28
        sn, cn, dn, xi_theta = ellipj(z2 * mino_time, k_theta**2)
        # equation 27
        return arccos(z1 * sn)

    def t_theta(q_theta):
        mino_time = q_theta / upsilon_theta
        # equation 28
        sn, cn, dn, xi_theta = ellipj(z2 * mino_time, k_theta**2)
        # equation 35
        return (
            E
            / (1 - E**2)
            * (
                (z2**2 - a**2 * (1 - E**2)) * mino_time
                - z2 * ellipeinc(xi_theta, k_theta**2)
            )
        )

    def phi_theta(q_theta):
        mino_time = q_theta / upsilon_theta
        # equation 28
        sn, cn, dn, xi_theta = ellipj(z2 * mino_time, k_theta**2)
        # equation 31
        return L / z2 * _ellippiinc(xi_theta, z1**2, k_theta**2)

    return theta, t_theta, phi_theta


def plunging_trajectory(
    a, E, L, Q, initial_phases=(0, 0, 0, 0), M=None, distance_units="natural", time_units="natural"
):
    r"""Computes the components of the trajectory as a function of mino time for a plunging orbit 
    with the given parameters.

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
    initial_phases : tuple, optional
        tuple of initial phases :math:`(q^t_0, q^r_0, q^\theta_0, q^\phi_0)`
    M : double, optional
        mass of the primary in solar masses
    distance_units : str, optional
        units to compute the radial component of the trajectory in
        (options are "natural", "mks", "cgs", "au" and "km"), defaults
        to "natural"
    time_units : str, optional
        units to compute the time component of the trajectory in
        (options are "natural", "mks", "cgs", and "days"), defaults to
        "natural"

    Returns
    -------
    tuple(function,function,function,function)
        components of the trajectory :math:`t(\lambda), r(\lambda),
        \theta(\lambda), \phi(\lambda)`
    """
    radial_roots = plunging_radial_roots(a, E, L, Q)

    if np.iscomplex(radial_roots[3]):
        return _complex_plunge_trajectory(
            a, E, L, Q, initial_phases, M, distance_units, time_units
        )
    else:
        return _real_plunge_trajectory(a, E, L, Q, initial_phases, M, distance_units, time_units)


def _complex_plunge_trajectory(
    a, E, L, Q, initial_phases=(0, 0, 0, 0), M=None, distance_units="natural", time_units="natural"
):
    r"""Computes the components of the trajectory for a plunging orbit with the given parameters 
    assuming that the radial polynomial has two complex roots and two real roots.

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
    initial_phases : tuple, optional
        tuple of initial phases :math:`(q^t_0, q^r_0, q^\theta_0, q^\phi_0)`
    M : double, optional
        mass of the primary in solar masses
    distance_units : str, optional
        units to compute the radial component of the trajectory in
        (options are "natural", "mks", "cgs", "au" and "km"), defaults
        to "natural"
    time_units : str, optional
        units to compute the time component of the trajectory in
        (options are "natural", "mks", "cgs", and "days"), defaults to
        "natural"

    Returns
    -------
    tuple(function,function,function,function)
        components of the trajectory :math:`t(\lambda), r(\lambda),
        \theta(\lambda), \phi(\lambda)`
    """
    if ((distance_units != "natural") or (time_units != "natural")) and M is None:
        raise ValueError("M must be specified to convert to physical units")

    upsilon_r, upsilon_theta = plunging_mino_frequencies(a, E, L, Q)
    r_phases, t_r, phi_r = plunging_radial_solutions_complex(a, E, L, Q)
    theta_phases, t_theta, phi_theta = plunging_polar_solutions(a, E, L, Q)
    q_t0, q_r0, q_theta0, q_phi0 = initial_phases

    distance_conversion_func = {
        "natural": lambda x, M: x,
        "mks": distance_in_meters,
        "cgs": distance_in_cm,
        "au": distance_in_au,
        "km": distance_in_km,
    }
    time_conversion_func = {
        "natural": lambda x, M: x,
        "mks": time_in_seconds,
        "cgs": time_in_seconds,
        "days": time_in_days,
    }

    # adjust q_theta0 so that initial conditions are consistent with stable orbits
    q_theta0 = q_theta0 + pi / 2

    # Calculate normalization constants so that t = 0 and phi = 0 at lambda = 0 when q_t0 = 0 and q_phi0 = 0
    C_t = t_r(q_r0) + t_theta(q_theta0)
    C_phi = phi_r(q_r0) + phi_theta(q_theta0)

    def t(mino_time):
        return time_conversion_func[time_units](
            t_r(upsilon_r * mino_time + q_r0)
            + t_theta(upsilon_theta * mino_time + q_theta0)
            - C_t
            + q_t0,
            M,
        )

    def r(mino_time):
        return distance_conversion_func[distance_units](r_phases(upsilon_r * mino_time + q_r0), M)

    def theta(mino_time):
        return theta_phases(upsilon_theta * mino_time + q_theta0)

    def phi(mino_time):
        return (
            phi_r(upsilon_r * mino_time + q_r0)
            + phi_theta(upsilon_theta * mino_time + q_theta0)
            - C_phi
            + q_phi0
        )

    return t, r, theta, phi


def _real_plunge_trajectory(
    a, E, L, Q, initial_phases=(0, 0, 0, 0), M=None, distance_units="natural", time_units="natural"
):
    r"""Computes the components of the trajectory for a plunging orbit with the given parameters 
    assuming that there are four real roots of the radial polynomial.

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
    initial_phases : tuple, optional
        tuple of initial phases :math:`(q^t_0, q^r_0, q^\theta_0, q^\phi_0)`
    M : double, optional
        mass of the primary in solar masses
    distance_units : str, optional
        units to compute the radial component of the trajectory in
        (options are "natural", "mks", "cgs", "au" and "km"), defaults
        to "natural"
    time_units : str, optional
        units to compute the time component of the trajectory in
        (options are "natural", "mks", "cgs", and "days"), defaults to
        "natural"

    Returns
    -------
    tuple(function,function,function,function)
        components of the trajectory :math:`t(\lambda), r(\lambda),
        \theta(\lambda), \phi(\lambda)`
    """
    if ((distance_units != "natural") or (time_units != "natural")) and M is None:
        raise ValueError("M must be specified to convert to physical units")

    constants = (E, L, Q)
    radial_roots = plunging_radial_roots(a, E, L, Q)

    # polar polynomial written in terms of z = cos^2(theta)
    Z = Polynomial([Q, -(Q + a**2 * (1 - E**2) + L**2), a**2 * (1 - E**2)])
    polar_roots = Z.roots()
    if a == 0:
        polar_roots = [polar_roots[0], polar_roots[0]]

    upsilon_r, upsilon_theta, upsilon_phi, gamma = _mino_frequencies_from_constants(
        a, constants, radial_roots, polar_roots
    )
    r_phases, t_r, phi_r = radial_solutions(a, constants, radial_roots)
    theta_phases, t_theta, phi_theta = polar_solutions(a, constants, polar_roots)
    q_t0, q_r0, q_theta0, q_phi0 = initial_phases

    distance_conversion_func = {
        "natural": lambda x, M: x,
        "mks": distance_in_meters,
        "cgs": distance_in_cm,
        "au": distance_in_au,
        "km": distance_in_km,
    }
    time_conversion_func = {
        "natural": lambda x, M: x,
        "mks": time_in_seconds,
        "cgs": time_in_seconds,
        "days": time_in_days,
    }

    # Calculate normalization constants so that t = 0 and phi = 0 at lambda = 0 when q_t0 = 0 and q_phi0 = 0
    C_t = t_r(q_r0) + t_theta(q_theta0)
    C_phi = phi_r(q_r0) + phi_theta(q_theta0)

    def t(mino_time):
        # equation 6
        return time_conversion_func[time_units](
            q_t0
            + gamma * mino_time
            + t_r(upsilon_r * mino_time + q_r0)
            + t_theta(upsilon_theta * mino_time + q_theta0)
            - C_t,
            M,
        )

    def r(mino_time):
        return distance_conversion_func[distance_units](r_phases(upsilon_r * mino_time + q_r0), M)

    def theta(mino_time):
        return theta_phases(upsilon_theta * mino_time + q_theta0)

    def phi(mino_time):
        # equation 6
        return (
            q_phi0
            + upsilon_phi * mino_time
            + phi_r(upsilon_r * mino_time + q_r0)
            + phi_theta(upsilon_theta * mino_time + q_theta0)
            - C_phi
        )

    return t, r, theta, phi
