"""Module containing the LightOrbit class for computing null geodesics in Kerr spacetime, as well as DistantLightOrbit for computing null geodesics with infinite initial radial coordinate"""
from .frequencies import _ellippiinc
from .spacetime import KerrSpacetime
from .plot_utils import plot, is_visible, animate
from .units import mass_in_kg, time_in_seconds, time_in_days, distance_in_meters, distance_in_cm, distance_in_au, distance_in_km

from math import e, pi
import numpy as np
from scipy.special import ellipj, ellipk, ellipkinc, ellipeinc

from os import environ

def _ellipj(u, m):
    r"""Extension of scipy's ellipj for m < 0.
 
    Parameters
    ----------
    u : double
        argument of the elliptic functions
    m : double
        parameter of the elliptic functions

    Returns
    -------
        (sn, cn, dn, am): tuple(double, double, double, double)
    """
    u, m = np.broadcast_arrays(u, m)

    dtype = np.result_type(u, m, np.float64)
    sn = np.empty(u.shape, dtype=dtype)
    cn = np.empty(u.shape, dtype=dtype)
    dn = np.empty(u.shape, dtype=dtype)
    am = np.empty(u.shape, dtype=dtype)

    mask_lt0 = m < 0
    if np.any(mask_lt0):
        ult0 = u[mask_lt0]
        mlt0 = m[mask_lt0]

        sn0, cn0, dn0, _ = ellipj((1 - mlt0) ** 0.5 * ult0, mlt0 / (mlt0 - 1))
        sn[mask_lt0] = sn0 / dn0 / (1 - mlt0) ** 0.5
        cn[mask_lt0] = cn0 / dn0
        dn[mask_lt0] = 1 / dn0
        am[mask_lt0] = ellipj((1 - mlt0) ** 0.5 * ult0 + ellipk(mlt0 / (mlt0 - 1)), mlt0 / (mlt0 - 1))[3] - np.pi / 2
    
    mask_std = ~mask_lt0
    if np.any(mask_std):
        ustd = u[mask_std]
        mstd = m[mask_std]

        sn0, cn0, dn0, am0 = ellipj(ustd, mstd)

        sn[mask_std] = sn0
        cn[mask_std] = cn0
        dn[mask_std] = dn0
        am[mask_std] = am0

    return (
        sn,
        cn,
        dn,
        am,
    )

def _ellipeinc_prime(phi, m):
    r"""Derivative of the incomplete elliptic integral of the second kind with respect to the parameter m, :math:`\partial E(\varphi | m)/\partial m`.
    Expression from Gralla & Lupsasca equation 32.
    
    Parameters
    ----------
    phi : double
        amplitude of the elliptic integral
    m : double
        parameter of the elliptic integral
    
    Returns
    -------
        double
    """
    return (ellipeinc(phi, m) - ellipkinc(phi, m)) / (2 * m)

def _sc(u, m):
    r"""sc function defined under equation 1.41. Definition as per Gralla & Lupsasca above equation B109.
    
    Parameters
    ----------
    u : double
        argument of the sc function
    m : double
        parameter of the sc function

    Returns
    -------
        double
    """
    sn, cn, _, _ = _ellipj(u, m)
    return sn / cn

def _polar_roots(a, eta, ell):
    r"""Computes the roots of the polar potential (equation 1.15), equation 1.20.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`

    Returns
    -------
        (:math:`u_+`, :math:`u_-`): tuple(double, double)"""
    delta_theta = (1 - (eta + ell ** 2) / a ** 2) / 2
    sqrt_term = (delta_theta ** 2 + eta / a ** 2) ** 0.5
    u_plus  = delta_theta + sqrt_term
    u_minus = delta_theta - sqrt_term

    return u_plus, u_minus

def _ordinary_theta(a, eta, ell, initial_theta, nu_theta, lambda_x=np.inf):
    r"""Computes the polar ordinary motion.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    initial_theta : double
        initial polar angle :math:`\theta_0`
    nu_theta : int
        sign of the initial polar momentum :math:`\nu_\theta = \text{sign}(p^\theta_0)`
    lambda_x : double, optional
        Mino time of capture/escape to infinity, defaults to infinity

    Returns
    -------
        :math:`(\theta(\lambda), G_\phi(\lambda), G_t(\lambda))`: tuple(function, function, function)"""
    u_plus, u_minus = _polar_roots(a, eta, ell)
    m = u_plus / u_minus

    phi = np.asin(np.clip(np.cos(initial_theta) / u_plus ** 0.5, -1, 1))
    if environ.get("KG_DEBUG", "0") == "1":
        print(f"{phi=}, {np.cos(initial_theta) / u_plus ** 0.5}")

    # Equation 1.23
    cal_G_i_theta = -1 / (-u_minus * a ** 2) ** 0.5 * ellipkinc(phi, m)
    cal_G_i_phi   = -1 / (-u_minus * a ** 2) ** 0.5 * _ellippiinc(phi, u_plus, m)
    cal_G_i_t = 2 * u_plus / (-u_minus * a ** 2) ** 0.5 * _ellipeinc_prime(phi, m)

    # Equation 1.24
    def Psi(lambda_):
        return _ellipj((-u_minus * a ** 2) ** 0.5 * (lambda_ + nu_theta * cal_G_i_theta), m)[3]

    # Equation 1.22
    def theta(lambda_):
        return np.where(np.logical_or(lambda_ > lambda_x, lambda_ < 0),
            np.nan,
            np.acos(
                -nu_theta * u_plus ** 0.5 * np.sin(Psi(lambda_))
            )
        )
    
    def G_phi(lambda_):
        nonlimit = lambda lambda__: ell * (1 / (-u_minus * a ** 2) ** 0.5 * _ellippiinc(Psi(lambda__), u_plus, m) - nu_theta * cal_G_i_phi)
        if not np.isclose(ell, 0) and np.isfinite(nonlimit(lambda_x)):
            return np.where(np.logical_or(lambda_ > lambda_x, lambda_ < 0),
                np.nan,
                nonlimit(lambda_)
            )
        # Kernel line 106
        return np.where(np.logical_or(lambda_ > lambda_x, lambda_ < 0),
            np.nan,
            (-1 if ell < 0 else 1) / (-u_minus * a ** 2) ** 0.5 * np.round(Psi(lambda_) / np.pi) * np.pi * (a ** 2 + eta) ** 0.5 / (1 - 1 / u_minus) ** 0.5
        )

    def G_t(lambda_):
        return np.where(np.logical_or(lambda_ > lambda_x, lambda_ < 0),
            np.nan,
            - 2 * u_plus / (-u_minus * a ** 2) ** 0.5 * _ellipeinc_prime(Psi(lambda_), m) - nu_theta * cal_G_i_t
        )
    
    return theta, G_phi, G_t

def _vortical_theta(a, eta, ell, initial_theta, nu_theta, lambda_x=np.inf):
    r"""Computes the polar vortical motion.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    initial_theta : double
        initial polar angle :math:`\theta_0`
    nu_theta : int
        sign of the initial polar momentum :math:`\nu_\theta = \text{sign}(p^\theta_0)`
    lambda_x : double, optional
        Mino time of capture/escape to infinity, defaults to infinity

    Returns
    -------
        :math:`(\theta(\lambda), G_\phi(\lambda), G_t(\lambda))`: tuple(function, function, function)"""
    u_plus, u_minus = _polar_roots(a, eta, ell)
    m = 1 - u_plus / u_minus

    h = np.sign(np.cos(initial_theta))
    # Equation 1.27
    upsilon_i = np.asin(np.clip((np.cos(initial_theta) ** 2 - u_minus) / (u_plus - u_minus), 0, 1) ** 0.5)
    if environ.get("KG_DEBUG", "0") == "1":
        print(f"{upsilon_i=}, {(np.cos(initial_theta) ** 2 - u_minus) / (u_plus - u_minus)}")

    # Equation 1.26
    cal_G_i_theta = -h / (u_minus * a ** 2) ** 0.5 * ellipkinc(upsilon_i, m)
    cal_G_i_phi = -h / ((1 - u_minus) * (u_minus * a ** 2) ** 0.5) * _ellippiinc(upsilon_i, (u_plus - u_minus) / (1 - u_minus), m)
    cal_G_i_t = -h * (u_minus / a ** 2) ** 0.5 * ellipeinc(upsilon_i, m)

    # Equation 1.27
    def upsilon(lambda_):
        return _ellipj((u_minus * a ** 2) ** 0.5 * (lambda_ + nu_theta * cal_G_i_theta), m)[3]

    # equation 1.25
    def theta(lambda_):
        return np.where(np.logical_or(lambda_ > lambda_x, lambda_ < 0),
            np.nan,
            np.acos(
                h * (u_minus + (u_plus - u_minus) * np.sin(upsilon(lambda_)) ** 2) ** 0.5
            )
        )
    
    def G_phi(lambda_):
        return ell * (1 / ((1 - u_minus) * (u_minus * a ** 2) ** 0.5) * _ellippiinc(upsilon(lambda_), (u_plus - u_minus) / (1 - u_minus), m) - nu_theta * cal_G_i_phi)
    
    def G_t(lambda_):
        return (u_minus / a ** 2) ** 0.5 * ellipeinc(upsilon(lambda_), m) - nu_theta * cal_G_i_t

    return theta, G_phi, G_t

def _theta(a, eta, ell, initial_theta, nu_theta, lambda_x=np.inf):
    r"""Computes the polar motion.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    initial_theta : double
        initial polar angle :math:`\theta_0`
    nu_theta : int
        sign of the initial polar momentum :math:`\nu_\theta = \text{sign}(p^\theta_0)`
    lambda_x : double, optional
        Mino time of capture/escape to infinity, defaults to infinity

    Returns
    -------
        :math:`(\theta(\lambda), G_\phi(\lambda), G_t(\lambda))`: tuple(function, function, function)"""
    if environ.get("KG_DEBUG", "0") == "1":
        polar_roots = _polar_roots(a, eta, ell)
        print(f"{polar_roots=}")

    if eta > 0:
        if environ.get("KG_DEBUG", "0") == "1":
            print("ordinary")
        return _ordinary_theta(a, eta, ell, initial_theta, nu_theta, lambda_x)
    else:
        if environ.get("KG_DEBUG", "0") == "1":
            print("vortical")
        return _vortical_theta(a, eta, ell, initial_theta, nu_theta, lambda_x)

def _cuberoot(z):
    r"""Computes the cube root of a number as per the definition under equation 1.31.
    If the input is real, then we return the real cube root.
    If the input is complex, then we return the root with maximal real part.

    Parameters
    ----------
    z : double or complex

    Returns
    -------
        double or complex
    """
    z = np.real_if_close(z)
    if np.isreal(z):
        return np.sign(z) * abs(z) ** (1/3)

    r = z ** (1/3)
    omega = e ** (2j * pi / 3)

    roots = (r, r*omega, r*omega**2)
    return np.real_if_close(max(roots, key=lambda w: w.real))

def _radial_roots(a, eta, ell):
    r"""Computes the roots of the radial potential, equation 1.15.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`

    Returns
    -------
        (:math:`r_1`, :math:`r_2`, :math:`r_3`, :math:`r_4`): tuple(double, double, double, double)"""
    # equation 1.29
    cal_A = a ** 2 - eta - ell ** 2
    cal_B = 2 * (eta + (ell - a) ** 2)
    cal_C = -a ** 2 * eta

    # equation 1.31
    cal_P = -cal_A ** 2 / 12 - cal_C
    cal_Q = -cal_A / 3 * ((cal_A / 6) ** 2 - cal_C) - cal_B ** 2 / 8
    sqrt_term = np.complex128((cal_P / 3) ** 3 + (cal_Q / 2) ** 2) ** 0.5
    omega_plus  = _cuberoot(-cal_Q / 2 + sqrt_term)
    omega_minus = _cuberoot(-cal_Q / 2 - sqrt_term)
    Z = ((omega_plus + omega_minus - cal_A / 3) / 2) ** 0.5

    # equation 1.30
    return (
        np.real_if_close(-Z - np.complex128(-cal_A / 2 - Z ** 2 + cal_B / (4 * Z)) ** 0.5),
        np.real_if_close(-Z + np.complex128(-cal_A / 2 - Z ** 2 + cal_B / (4 * Z)) ** 0.5),
        np.real_if_close( Z - np.complex128(-cal_A / 2 - Z ** 2 - cal_B / (4 * Z)) ** 0.5),
        np.real_if_close( Z + np.complex128(-cal_A / 2 - Z ** 2 - cal_B / (4 * Z)) ** 0.5),
    )

def _horizons(a):
    r"""Computes the horizons of the Kerr spacetime, :math:`r_\pm = 1 \pm \sqrt{1 - a^2}`.

    Parameters
    ----------
    a : double
        dimensionless spin parameter

    Returns
    -------
        (:math:`r_+`, :math:`r_-`): tuple(double, double)"""
    sqrt_term = (1 - a ** 2) ** 0.5
    return 1 + sqrt_term, 1 - sqrt_term

def _radial_potential(a, eta, ell):
    r"""Computes the radial potential as a function of r.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    
    Returns
    -------
    function
        :math:`\mathcal{R}(r)`"""
    def Delta(r):
        return r ** 2 - 2 * r + a ** 2
    def R(r):
        return (r ** 2 + a ** 2 - a * ell) ** 2 - Delta(r) * (eta + (ell - a) ** 2)
    
    return R

def _k(roots):
    r"""Computes the parameter k.

    Parameters
    ----------
    roots : tuple(double, double, double, double)
        roots of the radial potential
    horizons : tuple(double, double)
        horizons of the Kerr spacetime

    Returns
    -------
        k: double
    """
    r_1, r_2, r_3, r_4 = roots
    return (r_3 - r_2) * (r_4 - r_1) / ((r_3 - r_1) * (r_4 - r_2))

def _case_1_radial(a, eta, ell, initial_r, nu_r):
    r"""Computes the radial motion for case 1. Four real roots exist, two are outside horizons,
    considering :math:`r \in (r_+, r_3).

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    initial_r : double
        initial r :math:`r_0`
    nu_r : int
        sign of the initial r momentum :math:`\nu_r = \text{sign}(p^r_0)`.

    Returns
    -------
        :math:`(r(\lambda), I_1, I_2, I_+, I_-, \lambda_x)`: tuple(function, function, function, function, function, double)"""
    radial_roots = _radial_roots(a, eta, ell)
    r_1, r_2, r_3, r_4 = radial_roots
    r_plus, r_minus = _horizons(a)
    k = _k(radial_roots)

    r_21 = r_2 - r_1
    r_31 = r_3 - r_1
    r_32 = r_3 - r_2
    r_42 = r_4 - r_2
    
    # equation 1.34
    x_i = np.clip(r_31 / r_32 * (initial_r - r_2) / (initial_r - r_1), 0, 1) ** 0.5
    if environ.get("KG_DEBUG", "0") == "1":
        print(f"{x_i=}, {r_31 / r_32 * (initial_r - r_2) / (initial_r - r_1)}")
    asin_x_i = np.asin(x_i)
    cal_I_i_0 = 2 / (r_31 * r_42) ** 0.5 * ellipkinc(asin_x_i, k)
    def X(lambda_):
        return (r_31 * r_42) ** 0.5 / 2 * (lambda_ + nu_r * cal_I_i_0)
    def E_1(lambda_):
        return (r_31 * r_42) ** 0.5 * (
            ellipeinc(_ellipj(X(lambda_), k)[3], k)
            - nu_r * ellipeinc(asin_x_i, k)
        )
    def Pi_1(lambda_):
        return 2 / (r_31 * r_42) ** 0.5 * (
            _ellippiinc(_ellipj(X(lambda_), k)[3], r_32 / r_31, k)
            - nu_r * _ellippiinc(asin_x_i, r_32 / r_31, k)
        )
    def Pi_plusminus(r_plusminus):
        return lambda lambda_: 2 / (r_31 * r_42) ** 0.5 * r_21 / ((r_plusminus - r_1) * (r_plusminus - r_2)) * (
            _ellippiinc(_ellipj(X(lambda_), k)[3], (r_plusminus - r_1) * r_32 / ((r_plusminus - r_2) * r_31), k)
            - nu_r * _ellippiinc(asin_x_i, (r_plusminus - r_1) * r_32 / ((r_plusminus - r_2) * r_31), k)
        )

    radial_potential = _radial_potential(a, eta, ell)

    # Kernel line 204
    lambda_x = (
        -2 / (r_31 * r_42) ** 0.5 * ellipkinc(np.asin(np.clip(r_31 * (r_plus - r_2) / (r_32 * (r_plus - r_1)), 0, 1) ** 0.5), k) + cal_I_i_0
    ) if nu_r < 0 else (
        4 / (r_31 * r_42) ** 0.5 * ellipk(k) - 2 / (r_31 * r_42) ** 0.5 * ellipkinc(np.asin(np.clip((r_plus - r_2) * r_31 / ((r_plus - r_1) * r_32), 0, 1) ** 0.5), k) - cal_I_i_0
    )

    # equation 1.35
    def r(lambda_):
        sn_sq = _ellipj(X(lambda_), k)[0] ** 2
        return np.where(np.logical_or(lambda_ > lambda_x, lambda_ < 0),
            np.nan,
            (r_2 * r_31 - r_1 * r_32 * sn_sq) / (r_31 - r_32 * sn_sq)
        )
    def dr(lambda_):
        sn, cn, dn, _ = _ellipj(X(lambda_), k)
        return (
            r_31 * r_32 * r_21 * (r_31 * r_42) ** 0.5 * sn * cn * dn
        ) / (r_31 - r_32 * sn ** 2) ** 2
    def I_1(lambda_):
        return r_1 * lambda_ + r_21 * Pi_1(lambda_)
    def I_2(lambda_):
        return dr(lambda_) / (r(lambda_) - r_1) - nu_r * radial_potential(initial_r) ** 0.5 / (initial_r - r_1) - (r_1 * r_4 + r_2 * r_3) / 2 * lambda_ - E_1(lambda_)
    def I_plusminus(r_plusminus):
        return lambda lambda_: -lambda_ / (r_plusminus - r_1) - Pi_plusminus(r_plusminus)(lambda_)
    
    return r, I_1, I_2, I_plusminus(r_plus), I_plusminus(r_minus), lambda_x
    
def _case_2_radial(a, eta, ell, initial_r, nu_r, distant=False):
    r"""Computes the radial motion for case 2. Four real roots exist.
    Either two are outside horizons while considering :math:`r > r_4` or all are inside horizon.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    initial_r : double
        initial r :math:`r_0`
    nu_r : int
        sign of the initial r momentum :math:`\nu_r = \text{sign}(p^r_0)`.
    distant: bool, optional
        whether the initial position is at infinity, defaults to false

    Returns
    -------
        :math:`(r(\lambda), I_1, I_2, I_+, I_-, \lambda_x)`: tuple(function, function, function, function, function, double)"""
    roots = _radial_roots(a, eta, ell)
    r_1, r_2, r_3, r_4 = roots
    r_plus, r_minus = _horizons(a)
    k = _k(roots)

    r_31 = r_3 - r_1
    r_32 = r_3 - r_2
    r_41 = r_4 - r_1
    r_42 = r_4 - r_2
    r_43 = r_4 - r_3
    
    # equation 1.36
    x_i = np.clip(
        (
            (r_31 * (initial_r - r_4) / (r_41 * (initial_r - r_3)))
        ) if not distant else (r_31 / r_41),
        0, 1
    ) ** 0.5
    if environ.get("KG_DEBUG", "0") == "1":
        print(f"{x_i=}, {(r_31 * (initial_r - r_4) / (r_41 * (initial_r - r_3))) if not distant else r_31 / r_41}")
    asin_x_i = np.asin(x_i)
    cal_I_i_0 = 2 / (r_31 * r_42) ** 0.5 * ellipkinc(asin_x_i, k)
    def X(lambda_):
        return (r_31 * r_42) ** 0.5 / 2 * (lambda_ + nu_r * cal_I_i_0)
    def E_1(lambda_):
        return (r_31 * r_42) ** 0.5 * (ellipeinc(_ellipj(X(lambda_), k)[3], k) - nu_r * ellipeinc(asin_x_i, k))
    def Pi_1(lambda_):
        if not distant:
            return 2 / (r_31 * r_42) ** 0.5 * (_ellippiinc(_ellipj(X(lambda_), k)[3], r_41 / r_31, k) - nu_r * _ellippiinc(asin_x_i, r_41 / r_31, k))
        return 2 / (r_31 * r_42) ** 0.5 * (
            _ellippiinc(_ellipj(X(lambda_), k)[3], r_41 / r_31, k)
            - nu_r * ellipkinc(np.asin(np.clip(r_31 / r_41, 0, 1) ** 0.5), k)
            + nu_r * _ellippiinc(np.asin(np.clip(r_31 / r_41, 0, 1) ** 0.5), r_32 / r_42, k)
            - nu_r * 1 / (2 * ((1 - r_32 / r_42) * (r_41 / r_31 - 1)) ** 0.5) * np.log(4 / (r_31 + r_42))
        )
    def Pi_plusminus(r_plusminus):
        return lambda lambda_: 2 / (r_31 * r_42) ** 0.5 * r_43 / ((r_plusminus - r_3) * (r_plusminus - r_4)) * (
            _ellippiinc(_ellipj(X(lambda_), k)[3], (r_plusminus - r_3) * r_41 / ((r_plusminus - r_4) * r_31), k)
            - nu_r * _ellippiinc(asin_x_i, (r_plusminus - r_3) * r_41 / ((r_plusminus - r_4) * r_31), k)
        )

    radial_potential = _radial_potential(a, eta, ell)

    # Kernel line 250
    lambda_x = (
        (
            -(2 / (r_31 * r_42) ** 0.5 * ellipkinc(np.asin(np.clip((r_plus - r_4) * r_31 / ((r_plus - r_3) * r_41), 0, 1) ** 0.5), k) - cal_I_i_0)
        ) if r_4 < r_minus else (
            (
                2 * cal_I_i_0
            ) if distant else (
                2 / (r_31 * r_42) ** 0.5 * ellipkinc(np.asin(np.clip(r_31 / r_41, 0, 1) ** 0.5), k) + cal_I_i_0
            )
        )
    ) if nu_r < 0 else (
        2 / (r_31 * r_42) ** 0.5 * ellipkinc(np.asin(np.clip(r_31 / r_41, 0, 1) ** 0.5), k) - cal_I_i_0
    )

    # equation 1.37
    def r(lambda_):
        sn_sq = _ellipj(X(lambda_), k)[0] ** 2
        return np.where(np.logical_or(lambda_ > lambda_x, lambda_ < 0),
            np.nan,
            (r_4 * r_31 - r_3 * r_41 * sn_sq) / (r_31 - r_41 * sn_sq)
        )
    def dr(lambda_):
        sn, cn, dn, _ = _ellipj(X(lambda_), k)
        return (
            r_31 * r_41 * r_43 * (r_31 * r_42) ** 0.5 * sn * cn * dn
        ) / (r_31 - r_41 * sn ** 2) ** 2
    def I_1(lambda_):
        return r_3 * lambda_ + r_43 * Pi_1(lambda_)
    def I_2(lambda_):
        if not distant:
            return dr(lambda_) / (r(lambda_) - r_3) - nu_r * radial_potential(initial_r) ** 0.5 / (initial_r - r_3) - (r_1 * r_4 + r_2 * r_3) / 2 * lambda_ - E_1(lambda_)
        return dr(lambda_) / (r(lambda_) - r_3) + r_3 - (r_1 * r_4 + r_2 * r_3) / 2 * lambda_ - E_1(lambda_)
    def I_plusminus(r_plusminus):
        return lambda lambda_: - lambda_ / (r_plusminus - r_3) - Pi_plusminus(r_plusminus)(lambda_)
    
    return r, I_1, I_2, I_plusminus(r_plus), I_plusminus(r_minus), lambda_x

def _case_3_radial(a, eta, ell, initial_r, nu_r, distant=False):
    r"""Computes the radial motion for case 3. Two real roots exist, both inside horizon.
    :math:`r_3 = \bar{r_4}`.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    initial_r : double
        initial r :math:`r_0`
    nu_r : int
        sign of the initial r momentum :math:`\nu_r = \text{sign}(p^r_0)`.
    distant: bool, optional
        whether the initial position is at infinity, defaults to false

    Returns
    -------
        :math:`(r(\lambda), I_1, I_2, I_+, I_-, \lambda_x)`: tuple(function, function, function, function, function, double)"""
    roots = _radial_roots(a, eta, ell)
    r_1, r_2, r_3, _ = roots
    r_plus, r_minus = _horizons(a)

    r_21 = r_2 - r_1
    r_31 = r_3 - r_1
    r_32 = r_3 - r_2
    
    # equation 1.38
    A = np.abs(r_32)
    B = np.abs(r_31)
    alpha_0 = (B + A) / (B - A)
    def alpha_plusminus(r_plusminus):
        return (B * (r_plusminus - r_2) + A * (r_plusminus - r_1)) / (B * (r_plusminus - r_2) - A * (r_plusminus - r_1))
    k_3 = ((A + B) ** 2 - r_21 ** 2) / (4 * A * B)
    def p(alpha, j):
        alpha_sq = alpha ** 2
        return ((alpha_sq - 1) / (j + (1 - j) * alpha_sq)) ** 0.5
    def f(alpha, phi, j):
        return p(alpha, j) / 2 * np.log(np.abs((p(alpha, j) * (1 - j * np.sin(phi) ** 2) ** 0.5 + np.sin(phi)) / (p(alpha, j) * (1 - j * np.sin(phi) ** 2) ** 0.5 - np.sin(phi))))
    def R_1(alpha, phi, j):
        alpha_sq = alpha ** 2
        return 1 / (1 - alpha_sq) * (_ellippiinc(phi, alpha_sq / (alpha_sq - 1), j) - alpha * f(alpha, phi, j))
    def Red_R_1(alpha, phi, j):
        alpha_sq = alpha ** 2
        return 1 / (1 - alpha_sq) * (
            ellipkinc(phi, j)
            - _ellippiinc(phi, j * (alpha_sq - 1) / alpha_sq, j)
            - alpha * (A * B) ** 0.5 / r_21 * np.log(4 * r_21 / (B**2 - A**2))
            + alpha * (A * B) ** 0.5 / r_21 * np.log((B**2 - A**2) / (4 * r_21) + A * B * r_21 / (B**2 - A**2))
        )
    def R_2(alpha, phi, j):
        alpha_sq = alpha ** 2
        return 1 / (alpha_sq - 1) * (
            ellipkinc(phi, j)
            - alpha_sq / (j + (1 - j) * alpha_sq) * (
                ellipeinc(phi, j)
                - alpha * np.sin(phi) * (1 - j * np.sin(phi) ** 2) ** 0.5 / (1 + alpha * np.cos(phi))
            )
        ) + 1 / (j + (1 - j) * alpha_sq) * (2*j - alpha_sq/(alpha_sq - 1)) * R_1(alpha, phi, j)
    def Red_R_2(alpha, phi, j):
        alpha_sq = alpha ** 2
        return 1 / (alpha_sq - 1) * (
            ellipkinc(phi, j)
            - alpha_sq / (j + (1 - j) * alpha_sq) * ellipeinc(phi, j)
        )
    x_i = np.clip(
        (
            (1 - B * (initial_r - r_2) / (A * (initial_r - r_1))) / (1 + B * (initial_r - r_2) / (A * (initial_r - r_1)))
        ) if not distant else (A - B) / (A + B),
        -1, 1
    )
    if environ.get("KG_DEBUG", "0") == "1":
        print(f"""{x_i=}, {(
                (1 - B * (initial_r - r_2) / (A * (initial_r - r_1))) / (1 + B * (initial_r - r_2) / (A * (initial_r - r_1)))
            ) if not distant else (A - B) / (A + B)}""")
    acos_x_i = np.arccos(x_i)
    cal_I_i_0 = 1 / (A * B) ** 0.5 * ellipkinc(acos_x_i, k_3)
    def X(lambda_):
        return (A * B) ** 0.5 * (lambda_ + nu_r * cal_I_i_0)
    def Pi(m):
        if not distant:
            return lambda lambda_: (2 * r_21 * (A * B) ** 0.5 / (B ** 2 - A ** 2)) ** m * (
                [R_1, R_2][m - 1](alpha_0, _ellipj(X(lambda_), k_3)[3], k_3)
                - nu_r * [R_1, R_2][m - 1](alpha_0, acos_x_i, k_3)
            )
        return lambda lambda_: (2 * r_21 * (A * B) ** 0.5 / (B ** 2 - A ** 2)) ** m * (
            [R_1, R_2][m - 1](alpha_0, _ellipj(X(lambda_), k_3)[3], k_3)
            - nu_r * [Red_R_1, Red_R_2][m - 1](alpha_0, acos_x_i, k_3)
        )
    Pi_1 = Pi(1)
    Pi_2 = Pi(2)
    def Pi_plusminus(r_plusminus):
        alpha_plusminus_ = alpha_plusminus(r_plusminus)
        return lambda lambda_: (2 * r_21 * (A * B) ** 0.5 / (B * (r_plusminus - r_2) - A * (r_plusminus - r_1))) * (
            R_1(alpha_plusminus_, _ellipj(X(lambda_), k_3)[3], k_3)
            - nu_r * R_1(alpha_plusminus_, acos_x_i, k_3)
        )

    # Kernel line 330
    lambda_x = nu_r * (1 / (A * B) ** 0.5 * ellipkinc(np.acos(np.clip((A * (r_plus - r_1) - B * (r_plus - r_2)) / (A * (r_plus - r_1) + B * (r_plus - r_2)), -1, 1)), k_3) - cal_I_i_0)

    # equation 1.39
    def r(lambda_):
        cn = _ellipj(X(lambda_), k_3)[1]
        return np.where(np.logical_or(lambda_ > lambda_x, lambda_ < 0),
            np.nan,
            (
                (B * r_2 - A * r_1) + (B * r_2 + A * r_1) * cn 
            ) / (
                (B - A) + (B + A) * cn
            )
        )
    def I_1(lambda_):
        return (B * r_2 + A * r_1) / (B + A) * lambda_ + Pi_1(lambda_)
    def I_2(lambda_):
        if not distant:
            return ((B * r_2 + A * r_1) / (B + A)) ** 2 * lambda_ + 2 * (B * r_2 + A * r_1) / (B + A) * Pi_1(lambda_) + (A * B) ** 0.5 * Pi_2(lambda_)
        return ((B * r_2 + A * r_1) / (B + A)) ** 2 * lambda_ + 2 * (B * r_2 + A * r_1) / (B + A) * (2 * r_21 * (A * B) ** 0.5) / (B ** 2 - A ** 2) * R_1(alpha_0, _ellipj(X(lambda_), k_3)[3], k_3) + (A * B) ** 0.5 * Pi_2(lambda_) + (A ** 2 - B ** 2) / (2 * r_21) - (r_1 + r_2) + (A * r_1 + B * r_2) / (A + B)
    def I_plusminus(r_plusminus):
        return lambda lambda_: - (
            (A + B) * lambda_
            + Pi_plusminus(r_plusminus)(lambda_)
        ) / (
            B * (r_plusminus - r_2)
            + A * (r_plusminus - r_1)
        )
    
    return r, I_1, I_2, I_plusminus(r_plus), I_plusminus(r_minus), lambda_x

def _case_4_radial(a, eta, ell, initial_r, nu_r, distant=False):
    r"""Computes the radial motion for case 4. No real roots exist.
    :math:`r_1 = \bar{r_2}` and :math:`r_3 = \bar{r_4}`.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    initial_r : double
        initial r :math:`r_0`
    nu_r : int
        sign of the initial r momentum :math:`\nu_r = \text{sign}(p^r_0)`.
    distant: bool, optional
        whether the initial position is at infinity, defaults to false

    Returns
    -------
        :math:`(r(\lambda), I_1, I_2, I_+, I_-, \lambda_x)`: tuple(function, function, function, function, function, double)"""
    roots = _radial_roots(a, eta, ell)
    r_1, r_2, _, r_4 = roots
    r_plus, r_minus = _horizons(a)

    r_41 = r_4 - r_1
    r_42 = r_4 - r_2

    # equation 1.32
    b_1 = np.real(r_4)
    a_2 = np.imag(r_2)

    # equation 1.40
    def x_4(r):
        return (r + b_1) / a_2
    C = np.abs(r_42)
    D = np.abs(r_41)
    k_4 = 4 * C * D / (C + D) ** 2
    g_0 = (
        (4 * a_2 ** 2 - (C - D) ** 2) / ((C + D) ** 2 - 4 * a_2 ** 2)
    ) ** 0.5
    def p(alpha, j):
        alpha_sq = alpha ** 2
        return ((1 + alpha_sq) / (1 - j + alpha_sq)) ** 0.5
    def f(alpha, phi, j):
        return p(alpha, j) / 2 * np.log(np.abs((1 - p(alpha, j)) / (1 + p(alpha, j)) * (1 + p(alpha, j) * (1 - j * np.sin(phi) ** 2) ** 0.5) / (1 - p(alpha, j) * (1 - j * np.sin(phi) ** 2) ** 0.5)))
    def S_1(alpha, phi, j):
        alpha_sq = alpha ** 2
        return 1 / (1 + alpha_sq) * (ellipkinc(phi, j) + alpha_sq * _ellippiinc(phi, 1 + alpha_sq, j) - alpha * f(alpha, phi, j))
    def Red_S_1(alpha, phi, j):
        alpha_sq = alpha ** 2
        return ellipkinc(phi, j) - 1 / (1 + alpha_sq) * (
            alpha_sq * _ellippiinc(phi, j / (1 + alpha_sq), j)
            + alpha * (C + D) / (4 * a_2) * np.log(64 * a_2 ** 2 / ((2 * a_2 + C + D) ** 2 * (alpha_sq * (C + D) ** 2 + 4 * a_2 ** 2)))
        )
    def S_2(alpha, phi, j):
        alpha_sq = alpha ** 2
        return -1 / ((1 + alpha_sq) * (1 - j + alpha_sq)) * (
            (1 - j) * ellipkinc(phi, j)
            + alpha_sq * ellipeinc(phi, j)
            + alpha_sq * (1 - j * np.sin(phi) ** 2) ** 0.5 * (alpha - np.tan(phi)) / (1 + alpha * np.tan(phi))
            - alpha ** 3
        ) + (1 / (1 + alpha_sq) + (1 - j) / (1 - j + alpha_sq)) * S_1(alpha, phi, j)
    def Red_S_2(alpha, phi, j):
        alpha_sq = alpha ** 2
        return -1 / ((1 + alpha_sq) * (1 - j + alpha_sq)) * (
            (1 - j) * ellipkinc(phi, j)
            + alpha_sq * ellipeinc(phi, j)
            - alpha ** 3
        )
    cal_I_0 = (
        2 / (C + D) * ellipkinc(np.atan(x_4(initial_r)) + np.atan(g_0), k_4)
    ) if not distant else (
        2 / (C + D) * ellipkinc(pi/2 + np.atan(g_0), k_4)
    )
    def X(lambda_):
        return (C + D) / 2 * (nu_r * lambda_ + cal_I_0)
    def Pi(m):
        if not distant:
            return lambda lambda_: 2 * nu_r / (C + D) * (a_2 / g_0 * (1 + g_0 ** 2)) ** m * ([S_1, S_2][m - 1](g_0, _ellipj(X(lambda_), k_4)[3], k_4) - [S_1, S_2][m - 1](g_0, np.atan(x_4(initial_r)) + np.atan(g_0), k_4))
        return lambda lambda_: 2 * nu_r / (C + D) * (a_2 / g_0 * (1 + g_0 ** 2)) ** m * ([S_1, S_2][m - 1](g_0, _ellipj(X(lambda_), k_4)[3], k_4) - [Red_S_1, Red_S_2][m - 1](g_0, pi / 2 + np.atan(g_0), k_4))
    Pi_1 = Pi(1)
    Pi_2 = Pi(2)
    def g_plusminus(r_plusminus): # Gralla & Lupsasca B96
        return (g_0 * x_4(r_plusminus) - 1) / (g_0 + x_4(r_plusminus))
    def Pi_plusminus(r_plusminus):
        g_plusminus_ = g_plusminus(r_plusminus)
        return lambda lambda_: 2 * nu_r / (C + D) * ((1 + g_0 ** 2) / (g_0 * (g_0 + x_4(r_plusminus)))) * (S_1(g_plusminus_, _ellipj(X(lambda_), k_4)[3], k_4) - S_1(g_plusminus_, np.atan(x_4(initial_r)) + np.atan(g_0), k_4))

    # Kernel line 400
    lambda_x = nu_r * (2 / (C + D) * ellipkinc(np.atan(x_4(r_plus)) + np.atan(g_0), k_4) - cal_I_0)

    # equation 1.41
    def r(lambda_):
        sc = _sc(X(lambda_), k_4)
        return np.where(np.logical_or(lambda_ > lambda_x, lambda_ < 0),
            np.nan,
            -a_2 * (g_0 - sc) / (1 + g_0 * sc) - b_1
        )
    def I_1(lambda_):
        return (a_2 / g_0 - b_1) * lambda_ - Pi_1(lambda_)
    def I_2(lambda_):
        if not distant:
            return (a_2 / g_0 - b_1) ** 2 * lambda_ - 2 * (a_2 / g_0 - b_1) * Pi_1(lambda_) + Pi_2(lambda_)
        return (a_2 / g_0 - b_1) ** 2 * lambda_ + 4 * (a_2 / g_0 - b_1) * (1 + g_0 ** 2) / (C + D) * a_2 / g_0 * S_1(g_0, _ellipj(X(lambda_), k_4)[3], k_4) + Pi_2(lambda_) + b_1 - 2 * g_0 * C * D / ((g_0 ** 2 + 1) ** 0.5 * ((C - D) ** 2 + g_0 ** 2 * (C + D) ** 2) ** 0.5)
    def I_plusminus(r_plusminus):
        return lambda lambda_: g_0 / (a_2 * (1 - g_0 * x_4(r_plusminus))) * (lambda_ - Pi_plusminus(r_plusminus)(lambda_))
    
    return r, I_1, I_2, I_plusminus(r_plus), I_plusminus(r_minus), lambda_x

def photon_escapes(a, eta, ell, initial_r=np.inf, distant=False):
    r"""Determines whether a photon with given parameters escapes to infinity.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    initial_r : double
        initial r :math:`r_0`, defaults to infinity
    distant : bool, optional
        whether the initial position is at infinity, defaults to false

    Returns
    -------
    bool
        True if the photon escapes to infinity, otherwise False"""

    _, _, r_3, r_4 = _radial_roots(a, eta, ell)
    r_plus, r_minus = _horizons(a)

    if not np.isreal(r_4): return False

    if distant:
        return r_3 > r_minus

    return (r_4 < r_plus or initial_r > r_4) and r_4 > r_plus


def _r(a, eta, ell, initial_r, nu_r, distant=False):
    r"""Computes the radial motion.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    initial_r : double
        initial r :math:`r_0`
    nu_r : int
        sign of the initial r momentum :math:`\nu_r = \text{sign}(p^r_0)`.
    distant: bool, optional
        whether the initial position is at infinity, defaults to false

    Returns
    -------
    tuple(function, function, function, double)
        :math:`(r(\lambda), I_\phi(\lambda), I_t(\lambda), \lambda_x)`"""
    
    roots = _radial_roots(a, eta, ell)
    _, r_2, _, r_4 = roots
    r_plus, r_minus = _horizons(a)

    if environ.get("KG_DEBUG", "0") == "1":
        radial_roots = roots
        print(f"{radial_roots=}, {r_plus=}, {r_minus=}")

    if not np.isreal(r_2):
        if environ.get("KG_DEBUG", "0") == "1":
            print("case 4")
        r, I_1, I_2, I_plus, I_minus, lambda_x = _case_4_radial(a, eta, ell, initial_r, nu_r, distant)
    elif not np.isreal(r_4):
        if environ.get("KG_DEBUG", "0") == "1":
            print("case 3")
        r, I_1, I_2, I_plus, I_minus, lambda_x = _case_3_radial(a, eta, ell, initial_r, nu_r, distant)
    elif r_4 < r_plus or initial_r > r_4:
        if environ.get("KG_DEBUG", "0") == "1":
            print("case 2")
        r, I_1, I_2, I_plus, I_minus, lambda_x = _case_2_radial(a, eta, ell, initial_r, nu_r, distant)
    else:
        if environ.get("KG_DEBUG", "0") == "1":
            print("case 1")
        r, I_1, I_2, I_plus, I_minus, lambda_x = _case_1_radial(a, eta, ell, initial_r, nu_r)
    
    # equation 1.33
    def I_phi(lambda_):
        return 2 * a / (r_plus - r_minus) * ((r_plus - a * ell / 2) * I_plus(lambda_) - (r_minus - a * ell / 2) * I_minus(lambda_))
    def I_t(lambda_):
        out = 4 / (r_plus - r_minus) * (r_plus * (r_plus - a * ell / 2) * I_plus(lambda_) - r_minus * (r_minus - a * ell / 2) * I_minus(lambda_)) + 4 * lambda_ + 2 * I_1(lambda_) + I_2(lambda_)
        if distant: out += 2 * np.log(2)
        return out
    
    return lambda lambda_: np.where(np.isclose(lambda_, 0),
        initial_r,
        r(lambda_)
    ), I_phi, I_t, lambda_x

def trajectory(a, eta, ell, initial_pos, nu_theta, nu_r, distant=False):
    r"""Computes the trajectory of a photon in Kerr spacetime.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    initial_pos : tuple(double, double, double, double)
        initial position :math:`(t_0, r_0, \theta_0, \phi_0)`
    nu_theta : int
        sign of the initial polar momentum :math:`\nu_\theta = \text{sign}(p^\theta_0)`
    nu_r : int
        sign of the initial r momentum :math:`\nu_r = \text{sign}(p^r_0)`.
    distant: bool, optional
        whether the initial position is at infinity, defaults to false
    
    Returns
    -------
    tuple(function, function, function, function)
        :math:`(t(\lambda), r(\lambda), \theta(\lambda), \phi(\lambda))`"""
    if environ.get("KG_DEBUG", "0") == "1":
        print(f"{a=}, {eta=}, {ell=}, {distant=}")

    r, I_phi, I_t, lambda_x = _r(a, eta, ell, initial_pos[1], nu_r, distant)
    theta, G_phi, G_t = _theta(a, eta, ell, initial_pos[2], nu_theta, lambda_x)

    if environ.get("KG_DEBUG", "0") == "1":
        print(f"{lambda_x=}")

    def phi(lambda_):
        return np.where(np.logical_or(lambda_ > lambda_x, lambda_ < 0),
            np.nan,
            initial_pos[3] + G_phi(lambda_) + I_phi(lambda_)
        )

    def t(lambda_):
        return np.where(np.logical_or(lambda_ > lambda_x, lambda_ < 0),
            np.nan,
            np.where(np.isclose(lambda_, 0),
                initial_pos[0],
                initial_pos[0] + a ** 2 * G_t(lambda_) + I_t(lambda_) if not distant else
                a ** 2 * G_t(lambda_) + I_t(lambda_) + r(lambda_) + 2 * np.log(r(lambda_) / 2)
            )
        )
    
    return t, r, theta, phi, lambda_x

def _sign_p_theta(a, eta, ell, theta, p_theta):
    r"""Determines the sign of the initial polar momentum :math:`\nu_\theta = \text{sign}(p^\theta_0)`
    
    Parameters
    ----------
    a : double
        dimensionless spin parameter
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    theta : double
        initial polar angle :math:`\theta_0`
    p_theta : double
        initial polar momentum :math:`p^\theta_0`

    Returns
    -------
    int
        sign of the initial polar momentum :math:`\nu_\theta = \text{sign}(p^\theta_0)`
    """
    nu_theta = np.sign(p_theta)
    if nu_theta == 0:
        if eta > 0:
            nu_theta = np.sign(pi/2 - theta)
        else:
            u_plus, u_minus = _polar_roots(a, eta, ell)
            theta_plus = np.arccos(u_plus ** 0.5)
            theta_minus = np.arccos(u_minus ** 0.5)
            if theta > pi/2:
                theta_plus = np.arccos(-u_plus ** 0.5)
                theta_minus = np.arccos(-u_minus ** 0.5)
            
            nu_theta = np.sign(theta_minus - theta_plus)
            if np.isclose(theta, theta_minus):
                nu_theta *= -1
    
    return nu_theta


class LightOrbit:
    r"""Class representing a lightlike orbit in Kerr spacetime defined using initial conditions.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    initial_position : tuple(double,double,double,double)
        initial position of the orbit :math:`(t_0,r_0,\theta_0,\phi_0)`
    initial_momentum : tuple(double,double,double,double)
        initial four-momentum of the orbit
        :math:`(p^t_0,p^r_0,p^\theta_0,p^\phi_0)`
    M : double, optional
        mass of the primary in solar masses, if not specified, units are in terms of M

    Attributes
    ----------
    a : double
        dimensionless spin parameter
    initial_position : tuple(double, double, double, double)
        initial position of the orbit :math:`(t_0,r_0,\theta_0,\phi_0)`
    initial_momentum : tuple(double, double, double, double)
        initial four-momentum of the orbit
        :math:`(p^t_0,p^r_0,p^\theta_0,p^\phi_0)`
    M : double, optional
        mass of the primary in solar masses
    E : double
        energy :math:`E = -p_t`
    L : double
        angular momentum :math:`L = p_\phi`
    Q : double
        Carter constant :math:`Q = p_\theta^2 + \cos^2\theta (a^2 p_t^2 - p_\phi^2/\sin^2\theta)`
    eta : double
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell : double
        angular momentum per energy :math:`\ell = L/E`
    escapes : boolean
        true if the photon escapes to infinity
    lambda_x : double
        Mino time of capture/escape to infinity
    escape_coordinates : tuple(double, double)
        :math:`(\theta, \phi)` coordinates at which photon escapes to infinity or (np.nan, np.nan) if captured
    """

    def __init__(self, a, initial_position, initial_momentum, M=None):
        self.a = a
        self.initial_position = initial_position
        self.initial_momentum = initial_momentum
        self.M = M if M is None else mass_in_kg(M)

        # check if initial four-velocity is valid
        spacetime = KerrSpacetime(a)
        initial_norm = spacetime.norm(*initial_position, initial_momentum)
        if not np.isclose(initial_norm, 0):
            raise ValueError("Initial velocity is not lightlike")

        p_t, _, p_theta, p_phi = spacetime.metric(*initial_position) @ initial_momentum
        self.E = -p_t
        self.L = p_phi
        self.Q = p_theta**2 - np.cos(initial_position[2]) ** 2 * (
            a**2 * p_t**2 - p_phi**2 / np.sin(initial_position[2]) ** 2
        )

        self.eta = self.Q / self.E**2
        self.ell = self.L / self.E

        if self.eta < (0 if abs(self.ell) >= a else -(abs(self.ell) - a) ** 2):
            raise ValueError("Forbidden motion")

        self.escapes = photon_escapes(self.a, self.eta, self.ell, initial_r=self.initial_position[1])
    
    def trajectory(
        self, distance_units="natural", time_units="natural"
    ):
        r"""Computes the components of the trajectory as a function of Mino time.

        Parameters
        ----------
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
        tuple(function, function, function, function)
            tuple of functions :math:`(t(\lambda), r(\lambda),
            \theta(\lambda), \phi(\lambda))`
        """
        if ((distance_units != "natural") or (time_units != "natural")) and self.M is None:
            raise ValueError("M must be specified to convert to physical units")

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

        nu_r = np.sign(self.initial_momentum[1])
        nu_theta = _sign_p_theta(self.a, self.eta, self.ell, self.initial_position[2], self.initial_momentum[2])
        
        if environ.get("KG_DEBUG", "0") == "1":
            x = ", ".join(str(i) for i in self.initial_position)
            p_vec = ", ".join(str(i) for i in self.initial_momentum)
            p_cov = ", ".join(str(i) for i in KerrSpacetime(self.a).metric(*self.initial_position) @ self.initial_momentum)
            print(f"Initial conditions: {x=}, {p_vec=}, {p_cov=}, {nu_theta=}, {nu_r=}")

        *trajectory_, lambda_x = trajectory(
            self.a,
            self.eta,
            self.ell,
            self.initial_position,
            nu_theta,
            nu_r,
        )
        self.lambda_x = lambda_x
        self.escape_coordinates = (trajectory_[2](lambda_x), trajectory_[3](lambda_x)) if self.escapes else (np.nan, np.nan)

        return (
            lambda lambda_: time_conversion_func[time_units](trajectory_[0](lambda_), self.M),
            lambda lambda_: distance_conversion_func[distance_units](trajectory_[1](lambda_), self.M),
            trajectory_[2],
            trajectory_[3]
        )

    def plot(
        self,
        lambda0=0,
        lambda1=None,
        elevation=30,
        azimuth=-60,
        grid=True,
        axes=True,
        lw=1,
        color="red",
        tau=np.inf,
        point_density=200,
        axes_limits=None,
    ):
        r"""Creates a plot of the orbit

        Parameters
        ----------
        lambda0 : double, optional
            starting mino time
        lambda1 : double, optional
            ending mino time
        elevation : double, optional
            camera elevation angle in degrees, defaults to 30
        azimuth : double, optional
            camera azimuthal angle in degrees, defaults to -60
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        grid : bool, optional
            if true, grid lines are shown on plot
        axes : bool, optional
            if true, axes are shown on plot
        lw : double, optional
            linewidth of the orbital trajectory, defaults to 1
        color : str, optional
            color of the orbital trajectory, defaults to "red"
        tau : double, optional
            time constant for the exponential decay of the linewidth,
            defaults to infinity
        point_density : int, optional
            number of points to plot per unit of mino time, defaults to
            200
        axes_limits : tuple, optional
            limits for the axes (x_min, x_max, y_min, y_max, z_min, z_max), if None, limits are set automatically based on the trajectory and event horizon

        Returns
        -------
        matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
            matplotlib figure and axes
        """
        if lambda1 is None:
            lambda1 = self.lambda_x
        trajectory = self.trajectory()

        return plot(
            self.a,
            trajectory,
            lambda0,
            lambda1,
            elevation,
            azimuth,
            grid,
            axes,
            lw,
            color,
            tau,
            point_density,
            axes_limits,
        )

    def is_visible(self, points, elevation, azimuth):
        """Determines if a point is visible from a given viewing angle or obscured
        by the black hole. Viewing angles are defined as in
        https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html and
        black hole is centered at the origin.

        Parameters
        ----------
        points : array_like
            list of points given in cartesian coordinates
        elevation : double
            camera elevation angle in degrees
        azimuth : double
            camera azimuthal angle in degrees

        Returns
        -------
        np.array
            boolean array indicating whether each point is visible
        """
        return is_visible(self.a, points, elevation, azimuth)

    def animate(
        self,
        filename,
        lambda0=0,
        lambda1=None,
        elevation=30,
        azimuth=-60,
        grid=True,
        axes=True,
        color="red",
        tau=2,
        tail_length=5,
        lw=2,
        azimuthal_pan=None,
        elevation_pan=None,
        roll=None,
        speed=1,
        background_color=None,
        axis_limit=None,
        plot_components=False,
        axes_limits=None,
    ):
        r"""Saves an animation of the orbit as an mp4 file.
        Note that this function requires ffmpeg to be installed and may take several
        minutes to run depending on the length of the animation.

        Parameters
        ----------
        filename : str
            filename to save the animation to
        lambda0 : double, optional
            starting mino time, defaults to 0
        lambda1 : double, optional
            ending mino time, defaults to 10
        elevation : double, optional
            camera elevation angle in degrees, defaults to 30
        azimuth : double, optional
            camera azimuthal angle in degrees, defaults to -60
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        grid : bool, optional
            sets visibility of the grid, defaults to True
        axes : bool, optional
            sets visibility of axes, defaults to True
        color : str, optional
            color of the orbital tail, defaults to "red"
        tau : double, optional
            time constant for the exponential decay in the opacity of
            the tail, defaults to 2
        tail_length : double, optional
            length of the tail in units of mino time, defaults to 5
        lw : double, optional
            linewidth of the orbital trajectory, defaults to 2
        azimuthal_pan : function, optional
            function defining the azimuthal angle of the camera in
            degrees as a function of mino time, defaults to None
        elevation_pan : function, optional
            function defining the elevation angle of the camera in
            degrees as a function of mino time, defaults to None
        roll : function, optional
            function defining the roll angle of the camera in degrees as
            a function of mino time, defaults to None
        axis_limit : function, optional
            sets the axis limit as a function of mino time, defaults to
            None
        speed : double, optional
            playback speed of the animation in units of mino time per
            second (must be a multiple of 1/8), defaults to 1
        background_color : str, optional
            color of the background, defaults to None
        plot_components : bool, optional
            if true, plots the components of the trajectory in addition
            to the trajectory itself, defaults to False
        axes_limits : tuple, optional
            limits for the axes (x_min, x_max, y_min, y_max, z_min, z_max), if None, limits are set automatically based on the trajectory and event horizon
        """
        if lambda1 is None:
            lambda1 = self.lambda_x
        trajectory = self.trajectory()

        animate(
            self.a,
            trajectory,
            filename,
            lambda0,
            lambda1,
            elevation,
            azimuth,
            grid,
            axes,
            color,
            tau,
            tail_length,
            lw,
            azimuthal_pan,
            elevation_pan,
            roll,
            speed,
            background_color,
            axis_limit,
            plot_components,
            lambda t: fr"$a = {self.a}\quad \eta = {self.eta:.3f}\quad \ell = {self.ell:.3f}\quad \lambda = {t:.2f}$",
            axes_limits,
        )

# kernel line 469
def _shell_intersection_lambda(a, eta, ell, shell_radius):
    roots = _radial_roots(a, eta, ell)
    k = _k(roots)
    r_1, r_2, r_3, r_4 = roots

    x0 = np.clip((r_3 - r_1) / (r_4 - r_1), 0, 1) ** 0.5
    x_shell = np.clip((shell_radius - r_4) * (r_3 - r_1) / ((shell_radius - r_3) * (r_4 - r_1)), 0, 1) ** 0.5

    I0 = 2 / ((r_3 - r_1) * (r_4 - r_2)) ** 0.5 * ellipkinc(np.arcsin(x0), k)
    I0_shell = 2 / ((r_3 - r_1) * (r_4 - r_2)) ** 0.5 * ellipkinc(np.arcsin(x_shell), k)

    return I0 + I0_shell

class DistantLightOrbit(LightOrbit):
    r"""Class representing a distant lightlike orbit in Kerr spacetime defined using initial conditions.

    Parameters
    ----------
    a : double
        dimensionless spin parameter
    initial_theta : double
        initial polar angle :math:`\theta_0`
    initial_phi : double
        initial azimuthal angle :math:`\phi_0`
    alpha : double
        bardeen coordinate :math:`\alpha`
    beta : double
        bardeen coordinate :math:`\beta`
    shell_radius : double, optional
        radius of the shell used for generating image of distorted background, defaults to 50 in c = G = M = 1 units
    M : double, optional
        mass of the primary in solar masses, if not specified, units are in terms of M

    Attributes
    ----------
    a
        dimensionless spin parameter
    initial_theta : double
        initial polar angle :math:`\theta_0`
    alpha : double
        bardeen coordinate :math:`\alpha`
    beta : double
        bardeen coordinate :math:`\beta`
    shell_radius : double
        radius of the shell used for generating image of distorted background
    M
        mass of the primary in solar masses
    E
        energy :math:`E = -p_t`
    L
        angular momentum :math:`L = p_\phi`
    Q
        Carter constant :math:`Q = p_\theta^2 + \cos^2\theta (a^2 p_t^2 - p_\phi^2/\sin^2\theta)`
    eta
        Carter constant per square energy :math:`\eta = Q/E^2`
    ell
        angular momentum per energy :math:`\ell = L/E`
    escapes : boolean
        true if the photon escapes to infinity
    lambda_x : double
        Mino time of capture/escape to infinity
    escape_coordinates : tuple(double, double)
        :math:`(\theta, \phi)` coordinates at which photon escapes to infinity or (np.nan, np.nan) if captured
    lambda_shell : double
        Mino time of intersection with shell of radius shell_radius
    shell_intersection_coordinates : tuple(double, double, double)
        :math:`(\Delta v, \theta, \phi)` coordinates at which photon intersects shell of radius shell_radius or (np.nan, np.nan, np.nan) if photon does not intersect the shell
    """

    def __init__(self, a, initial_theta, initial_phi, alpha, beta, shell_radius=50, M=None):
        self.a = a
        self.initial_theta = initial_theta
        self.initial_phi = initial_phi
        self.alpha = alpha
        self.beta = beta
        self.M = M if M is None else mass_in_kg(M)
        self.shell_radius = shell_radius

        self.ell = alpha * np.sin(initial_theta)
        self.eta = beta ** 2 + (alpha ** 2 - a ** 2) * np.cos(initial_theta) ** 2

        self.E = 1
        self.L = self.ell * self.E
        self.Q = self.eta * self.E ** 2

        if self.eta < (0 if abs(self.ell) >= a else -(abs(self.ell) - a) ** 2):
            raise ValueError("Forbidden motion")

        self.escapes = photon_escapes(self.a, self.eta, self.ell, distant=True)
        if self.escapes:
            self.lambda_shell = _shell_intersection_lambda(self.a, self.eta, self.ell, self.shell_radius)
        else:
            self.lambda_shell = np.nan

    def trajectory(
        self, distance_units="natural", time_units="natural"
    ):
        r"""Computes the components of the trajectory as a function of Mino time.

        Parameters
        ----------
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
        tuple(function, function, function, function)
            tuple of functions :math:`(t(\lambda), r(\lambda),
            \theta(\lambda), \phi(\lambda))`
        """
        if ((distance_units != "natural") or (time_units != "natural")) and self.M is None:
            raise ValueError("M must be specified to convert to physical units")

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

        nu_r = -1
        nu_theta = -np.sign(self.beta)
        nu_theta = _sign_p_theta(self.a, self.eta, self.ell, self.initial_theta, nu_theta)

        *trajectory_, lambda_x = trajectory(
            self.a,
            self.eta,
            self.ell,
            np.array([0, np.inf, self.initial_theta, self.initial_phi]),
            nu_theta,
            nu_r,
            True
        )
        self.lambda_x = lambda_x
        self.escape_coordinates = (trajectory_[2](lambda_x), trajectory_[3](lambda_x)) if self.escapes else (np.nan, np.nan)

        trajectory_units = (
            lambda lambda_: time_conversion_func[time_units](trajectory_[0](lambda_), self.M),
            lambda lambda_: distance_conversion_func[distance_units](trajectory_[1](lambda_), self.M),
            trajectory_[2],
            trajectory_[3]
        )
        self.shell_intersection_coordinates = (trajectory_units[0](self.lambda_shell), trajectory_units[2](self.lambda_shell), trajectory_units[3](self.lambda_shell)) if self.escapes else (np.nan, np.nan, np.nan)

        return trajectory_units
