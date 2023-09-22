"""
Module containing functions for computing frequencies of motion from the spin parameter, constants of motion, and radial and polar roots.
Frequencies are computed using the method derived in `Fujita and Hikida <https://doi.org/10.48550/arXiv.0906.1420>`_
"""
from scipy.special import ellipk, ellipe, elliprj, elliprf
from .constants import *

def _ellippi(n,k):
    r"""
    Complete elliptic integral of the third kind defined as :math:`\Pi(n,k) = \int_0^{\frac{\pi}{2}} \frac{d\theta}{(1-n\sin^2{\theta})\sqrt{1-k^2\sin^2{\theta}}}`
    
    :type n: double
    :type k: double

    :rtype: double
    """
    # Note: sign of n is reversed from the definition in Fujita and Hikida

    # formula from https://en.wikipedia.org/wiki/Carlson_symmetric_form
    return elliprf(0,1-k**2,1)+1/3*n*elliprj(0,1-k**2,1,1-n)

def r_frequency_from_constants(constants,radial_roots):
    """
    Computes the frequency of motion in r in Mino time.

    :param constants: dimensionless constants of motion for the orbit in the form :math:`(E,L,Q)`
    :type constants: tuple(double, double, double)
    :param radial_roots: tuple containing the four roots of the radial equation :math:`(r_1, r_2, r_3, r_4)`.
    :type radial_roots: tuple(double, double, double, double)

    :rtype: double
    """
    E, L, Q = constants

    r1,r2,r3,r4 = radial_roots
    # equation 13
    k_r = sqrt((r1-r2)*(r3-r4)/((r1-r3)*(r2-r4)))
    # equation 15
    return pi*sqrt((1-E**2)*(r1-r3)*(r2-r4))/(2*ellipk(k_r**2))

def theta_frequency_from_constants(a,constants,radial_roots,polar_roots):
    """
    Computes the frequency of motion in theta in Mino time.
    
    :param a: dimensionless spin of the black hole
    :type a: double
    :param constants: dimensionless constants of motion for the orbit in the form :math:`(E,L,Q)`
    :type constants: tuple(double, double, double)
    :param radial_roots: tuple containing the four roots of the radial polynomial :math:`(r_1, r_2, r_3, r_4)`.
    :type radial_roots: tuple(double, double, double, double)
    :param polar_roots: tuple containing the roots of the polar equation :math:`(z_-, z_+)`
    :type polar_roots: tuple(double, double)
    
    :rtype: double
    """
    r1, r2, r3, r4 = radial_roots
    z_minus, z_plus = polar_roots
    E, L, Q = constants

    # Schwarzschild case
    if a == 0:
        p = 2*r1*r2/(r1+r2)
        e = (r1-r2)/(r1+r2)
        return p/sqrt(-3-e**2+p)*(sign(L) if e == 0 else 1)
    
    
    # equation 13
    k_theta = sqrt(z_minus/z_plus)
    # simplified form of epsilon0*z_plus
    e0zp = (a**2*(1-E**2)*(1-z_minus)+L**2)/(L**2*(1-z_minus))
    
    # equation 15
    return pi*L*sqrt(e0zp)/(2*ellipk(k_theta**2))

def phi_frequency_from_constants(a,constants,radial_roots,polar_roots,upsilon_r=None,upsilon_theta=None):
    """
    Computes the frequency of motion in phi in Mino time.
    
    :param a: dimensionless spin of the black hole
    :type a: double
    :param constants: dimensionless constants of motion for the orbit in the form :math:`(E,L,Q)`
    :type constants: tuple(double, double, double)
    :param radial_roots: tuple containing the four roots of the radial polynomial :math:`(r_1, r_2, r_3, r_4)`.
    :type radial_roots: tuple(double, double, double, double)
    :param polar_roots: tuple containing the roots of the polar equation :math:`(z_-, z_+)`
    :type polar_roots: tuple(double, double)
    :param upsilon_r: Mino frequency of motion in r can be passed in to speed computation if it is already known
    :type upsilon_r: double, optional
    :param upsilon_theta: Mino frequency of motion in theta can be passed in to speed computation if it is already known
    :type upsilon_theta: double, optional
    
    :rtype: double
    """

    E, L, Q = constants
    r1,r2,r3,r4 = radial_roots
    z_minus, z_plus = polar_roots

    # Schwarzschild case
    if a == 0:
        p = 2*r1*r2/(r1+r2)
        e = (r1-r2)/(r1+r2)
        return sign(L)*p/sqrt(-3-e**2+p)
    
    # compute frequencies if they are not passed in
    if upsilon_r is None: upsilon_r = r_frequency_from_constants(a,constants,radial_roots)
    if upsilon_theta is None: upsilon_theta = theta_frequency_from_constants(a,constants,radial_roots,polar_roots)
    
    # simplified form of epsilon0*z_plus
    e0zp = (a**2*(1-E**2)*(1-z_minus)+L**2)/(L**2*(1-z_minus))

    r_plus = 1+sqrt(1-a**2)
    r_minus = 1-sqrt(1-a**2)
    
    h_plus = (r1-r2)*(r3-r_plus)/((r1-r3)*(r2-r_plus))
    h_minus = (r1-r2)*(r3-r_minus)/((r1-r3)*(r2-r_minus))
    
    # equation 13
    k_theta = sqrt(z_minus/z_plus)
    k_r = sqrt((r1-r2)*(r3-r4)/((r1-r3)*(r2-r4)))
    
    # equation 21
    return  2*upsilon_theta/(pi*sqrt(e0zp))*_ellippi(z_minus,k_theta) \
            + 2*a*upsilon_r/(pi*(r_plus-r_minus)*sqrt((1-E**2)*(r1-r3)*(r2-r4))) \
            * ((2*E*r_plus-a*L)/(r3-r_plus)*(ellipk(k_r**2)-(r2-r3)/(r2-r_plus)*_ellippi(h_plus,k_r)) \
               - (2*E*r_minus-a*L)/(r3-r_minus)*(ellipk(k_r**2)-(r2-r3)/(r2-r_minus)*_ellippi(h_minus,k_r))
              )

def gamma_from_constants(a,constants,radial_roots,polar_roots,upsilon_r=None,upsilon_theta=None):
    """
    Computes the average rate at which observer time accumulates in Mino time.
    
    :param a: dimensionless spin of the black hole
    :type a: double
    :param constants: dimensionless constants of motion for the orbit in the form :math:`(E,L,Q)`
    :type constants: tuple(double, double, double)
    :param radial_roots: tuple containing the four roots of the radial polynomial :math:`(r_1, r_2, r_3, r_4)`.
    :type radial_roots: tuple(double, double, double, double)
    :param polar_roots: tuple containing the roots of the polar equation :math:`(z_-, z_+)`
    :type polar_roots: tuple(double, double)
    :param upsilon_r: Mino frequency of motion in r can be passed in to speed computation if it is already known
    :type upsilon_r: double, optional
    :param upsilon_theta: Mino frequency of motion in theta can be passed in to speed computation if it is already known
    :type upsilon_theta: double, optional
    
    :rtype: double
    """
    r1,r2,r3,r4 = radial_roots
    z_minus, z_plus = polar_roots

    e = (r1-r2)/(r1+r2)
    # marginally bound case
    if e == 1:
        return inf
    
    E, L, Q = constants
    
    epsilon0 =  a**2*(1-E**2)/L**2
    # simplified form of a**2*sqrt(z_plus/epsilon0)
    a2sqrt_zp_over_e0 = L**2/((1-E**2)*sqrt(1-z_minus)) if a == 0 else a**2*z_plus/sqrt(epsilon0*z_plus)
    
    # compute frequencies if they are not passed in
    if upsilon_r is None: upsilon_r = r_frequency_from_constants(a,constants,radial_roots)
    if upsilon_theta is None: upsilon_theta = theta_frequency_from_constants(a,constants,radial_roots,polar_roots)
    
    r_plus = 1+sqrt(1-a**2)
    r_minus = 1-sqrt(1-a**2)
    
    h_r = (r1-r2)/(r1-r3)
    h_plus = (r1-r2)*(r3-r_plus)/((r1-r3)*(r2-r_plus))
    h_minus = (r1-r2)*(r3-r_minus)/((r1-r3)*(r2-r_minus))
    
    # equation 13
    k_theta = 0 if a == 0 else sqrt(z_minus/z_plus)
    k_r = sqrt((r1-r2)*(r3-r4)/((r1-r3)*(r2-r4)))
    
    # equation 21
    return 4*E + 2*a2sqrt_zp_over_e0*E*upsilon_theta*(ellipk(k_theta**2)-ellipe(k_theta**2))/(pi*L) \
            + 2*upsilon_r/(pi*sqrt((1-E**2)*(r1-r3)*(r2-r4))) \
            * (E/2*((r3*(r1+r2+r3)-r1*r2)*ellipk(k_r**2) + (r2-r3)*(r1+r2+r3+r4)*_ellippi(h_r,k_r)+ (r1-r3)*(r2-r4)*ellipe(k_r**2)) \
               + 2*E*(r3*ellipk(k_r**2)+(r2-r3)*_ellippi(h_r,k_r))\
               +2/(r_plus-r_minus)*(((4*E-a*L)*r_plus-2*a**2*E)/(r3-r_plus)*(ellipk(k_r**2)-(r2-r3)/(r2-r_plus)*_ellippi(h_plus,k_r)) \
                                    -((4*E-a*L)*r_minus-2*a**2*E)/(r3-r_minus)*(ellipk(k_r**2)-(r2-r3)/(r2-r_minus)*_ellippi(h_minus,k_r))
                                   )
              )

def mino_frequencies_from_constants(a,constants,radial_roots,polar_roots):
    r"""
    Computes frequencies of orbital motion in Mino time.

    :param a: dimensionless spin of the black hole (must satisfy -1 < a < 1)
    :type a: double
    :param constants: dimensionless constants of motion for the orbit in the form :math:`(E,L,Q)`
    :type constants: tuple(double, double, double)
    :param radial_roots: tuple containing the four roots of the radial polynomial :math:`(r_1, r_2, r_3, r_4)`.
    :type radial_roots: tuple(double, double, double, double)
    :param polar_roots: tuple containing the roots of the polar equation :math:`(z_-, z_+)`
    :type polar_roots: tuple(double, double)

    :return: tuple of frequencies in the form :math:`(\Upsilon_r, \Upsilon_\theta, \Upsilon_\phi, \Gamma)`
    :rtype: tuple(double, double, double, double)
    """
    upsilon_r = r_frequency_from_constants(constants,radial_roots)
    upsilon_theta = theta_frequency_from_constants(a,constants,radial_roots,polar_roots)
    upsilon_phi = phi_frequency_from_constants(a,constants,radial_roots,polar_roots,upsilon_r,upsilon_theta)
    Gamma = gamma_from_constants(a,constants,radial_roots,polar_roots,upsilon_r,upsilon_theta)

    return upsilon_r, abs(upsilon_theta), upsilon_phi, Gamma
    
def frequencies_from_constants(a,constants,radial_roots,polar_roots):
    r"""
    Computes frequencies of orbital motion in Boyer-Lindquist time.

    :param a: dimensionless spin of the black hole (must satisfy -1 < a < 1)
    :type a: double
    :param constants: dimensionless constants of motion for the orbit in the form :math:`(E,L,Q)`
    :type constants: tuple(double, double, double)
    :param radial_roots: tuple containing the four roots of the radial polynomial :math:`(r_1, r_2, r_3, r_4)`.
    :type radial_roots: tuple(double, double, double, double)
    :param polar_roots: tuple containing the roots of the polar equation :math:`(z_-, z_+)`
    :type polar_roots: tuple(double, double)

    :return: tuple of frequencies in the form :math:`(\Omega_r, \Omega_\theta, \Omega_\phi)`
    :rtype: tuple(double, double, double)
    """
    upsilon_r = r_frequency_from_constants(constants,radial_roots)
    upsilon_theta = theta_frequency_from_constants(a,constants,radial_roots,polar_roots)
    upsilon_phi = phi_frequency_from_constants(a,constants,radial_roots,polar_roots,upsilon_r,upsilon_theta)
    Gamma = gamma_from_constants(a,constants,radial_roots,polar_roots,upsilon_r,upsilon_theta)

    return upsilon_r/Gamma, abs(upsilon_theta)/Gamma, upsilon_phi/Gamma