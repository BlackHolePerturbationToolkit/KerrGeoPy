from numpy import sort_complex, sqrt, arctan, arctan2, arccos, log, sin, cos, pi
from numpy.polynomial import Polynomial
import numpy as np
from scipy.special import ellipj, ellipeinc, ellipk
from .bound_solutions import _ellippiinc

def _plunging_radial_roots(a,E,L,Q):
    """
    Computes the radial roots for a plunging orbit. The roots are sorted in ascending order with real roots first.
    If all roots are real, the roots are sorted such that the motion is between the first two roots.

    :param a: dimensionless spin parameter
    :type a: double
    :param E: dimensionless energy
    :type E: double
    :param L: dimensionless angular momentum
    :type L: double
    :param Q: dimensionless Carter constant
    :type Q: double

    :return: tuple of radial roots
    :rtype: tuple(double,double,double,double)
    """
    # standard form of the radial polynomial R(r)
    R = Polynomial([-a**2*Q, 2*L**2+2*Q+2*a**2*E**2-4*a*E*L, a**2*E**2-L**2-Q-a**2, 2, E**2-1])
    roots = R.roots()
    # get the real roots and the complex roots
    real_roots = np.sort(np.real(roots[np.isreal(roots)]))
    complex_roots = roots[np.iscomplex(roots)]

    r_minus = 1-sqrt(1-a**2)

    if len(real_roots) == 4:
        # if there are three roots outside the event horizon swap r1/r3 and r2/r4
        if real_roots[1] > r_minus:
            r1 = real_roots[1]
            r2 = real_roots[0]
            r3 = real_roots[3]
            r4 = real_roots[2]
        else:
            r1, r2, r3, r4 = real_roots

    elif len(real_roots) == 2:
        r1, r2 = real_roots
        r3, r4 = complex_roots

    return r1, r2, r3, r4

def plunging_mino_frequencies(a,E,L,Q):
    r"""
    Computes the radial and polar mino frequencies for a plunging orbit.

    :param a: dimensionless spin parameter
    :type a: double
    :param E: dimensionless energy
    :type E: double
    :param L: dimensionless angular momentum
    :type L: double
    :param Q: dimensionless Carter constant
    :type Q: double

    :return: radial and polar mino frequencies :math:`(\Upsilon_r,\Upsilon_\theta)`
    :rtype: tuple(double,double)
    """
    r1, r2, r3, r4 = _plunging_radial_roots(a,E,L,Q)
    rho_r = np.real(r3)
    rho_i = np.imag(r4)

    # equation 42
    A = sqrt((r2-rho_r)**2+rho_i**2)
    B = sqrt((r1-rho_r)**2+rho_i**2)
    k_r = sqrt(((r2-r1)**2-(A-B)**2)/(4*A*B))

    # equation 52
    upsilon_r = pi*sqrt(A*B*(1-E**2))/(2*ellipk(k_r**2))

    # equation 14
    z1 = sqrt(1/2*(1+(L**2+Q)/(a**2*(1-E**2))-sqrt((1+(L**2+Q)/(a**2*(1-E**2)))**2-4*Q/(a**2*(1-E**2)))))
    z2 = sqrt(a**2*(1-E**2)/2*(1+(L**2+Q)/(a**2*(1-E**2))+sqrt((1+(L**2+Q)/(a**2*(1-E**2)))**2-4*Q/(a**2*(1-E**2)))))
    k_theta = a*sqrt(1-E**2)*z1/z2

    # equation 53
    upsilon_theta = pi*ellipk(k_theta**2)/(2*z2)

    return upsilon_r, upsilon_theta

def plunging_radial_integrals(a,E,L,Q):
    r"""
    Computes the radial integrals :math:`I_r`, :math:`I_{r^2}` and :math:`I_{r_\pm}` defined in equation 39 of Dyson and van de Meent (arXiv:2302.03704) as a function of the radial phase.

    :param a: dimensionless spin parameter
    :type a: double
    :param E: dimensionless energy
    :type E: double
    :param L: dimensionless angular momentum
    :type L: double
    :param Q: dimensionless Carter constant
    :type Q: double

    :return: radial integrals :math:`(I_r,I_{r^2},I_{r_+},I_{r_-})`
    :rtype: tuple(function,function,function,function)
    """
    r1, r2, r3, r4 = _plunging_radial_roots(a,E,L,Q)
    rho_r = np.real(r3)
    rho_i = np.imag(r4)

    # inner and outer horizons
    r_plus = 1+sqrt(1-a**2)
    r_minus = 1-sqrt(1-a**2)

    # equation 42
    A = sqrt((r2-rho_r)**2+rho_i**2)
    B = sqrt((r1-rho_r)**2+rho_i**2)
    f = 4*A*B/(A-B)**2
    k_r = sqrt(((r2-r1)**2-(A-B)**2)/(4*A*B))

    upsilon_r, upsilon_theta = plunging_mino_frequencies(a,E,L,Q)

    # equation 48
    D_plus = sqrt(4*A*B*(r2-r_plus)*(r_plus-r1))/(A*(r_plus-r1)+B*(r2-r_plus))
    D_minus = sqrt(4*A*B*(r2-r_minus)*(r_minus-r1))/(A*(r_minus-r1)+B*(r2-r_minus))

    def I_r(q_r):
        # equation 43
        sn, cn, dn, xi_r = ellipj(2*ellipk(k_r**2)*q_r/pi,k_r**2)
        mino_time = q_r/upsilon_r
        # equation 46
        return (A*r1-B*r2)/(A-B)*mino_time - 1/sqrt(1-E**2)*arctan((r2-r1)*sn/(2*sqrt(A*B*dn))) \
                + (A+B)*(r2-r1)/(2*(A-B)*sqrt(A*B*(1-E**2)))*_ellippiinc(xi_r,-1/f,k_r)
    
    def I_r2(q_r):
        # equation 43
        sn, cn, dn, xi_r = ellipj(2*ellipk(k_r**2)*q_r/pi,k_r**2)
        mino_time = q_r/upsilon_r
        # equation 47
        return (A*r1**2-B*r2**2)/(A-B)*mino_time + sqrt(A*B)/sqrt(1-E**2)*ellipeinc(xi_r,k_r**2) \
                - (A+B)*(A**2+2*r1**2-B**2-2*r2**2)/(4*(A-B)*sqrt((1-E**2)*A*B))*_ellippiinc(xi_r,-1/f,k_r) \
                - sqrt(A*B)*(A+B-(A-B)*cn)*sn*dn/((A-B)*sqrt(1-E**2)*(f+sn**2)) \
                + (A**2+2*r1**2-B**2-2*r2**2)/(4*(r2-r1)*sqrt(1-E**2))*arctan2(2*sn*dn*sqrt(f*(1+f*k_r**2)),f-(1+2*f*k_r**2)*sn**2)
        
    def I_r_plus(q_r):
        # equation 43
        sn, cn, dn, xi_r = ellipj(2*ellipk(k_r**2)*q_r/pi,k_r**2)
        mino_time = q_r/upsilon_r
        # equation 48
        return (A-B)*mino_time/(A*(r1-r_plus)-B*(r2-r_plus)) \
            + (r2-r1)*(A*(r1-r_plus)+B*(r2-r_plus))/(2*sqrt(A*B*(1-E**2))*(r_plus-r1)*(r2-r_plus)*(A*(r1-r_plus)-B*(r2-r_plus)))*_ellippiinc(xi_r,1/D_plus**2,k_r) \
            - (sqrt(r2-r1)*log(
                ((D_plus*sqrt(1-D_plus**2*k_r**2)+dn*sn)**2 + (k_r*(D_plus**2-sn**2))**2) /
                ((D_plus*sqrt(1-D_plus**2*k_r**2)-dn*sn)**2 + (k_r*(D_plus**2-sn**2))**2)
                )  /
                (4*sqrt((1-E**2)*(r2-r_plus)*(r_plus-r1))*sqrt(A**2*(r_plus-r1)-(r2-r_plus)*(r1**2-B**2+r2*r_plus-r1*(r2+r_plus))))
            )
    
    def I_r_minus(q_r):
        # equation 43
        sn, cn, dn, xi_r = ellipj(2*ellipk(k_r**2)*q_r/pi,k_r**2)
        mino_time = q_r/upsilon_r
        # equation 48
        return (A-B)*mino_time/(A*(r1-r_minus)-B*(r2-r_minus)) \
            + (r2-r1)*(A*(r1-r_minus)+B*(r2-r_minus))/(2*sqrt(A*B*(1-E**2))*(r_minus-r1)*(r2-r_minus)*(A*(r1-r_minus)-B*(r2-r_minus)))*_ellippiinc(xi_r,1/D_minus**2,k_r) \
            - (sqrt(r2-r1)*log(
                ((D_minus*sqrt(1-D_minus**2*k_r**2)+dn*sn)**2 + (k_r*(D_minus**2-sn**2))**2) /
                ((D_minus*sqrt(1-D_minus**2*k_r**2)-dn*sn)**2 + (k_r*(D_minus**2-sn**2))**2)
                )  /
                (4*sqrt((1-E**2)*(r2-r_minus)*(r_minus-r1))*sqrt(A**2*(r_minus-r1)-(r2-r_minus)*(r1**2-B**2+r2*r_minus-r1*(r2+r_minus))))
            )

    return I_r, I_r2, I_r_plus, I_r_minus

def plunging_radial_solutions_complex(a,E,L,Q):
    r"""
    Computes the radial solutions :math:`r(q_r), t_r(q_r), \phi_r(q_r)` from equation 50 and 51 of Dyson and van de Meent (arXiv:2302.03704) for the case of two complex radial roots.

    :param a: dimensionless spin parameter
    :type a: double
    :param E: dimensionless energy
    :type E: double
    :param L: dimensionless angular momentum
    :type L: double
    :param Q: dimensionless Carter constant
    :type Q: double

    :return: tuple of radial solutions :math:`(r(q_r), t_r(q_r), \phi_r(q_r))`
    :rtype: tuple(function, function, function)
    """
    roots = _plunging_radial_roots(a,E,L,Q)
    if np.iscomplex(roots).sum() != 2: raise ValueError("There should be two complex roots")

    r1, r2, r3, r4 = roots
    rho_r = np.real(r3)
    rho_i = np.imag(r4)

    # inner and outer horizons
    r_plus = 1+sqrt(1-a**2)
    r_minus = 1-sqrt(1-a**2)

    # equation 42
    A = sqrt((r2-rho_r)**2+rho_i**2)
    B = sqrt((r1-rho_r)**2+rho_i**2)
    f = 4*A*B/(A-B)**2
    k_r = sqrt(((r2-r1)**2-(A-B)**2)/(4*A*B))

    upsilon_r, upsilon_theta = plunging_mino_frequencies(a,E,L,Q)

    I_r, I_r2, I_r_plus, I_r_minus = plunging_radial_integrals(a,E,L,Q)

    def r(q_r):
        # equation 43
        sn, cn, dn, xi_r = ellipj(2*ellipk(k_r**2)*q_r/pi,k_r**2)
        # equation 49
        return ((A-B)*(A*r1-B*r2)*sn**2+2*A*B*(r1+r2)-2*A*B*(r2-r1)*cn)/(4*A*B+(A-B)**2*sn**2)
    
    def t_r(q_r):
        mino_time = q_r/upsilon_r
        # equation 41
        return E*(r_plus**2+r_minus**2+r_plus*r_minus+2*a**2)*mino_time + \
            E*(I_r2(q_r)+I_r(q_r)*(r_minus+r_plus)) + \
            ((r_minus**2+a**2)*(E*(r_minus**2+a**2)-a*L)/(r_minus-r_plus)*I_r_minus(q_r) + 
             (r_plus**2+a**2)*(E*(r_plus**2+a**2)-a*L)/(r_plus-r_minus)*I_r_plus(q_r)
            ) - a*L*mino_time

    def phi_r(q_r):
        mino_time = q_r/upsilon_r
        # equation 40
        return a*(
            (E*(r_minus**2+a**2)-a*L)/(r_minus-r_plus)*I_r_minus(q_r) +
            (E*(r_plus**2+a**2)-a*L)/(r_plus-r_minus)*I_r_plus(q_r)
            ) + a*E*mino_time
    
    return r, t_r, phi_r

def plunging_polar_solutions(a,E,L,Q):
    r"""
    Computes the polar solutions :math:`\theta(q_\theta), t_\theta(q_\theta), \phi_\theta(q_\theta)` from equation 33 and 37 of Dyson and van de Meent (arXiv:2302.03704).

    :param a: dimensionless spin parameter
    :type a: double
    :param E: dimensionless energy
    :type E: double
    :param L: dimensionless angular momentum
    :type L: double
    :param Q: dimensionless Carter constant
    :type Q: double
    
    :return: tuple of polar solutions :math:`(\theta(q_\theta), t_\theta(q_\theta), \phi_\theta(q_\theta))`
    :rtype: tuple(function, function, function)
    """
    z1 = sqrt(1/2*(1+(L**2+Q)/(a**2*(1-E**2))-sqrt((1+(L**2+Q)/(a**2*(1-E**2)))**2-4*Q/(a**2*(1-E**2)))))
    z2 = sqrt(a**2*(1-E**2)/2*(1+(L**2+Q)/(a**2*(1-E**2))+sqrt((1+(L**2+Q)/(a**2*(1-E**2)))**2-4*Q/(a**2*(1-E**2)))))
    k_theta = a*sqrt(1-E**2)*z1/z2

    upsilon_r, upsilon_theta = plunging_mino_frequencies(a,E,L,Q)

    def theta(q_theta):
        mino_time = q_theta/upsilon_theta
        # equation 28
        sn, cn, dn, xi_theta = ellipj(z2*mino_time,k_theta**2)
        # equation 27
        return arccos(z1*sn)
    
    def t_theta(q_theta):
        mino_time = q_theta/upsilon_theta
        # equation 28
        sn, cn, dn, xi_theta = ellipj(z2*mino_time,k_theta**2)
        # equation 35
        return E/(1-E**2)*((z2**2-a**2*(1-E**2))*mino_time-z2*ellipeinc(xi_theta,k_theta**2))
    
    def phi_theta(q_theta):
        mino_time = q_theta/upsilon_theta
        # equation 28
        sn, cn, dn, xi_theta = ellipj(z2*mino_time,k_theta**2)
        # equation 31
        return L/z2*_ellippiinc(xi_theta,z1**2,k_theta**2)
    
    return theta, t_theta, phi_theta
