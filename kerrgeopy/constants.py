from numpy import sign, sqrt, copysign, inf, nan
from math import pi
from scipy.optimize import root_scalar
from .units import *
from scipy.interpolate import RectBivariateSpline
import numpy as np

def _coefficients(r,a,x):
    """
    Computes the coefficients f, g, h and d from equation B.5 in Schmidt (arXiv:gr-qc/0202090)
    
    :param r: dimensionless distance from the black hole
    :type r: double
    :param a: dimensionless spin of the black hole
    :type a: double
    :param x: cosine of the orbital inclination
    :type x: double
    
    :rtype: tuple(double, double, double, double)
    """
    z = sqrt(1-x**2)
    delta = r**2-2*r+a**2
    f = lambda r: r**4+a**2*(r*(r+2)+z**2*delta)
    g = lambda r: 2*a*r
    h = lambda r: r*(r-2)+z**2/(1-z**2)*delta
    d = lambda r: (r**2+a**2*z**2)*delta
    
    return f(r), g(r), h(r), d(r)

def _coefficients_derivative(r,a,x):
    """
    Computes the derivatives f', g', h' and d' of the coefficients from equation B.5 in Schmidt (arXiv:gr-qc/0202090)
    
    :param r: dimensionless distance from the black hole
    :type r: double
    :param a: dimensionless spin of the black hole
    :type a: double
    :param x: cosine of the orbital inclination
    :type x: double
    
    :rtype: tuple(double, double, double, double)
    """
    z = sqrt(1-x**2)
    f_prime = lambda r: 4*r**3+2*a**2*((1+z**2)*r+(1-z**2))
    g_prime = lambda r: 2*a
    h_prime = lambda r: 2*(r-1)/(1-z**2)
    d_prime = lambda r: 2*(2*r-3)*r**2+2*a**2*((1+z**2)*r-z**2)
    
    return f_prime(r), g_prime(r), h_prime(r), d_prime(r)

def _standardize_params(a,x):
    """
    Changes signs of a and x so that a is positive and x encodes the direction of the orbit.

    :param a: dimensionless spin of the black hole
    :type a: double
    :param x: cosine of the orbital inclination
    :type x: double

    :rtype: tuple(double, double)
    """
    return abs(a), x*copysign(1,a)

def energy(a,p,e,x):
    """
    Computes the dimensionless energy of a bound orbit with the given parameters using calculations from Appendix B of Schmidt (arXiv:gr-qc/0202090)
    
    :param a: dimensionless spin of the black hole (must satisfy -1 < a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e <= 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)
    :type x: double
    
    :rtype: double
    """
    a, x = _standardize_params(a,x)

    if a == 1: raise ValueError("Extreme Kerr not supported")
    if not valid_params(a,e,x): raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a,p,e,x): raise ValueError("Not a stable orbit")

    # marginally bound case
    if e == 1:
        return 1
    
    # polar case
    if x == 0:
        # expression from ConstantsOfMotion.m in the KerrGeodesics mathematica library
        return sqrt(-((p*(a**4*(-1 + e**2)**2 + (-4*e**2 + (-2 + p)**2)*p**2 + \
                2*a**2*p*(-2 + p + e**2*(2 + p))))/ \
                (a**4*(-1 + e**2)**2*(-1 + e**2 - p) + (3 + e**2 - p)*p**4 - \
                2*a**2*p**2*(-1 - e**4 + p + e**2*(2 + p)))))
    
    # spherical case
    if e == 0:
        r0 = p
        f1, g1, h1, d1 = _coefficients(r0,a,x)
        f2, g2, h2, d2 = _coefficients_derivative(r0,a,x)
    # generic case
    else:
        r1 = p/(1-e)
        r2 = p/(1+e)
        f1, g1, h1, d1 = _coefficients(r1,a,x)
        f2, g2, h2, d2 = _coefficients(r2,a,x)
    
    # equation B.19 - B.21
    kappa = d1*h2-h1*d2
    rho = f1*h2-h1*f2
    sigma = g1*h2-h1*g2
    epsilon = d1*g2-g1*d2
    eta = f1*g2-g1*f2
    
    # equation B.22
    return sqrt(
                (kappa*rho+2*epsilon*sigma-sign(x)*2*sqrt(sigma*(sigma*epsilon**2+rho*epsilon*kappa-eta*kappa**2)))
                /(rho**2+4*eta*sigma)
               )

def angular_momentum(a,p,e,x,E=None):
    """
    Computes the dimensionless angular momentum of a bound orbit with the given parameters using calculations from Appendix B of Schmidt (arXiv:gr-qc/0202090)
    
    :param a: dimensionless spin of the black hole (must satisfy -1 < a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e <= 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)
    :type x: double
    :param E: dimensionless energy of the orbit can be passed in to speed computation if it is already known
    :type E: double, optional
    
    :rtype: double
    """
    a, x = _standardize_params(a,x)

    if a == 1: raise ValueError("Extreme Kerr not supported")
    if not valid_params(a,e,x): raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a,p,e,x): raise ValueError("Not a stable orbit")

    # compute energy if not given
    if E is None: E = energy(a,p,e,x)

    # polar case
    if x == 0:
        return 0
    
    # marginally bound case
    if e == 1:
        r2 = p/(1+e)
        f2, g2, h2, d2 = _coefficients(r2,a,x)
        # obtained by solving equation B.17 for L
        return (-E*g2 + sign(x)*sqrt(-d2*h2 + E**2*(g2**2 + f2*h2)))/h2
    
    # generic case
    else:
        r1 = p/(1-e)
        f1, g1, h1, d1 = _coefficients(r1,a,x)
        # obtained by solving equation B.17 for L
        return (-E*g1 + sign(x)*sqrt(-d1*h1 + E**2*(g1**2 + f1*h1)))/h1
    
def carter_constant(a,p,e,x,E=None,L=None):
    """
    Computes the dimensionless carter constant of a bound orbit with the given parameters using calculations from Appendix B of Schmidt (arXiv:gr-qc/0202090)
    
    :param a: dimensionless spin of the black hole (must satisfy -1 < a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e <= 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)
    :type x: double
    :param E: dimensionless energy of the orbit can be passed in to speed computation if it is already known
    :type E: double, optional
    :param L: dimensionless angular momentum of the orbit can be passed in to speed computation if it is already known
    :type L: double, optional
    
    :rtype: double
    """
    a, x = _standardize_params(a,x)

    if a == 1:  raise ValueError("Extreme Kerr not supported")
    if not valid_params(a,e,x): raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a,p,e,x): raise ValueError("Not a stable orbit")

    # polar case
    if x == 0:
        # expression from ConstantsOfMotion.m in the KerrGeodesics mathematica library
        return -((p**2*(a**4*(-1+e**2)**2+p**4+2*a**2*p*(-2+p+e**2*(2+p)))) \
                 /(a**4*(-1+e**2)**2*(-1+e**2-p)+(3+e**2-p)*p**4-2*a**2*p**2*(-1-e**4+p+e**2*(2+p))))
    
    z = sqrt(1-x**2)
    # compute energy and angular momentum if not given
    if E is None: E = energy(a,p,e,x)
    if L is None: L = angular_momentum(a,p,e,x,E)
    #  equation B.4
    return z**2 * (a**2 * (1 - E**2) + L**2/(1 - z**2))

def constants_of_motion(a,p,e,x):
    """
    Computes the dimensionless energy, angular momentum, and carter constant of a bound orbit with the given parameters. Returns a tuple of the form (E, L, Q)
    
    :param a: dimensionless spin of the black hole (must satisfy -1 < a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e <= 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)
    :type x: double
    
    :rtype: tuple(double, double, double)
    """
    E = energy(a,p,e,x)
    L = angular_momentum(a,p,e,x,E)
    Q = carter_constant(a,p,e,x,E,L)
    return E, L, Q

def _S_polar(p,a,e):
    """
    Separatrix polynomial for a polar orbit from equation 37 in Stein and Warburton (arXiv:1912.07609)

    :param p: orbital semi-latus rectum
    :type p: double
    :param a: dimensionless spin of the black hole
    :type a: double
    :param e: orbital eccentricity
    :type e: double

    :rtype: double
    """
    return      p**5*(-6 - 2*e + p) \
                + a**2*p**3*(-4*(-1+e)*(1+e)**2 + (3 + e*(2 + 3*e))*p) \
                - a**4*(1 + e)**2*p*(6 + 2*e**3 + 2*e*(-1 + p) - 3*p - 3*e**2*(2 + p)) \
                + a**6*(-1 + e)**2*(1 + e)**4

def _S_equatorial(p,a,e):
    """
    Separatrix polynomial for an equatorial orbit from equation 23 in Stein and Warburton (arXiv:1912.07609)

    :param p: orbital semi-latus rectum
    :type p: double
    :param a: dimensionless spin of the black hole
    :type a: double
    :param e: orbital eccentricity
    :type e: double

    :rtype: double
    """
    return      a**4*(-3 - 2*e + e**2)**2 \
                + p**2*(-6 - 2*e + p)**2 \
                - 2*a**2*(1 + e)*p*(14 + 2*e**2 + 3*p - e*p)

def _S(p,a,e,x):
    """
    Full separatrix polynomial from equation A1 in Stein and Warburton (arXiv:1912.07609)

    :param p: orbital semi-latus rectum
    :type p: double
    :param a: dimensionless spin of the black hole
    :type a: double
    :param e: orbital eccentricity
    :type e: double

    :rtype: double
    """
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

def separatrix(a,e,x):
    """
    Returns the value of p at the separatrix for the given orbital parameters computed using the bracked root finding method described in Stein and Warburton (arXiv:1912.07609)
    
    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param e: orbital eccentricity (must satisfy 0 <= e <= 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)
    :type x: double
    
    :rtype: double
    """

    if a == 1: raise ValueError("Extreme Kerr not supported")
    if not valid_params(a,e,x): raise ValueError("a^2, e and x^2 must be between 0 and 1")

    if a == 0:
        return 6+2*e
    
    polar_bracket = [1+sqrt(3)+sqrt(3+2*sqrt(3)), 8]
    p_polar = root_scalar(_S_polar, args=(a,e), bracket=polar_bracket)
    
    if x == 0:
        return p_polar.root
        
    equatorial_prograde_bracket = [1+e, 6+2*e]
    p_equatorial_prograde = root_scalar(_S_equatorial,args=(a,e),bracket=equatorial_prograde_bracket)
    
    if x == 1: 
        return p_equatorial_prograde.root
    
    if x == -1:
        equatorial_retrograde_bracket = [6+2*e, 5+e+4*sqrt(1+e)]
        p_equatorial_retrograde = root_scalar(_S_equatorial,args=(a,e),bracket=equatorial_retrograde_bracket)
        return p_equatorial_retrograde.root
    
    if x > 0:
        p = root_scalar(_S,args=(a,e,x),bracket=[p_equatorial_prograde.root, p_polar.root])
        return p.root
    
    if x < 0:
        p = root_scalar(_S,args=(a,e,x),bracket=[p_polar.root, 12])
        return p.root
    
def fast_separatrix(a, grid_spacing=0.01):
    """
    Constructs a faster separatrix function for a given value of a by interpolating over a grid of e and x values.

    :param a: dimensionless spin of the black hole
    :type a: double
    :param grid_spacing: spacing of the grid over which to interpolate, defaults to 0.01
    :type grid_spacing: double, optional

    :return: interpolated function of e and x
    :rtype: scipy.interpolate.RectBivariateSpline
    """
    
    # create grid of e and x values to interpolate over
    num_e_pts = int(1/grid_spacing)
    num_x_pts = int(2/grid_spacing)
    e = np.linspace(0,1,num_e_pts)
    x = np.linspace(-1,1,num_x_pts)
    E, X = np.meshgrid(e,x)

    # compute separatrix values on grid
    P = np.zeros((num_e_pts,num_x_pts))
    for i in range(num_e_pts):
        for j in range(num_x_pts):
            P[i,j] = separatrix(a,E[j,i],X[j,i])

    # create interpolator
    interpolator = RectBivariateSpline(e,x,P)

    return interpolator

def is_stable(a,p,e,x):
    """
    Tests whether or not the given orbital parameters define a stable bound orbit
    
    :param a: dimensionless spin of the black hole
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity
    :type e: double
    :param x: cosine of the orbital inclination
    :type x: double
    
    :rtype: boolean
    """
    if p > separatrix(a,e,x):
        return True
    return False

def valid_params(a,e,x):
    """
    Tests whether the given parameters fall into the allowed ranges

    :param a: dimensionless spin of the black hole
    :type a: double
    :param e: orbital eccentricity
    :type e: double
    :param x: cosine of the orbital inclination
    :type x: double

    :rtype: boolean
    """
    if (0 <= a <= 1) and (0 <= e <= 1) and (-1 <= x <= 1):
        return True
    return False

def scale_constants(constants,M,mu):
    """
    Scales the dimensionless constants of motion to the given mass parameters
    
    :param constants: dimensionless constants of motion in the form (E, L, Q)
    :type constants: tuple
    :param M: mass of the black hole
    :type M: double
    :param mu: mass ratio
    :type mu: double
    
    :rtype: tuple(double, double, double)
    """
    M = mass_in_kg(M)
    return constants[0]*mu, constants[1]*mu*M, constants[2]*mu**2*M**2