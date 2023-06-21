from scipy.special import ellipk, ellipe, elliprj, elliprf
from kerrgeopy.constants import *

def _ellippi(n,k):
    """
    Complete elliptic integral of the third kind defined as :math:`\Pi(n,k) = \int_0^{\frac{\pi}{2}} \frac{d\theta}{(1-n\sin^2{\theta})\sqrt{1-k^2\sin^2{\theta}}}`
    
    :type n: double
    :type k: double

    :rtype: double
    """
    return elliprf(0,1-k**2,1)+1/3*n*elliprj(0,1-k**2,1,1-n)

def _radial_roots(a,p,e,constants):
    """
    Computes r1, r2, r3 and r4 as defined in equation 10 of Fujita and Hikida (arXiv:0906.1420)

    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e < 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    :type x: tuple(double, double, double, double)
    :param constants: dimensionless constants of motion for the orbit
    :type constants: tuple(double, double, double)

    :rtype: tuple(double, double, double, double)
    """
    E, L, Q = constants
    
    r1 = p/(1-e)
    r2 = p/(1+e)
    
    A_plus_B = 2/(1-E**2)-r1-r2
    AB = a**2*Q/(r1*r2*(1-E**2))
    
    r3 = (A_plus_B+sqrt(A_plus_B**2-4*AB))/2
    r4 = AB/r3
    
    return r1, r2, r3, r4

def _azimuthal_roots(a,x,constants):
    """
    Computes epsilon_0, z_minus and z_plus as defined in equation 10 of Fujita and Hikida (arXiv:0906.1420)

    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param x: cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    :type x: tuple(double, double, double)
    :param constants: dimensionless constants of motion for the orbit
    :type constants: tuple(double, double, double)

    :rtype: tuple(double, double, double, double)
    """
    E, L, Q = constants
    epsilon0 = a**2*(1-E**2)/L**2
    z_minus = 1-x**2
    #z_plus = a**2*(1-E**2)/(L**2*epsilon0)+1/(epsilon0*(1-z_minus))
    # simplified using definition of carter constant
    z_plus = nan if a == 0 else 1+1/(epsilon0*(1-z_minus)) 
    
    return epsilon0, z_minus, z_plus

def r_frequency(a,p,e,x,constants=None):
    """
    Computes the frequency of motion in r in Mino time using the method derived in Fujita and Hikida (arXiv:0906.1420)

    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e < 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    :type x: double
    :param constants: dimensionless constants of motion for the orbit can be passed in to speed computation if they are already known
    :type constants: tuple(double, double, double), optional

    :rtype: double
    """

    if a == 1: raise ValueError("Extreme Kerr not supported")
    if x == 0: raise ValueError("Polar orbits not supported")
    if e == 1: raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a,e,x): raise ValueError("a, e and x^2 must be between 0 and 1")
    if not is_stable(a,p,e,x): raise ValueError("Not a stable orbit")

    if constants is None: constants = constants_of_motion(a,p,e,x)
    E, L, Q = constants

    r1,r2,r3,r4 = _radial_roots(a,p,e,constants)
    # equation 13
    k_r = sqrt((r1-r2)*(r3-r4)/((r1-r3)*(r2-r4)))
    # equation 15
    return pi*sqrt((1-E**2)*(r1-r3)*(r2-r4))/(2*ellipk(k_r**2))

def theta_frequency(a,p,e,x,constants=None):
    """
    Computes the frequency of motion in theta in Mino time using the method derived in Fujita and Hikida (arXiv:0906.1420)
    
    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e < 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    :type x: double
    :param constants: dimensionless constants of motion for the orbit can be passed in to speed computation if they are already known
    :type constants: tuple(double, double, double), optional
    
    :rtype: double
    """
    if a == 1: raise ValueError("Extreme Kerr not supported")
    if x == 0: raise ValueError("Polar orbits not supported")
    if e == 1: raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a,e,x): raise ValueError("a, e and x^2 must be between 0 and 1")
    if not is_stable(a,p,e,x): raise ValueError("Not a stable orbit")

    if a == 0:
        return p/sqrt(-3-e**2+p)*(sign(x) if e == 0 else 1)
    
    if constants is None: constants = constants_of_motion(a,p,e,x)
    E, L, Q = constants
    epsilon0, z_minus, z_plus = _azimuthal_roots(a,x,constants)
    
    # equation 13
    k_theta = sqrt(z_minus/z_plus)
    # simplified form of epsilon0*z_plus
    e0zp = (a**2*(1-E**2)*(1-z_minus)+L**2)/(L**2*(1-z_minus))
    
    # equation 15
    return pi*L*sqrt(e0zp)/(2*ellipk(k_theta**2))

def phi_frequency(a,p,e,x,constants=None,upsilon_r=None,upsilon_theta=None):
    """
    Computes the frequency of motion in phi in Mino time using the method derived in Fujita and Hikida (arXiv:0906.1420)
    
    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e < 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    :type x: double
    :param constants: dimensionless constants of motion for the orbit can be passed in to speed computation if they are already known
    :type constants: tuple(double, double, double), optional
    :param upsilon_r: Mino frequency of motion in r can be passed in to speed computation if it is already known
    :type upsilon_r: double, optional
    :param upsilon_theta: Mino frequency of motion in theta can be passed in to speed computation if it is already known
    :type upsilon_theta: double, optional
    
    :rtype: double
    """
    if a == 1: raise ValueError("Extreme Kerr not supported")
    if x == 0: raise ValueError("Polar orbits not supported")
    if e == 1: raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a,e,x): raise ValueError("a, e and x^2 must be between 0 and 1")
    if not is_stable(a,p,e,x): raise ValueError("Not a stable orbit")

    if a == 0:
        return sign(x)*p/sqrt(-3-e**2+p)
    
    if constants is None: constants = constants_of_motion(a,p,e,x)
    E, L, Q = constants
    r1,r2,r3,r4 = _radial_roots(a,p,e,constants)
    epsilon0, z_minus, z_plus = _azimuthal_roots(a,x,constants)
    
    if upsilon_r is None: upsilon_r = r_frequency(a,p,e,x,constants)
    if upsilon_theta is None: upsilon_theta = theta_frequency(a,p,e,x,constants)
    
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

def gamma(a,p,e,x,constants=None,upsilon_r=None,upsilon_theta=None):
    """
    Computes the average rate at which observer time accumulates in Mino time using the method derived in Fujita and Hikida (arXiv:0906.1420)
    
    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e < 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    :type x: double
    :param constants: dimensionless constants of motion for the orbit can be passed in to speed computation if they are already known
    :type constants: tuple(double, double, double), optional
    :param upsilon_r: Mino frequency of motion in r can be passed in to speed computation if it is already known
    :type upsilon_r: double, optional
    :param upsilon_theta: Mino frequency of motion in theta can be passed in to speed computation if it is already known
    :type upsilon_theta: double, optional
    
    :rtype: double
    """
    if a == 1: raise ValueError("Extreme Kerr not supported")
    if x == 0: raise ValueError("Polar orbits not supported")
    if e == 1: raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a,e,x): raise ValueError("a, e and x^2 must be between 0 and 1")
    if not is_stable(a,p,e,x): raise ValueError("Not a stable orbit")
    
    if e == 1:
        return inf
    
    if constants is None: constants = constants_of_motion(a,p,e,x)
    E, L, Q = constants
    r1,r2,r3,r4 = _radial_roots(a,p,e,constants)
    epsilon0, z_minus, z_plus = _azimuthal_roots(a,x,constants)
    # simplified form of a**2*sqrt(z_plus/epsilon0)
    a2sqrt_zp_over_e0 = L**2/((1-E**2)*sqrt(1-z_minus)) if a == 0 else a**2*z_plus/sqrt(epsilon0*z_plus)
    
    if upsilon_r is None: upsilon_r = r_frequency(a,p,e,x,constants)
    if upsilon_theta is None: upsilon_theta = theta_frequency(a,p,e,x,constants)
    
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

def orbital_frequencies(a,p,e,x,time="Mino"):
    """
    Computes frequencies of orbital motion. Returns Mino frequencies by default.

    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e < 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    :type x: double
    :param time: specifies the time in which to compute frequencies (options are "Mino" and "Boyer-Lindquist")
    :type time: str, optional

    :rtype: tuple
    """
    constants = constants_of_motion(a,p,e,x)
    upsilon_r = r_frequency(a,p,e,x,constants)
    upsilon_theta = theta_frequency(a,p,e,x,constants)
    upsilon_phi = phi_frequency(a,p,e,x,constants,upsilon_r,upsilon_theta)
    Gamma = gamma(a,p,e,x,constants,upsilon_r,upsilon_theta)

    if time == "Mino":
        return upsilon_r, abs(upsilon_theta), upsilon_phi, Gamma
    
    if time == "Boyer-Lindquist":
        return upsilon_r/Gamma, abs(upsilon_theta)/Gamma, upsilon_phi/Gamma