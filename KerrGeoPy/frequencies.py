from scipy.special import ellipk, ellipe, elliprj, elliprf
from kerrgeopy.constants import *

def ellippi(n,k):
    """
    Complete elliptic integral of the third kind defined as :math:`\Pi(n,k) = \int_0^{\frac{\pi}{2}} \frac{d\theta}{(1-n\sin^2{\theta})\sqrt{1-k^2\sin^2{\theta}}}`
    
    :type n: double
    :type k: double
    """
    return elliprf(0,1-k**2,1)+1/3*n*elliprj(0,1-k**2,1,1-n)

def radial_roots(a,p,e,x):
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
    """
    E, L, Q = kerr_constants(a,p,e,x)
    
    r1 = p/(1-e)
    r2 = p/(1+e)
    
    A_plus_B = 2/(1-E**2)-r1-r2
    AB = a**2*Q/(r1*r2*(1-E**2))
    
    r3 = (A_plus_B+sqrt(A_plus_B**2-4*AB))/2
    r4 = AB/r3
    
    return r1, r2, r3, r4

def azimuthal_roots(a,p,e,x):
    """
    Computes epsilon_0, z_minus and z_plus as defined in equation 10 of Fujita and Hikida (arXiv:0906.1420)

    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e < 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 < x^2 <= 1)
    :type x: tuple(double, double, double)
    """
    E, L, Q = kerr_constants(a,p,e,x)
    epsilon0 = a**2*(1-E**2)/L**2
    z_minus = 1-x**2
    #z_plus = a**2*(1-E**2)/(L**2*epsilon0)+1/(epsilon0*(1-z_minus))
    # simplified using definition of carter constant
    z_plus = nan if a == 0 else 1+1/(epsilon0*(1-z_minus)) 
    
    return epsilon0, z_minus, z_plus

def r_frequency(a,p,e,x):
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
    """
    E, L, Q = kerr_constants(a,p,e,x)
    r1,r2,r3,r4 = radial_roots(a,p,e,x)
    # equation 13
    k_r = sqrt((r1-r2)*(r3-r4)/((r1-r3)*(r2-r4)))
    # equation 15
    return pi*sqrt((1-E**2)*(r1-r3)*(r2-r4))/(2*ellipk(k_r**2))

def theta_frequency(a,p,e,x):
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
    
    :rtype: double
    """
    if a == 0:
        return p/sqrt(-3-e**2+p)*(sign(x) if e == 0 else 1)
    
    E, L, Q = kerr_constants(a,p,e,x)
    epsilon0, z_minus, z_plus = azimuthal_roots(a,p,e,x)
    
    # equation 13
    k_theta = sqrt(z_minus/z_plus)
    # simplified form of epsilon0*z_plus
    e0zp = (a**2*(1-E**2)*(1-z_minus)+L**2)/(L**2*(1-z_minus))
    
    # equation 15
    return pi*L*sqrt(e0zp)/(2*ellipk(k_theta**2))

def phi_frequency(a,p,e,x):
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
    
    :rtype: double
    """
    if a == 0:
        return sign(x)*p/sqrt(-3-e**2+p)
    
    E, L, Q = kerr_constants(a,p,e,x)
    r1,r2,r3,r4 = radial_roots(a,p,e,x)
    epsilon0, z_minus, z_plus = azimuthal_roots(a,p,e,x)
    
    upsilon_r = r_frequency(a,p,e,x)
    upsilon_theta = theta_frequency(a,p,e,x)
    
    # simplified form of epsilon0*z_plus
    e0zp = (a**2*(1-E**2)*(1-z_minus)+L**2)/(L**2*(1-z_minus))

    r_plus = 1+sqrt(1-a**2)
    r_minus = 1-sqrt(1-a**2)
    
    h_r = (r1-r2)/(r1-r3)
    h_plus = (r1-r2)*(r3-r_plus)/((r1-r3)*(r2-r_plus))
    h_minus = (r1-r2)*(r3-r_minus)/((r1-r3)*(r2-r_minus))
    
    # equation 13
    k_theta = sqrt(z_minus/z_plus)
    k_r = sqrt((r1-r2)*(r3-r4)/((r1-r3)*(r2-r4)))
    
    # equation 21
    return  2*upsilon_theta/(pi*sqrt(e0zp))*ellippi(z_minus,k_theta) \
            + 2*a*upsilon_r/(pi*(r_plus-r_minus)*sqrt((1-E**2)*(r1-r3)*(r2-r4))) \
            * ((2*E*r_plus-a*L)/(r3-r_plus)*(ellipk(k_r**2)-(r2-r3)/(r2-r_plus)*ellippi(h_plus,k_r)) \
               - (2*E*r_minus-a*L)/(r3-r_minus)*(ellipk(k_r**2)-(r2-r3)/(r2-r_minus)*ellippi(h_minus,k_r))
              )

def gamma(a,p,e,x):
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
    
    :rtype: double
    """
    if e == 1:
        return inf
    
    E, L, Q = kerr_constants(a,p,e,x)
    r1,r2,r3,r4 = radial_roots(a,p,e,x)
    epsilon0, z_minus, z_plus = azimuthal_roots(a,p,e,x)
    # simplified form of a**2*sqrt(z_plus/epsilon0)
    a2sqrt_zp_over_e0 = L**2/((1-E**2)*sqrt(1-z_minus)) if a == 0 else a**2*z_plus/sqrt(epsilon0*z_plus)
    
    upsilon_r = r_frequency(a,p,e,x)
    upsilon_theta = theta_frequency(a,p,e,x)
    
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
            * (E/2*((r3*(r1+r2+r3)-r1*r2)*ellipk(k_r**2) + (r2-r3)*(r1+r2+r3+r4)*ellippi(h_r,k_r)+ (r1-r3)*(r2-r4)*ellipe(k_r**2)) \
               + 2*E*(r3*ellipk(k_r**2)+(r2-r3)*ellippi(h_r,k_r))\
               +2/(r_plus-r_minus)*(((4*E-a*L)*r_plus-2*a**2*E)/(r3-r_plus)*(ellipk(k_r**2)-(r2-r3)/(r2-r_plus)*ellippi(h_plus,k_r)) \
                                    -((4*E-a*L)*r_minus-2*a**2*E)/(r3-r_minus)*(ellipk(k_r**2)-(r2-r3)/(r2-r_minus)*ellippi(h_minus,k_r))
                                   )
              )

def kerr_frequencies(a,p,e,x,time="Mino"):
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
    :type time: string

    :rtype: tuple
    """
    if time == "Mino":
        return r_frequency(a,p,e,x), abs(theta_frequency(a,p,e,x)), phi_frequency(a,p,e,x), gamma(a,p,e,x)
    
    if time == "Boyer-Lindquist":
        Gamma = gamma(a,p,e,x)
        return r_frequency(a,p,e,x)/Gamma, abs(theta_frequency(a,p,e,x))/Gamma, phi_frequency(a,p,e,x)/Gamma