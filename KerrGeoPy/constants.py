from numpy import sign
from math import sqrt, pi, inf, nan

def coefficients(r,a,x):
    """
    Computes the coefficients f, g, h and d from equation B.5 in Schmidt
    
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

def coefficients_derivative(r,a,x):
    """
    Computes the derivatives f', g', h' and d' of the coefficients from equation B.5 in Schmidt
    
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

def kerr_energy(a,p,e,x):
    """
    Computes the dimensionless energy of a bound orbit with the given parameters
    
    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e <= 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)
    :type x: double
    
    :rtype: double
    """
    if e == 1:
        return 1
    if x == 0:
        return sqrt(-((p*(a**4*(-1 + e**2)**2 + (-4*e**2 + (-2 + p)**2)*p**2 + \
                2*a**2*p*(-2 + p + e**2*(2 + p))))/ \
                (a**4*(-1 + e**2)**2*(-1 + e**2 - p) + (3 + e**2 - p)*p**4 - \
                2*a**2*p**2*(-1 - e**4 + p + e**2*(2 + p)))))
    if e == 0:
        r0 = p
        f1, g1, h1, d1 = coefficients(r0,a,x)
        f2, g2, h2, d2 = coefficients_derivative(r0,a,x)
    else:
        r1 = p/(1-e)
        r2 = p/(1+e)
        f1, g1, h1, d1 = coefficients(r1,a,x)
        f2, g2, h2, d2 = coefficients(r2,a,x)
    
    kappa = d1*h2-h1*d2
    rho = f1*h2-h1*f2
    sigma = g1*h2-h1*g2
    epsilon = d1*g2-g1*d2
    eta = f1*g2-g1*f2
    
    return sqrt(
                (kappa*rho+2*epsilon*sigma-
                 sign(x)*2*sqrt(sigma*(sigma*epsilon**2+rho*epsilon*kappa-eta*kappa**2)))
                /(rho**2+4*eta*sigma)
               )

def kerr_angular_momentum(a,p,e,x):
    """
    Computes the dimensionless angular momentum of a bound orbit with the given parameters
    
    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e <= 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)
    :type x: double
    
    :rtype: double
    """
    E = kerr_energy(a,p,e,x)
    # angular momentum is zero for polar orbits
    if x == 0:
        return 0
    if e == 1:
        r2 = p/(1+e)
        f2, g2, h2, d2 = coefficients(r2,a,x)
        return (-E*g2 + sqrt(-d2*h2 + E**2*(g2**2 + f2*h2)))/h2
    else:
        r1 = p/(1-e)
        f1, g1, h1, d1 = coefficients(r1,a,x)
        return (-E*g1 + sign(x)*sqrt(-d1*h1 + E**2*(g1**2 + f1*h1)))/h1
    
def kerr_carter_constant(a,p,e,x):
    """
    Computes the dimensionless carter constant of a bound orbit with the given parameters
    
    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e <= 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)
    :type x: double
    
    :rtype: double
    """
    if x == 0:
        return -((p**2*(a**4*(-1+e**2)**2+p**4+2*a**2*p*(-2+p+e**2*(2+p)))) \
                 /(a**4*(-1+e**2)**2*(-1+e**2-p)+(3+e**2-p)*p**4-2*a**2*p**2*(-1-e**4+p+e**2*(2+p))))
    
    z = sqrt(1-x**2)
    E = kerr_energy(a,p,e,x)
    L = kerr_angular_momentum(a,p,e,x)
    return z**2 * (a**2 * (1 - E**2) + L**2/(1 - z**2))

def kerr_constants(a,p,e,x):
    """
    Computes the dimensionless energy, angular momentum, and carter constant of a bound orbit with the given parameters. Returns a tuple of the form (E, L, Q)
    
    :param a: dimensionless spin of the black hole (must satisfy 0 <= a < 1)
    :type a: double
    :param p: orbital semi-latus rectum
    :type p: double
    :param e: orbital eccentricity (must satisfy 0 <= e <= 1)
    :type e: double
    :param x: cosine of the orbital inclination (must satisfy 0 <= x^2 <= 1)
    :type x: double
    
    :rtype: tuple(double, double, double)
    """
    return kerr_energy(a,p,e,x), kerr_angular_momentum(a,p,e,x), kerr_carter_constant(a,p,e,x)