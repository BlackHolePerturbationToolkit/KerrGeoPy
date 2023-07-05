from .constants import *
from .frequencies import _radial_roots, _polar_roots, _ellippi
from .frequencies import *
from scipy.special import ellipj, ellipeinc
from numpy import sin, cos, arcsin, arccos, floor, where

def _ellippiinc(phi,n,k):
    """
    Incomplete elliptic integral of the third kind.

    :type phi: double
    :type n: double
    :type k: double

    :rtype: double
    """
    # count the number of half periods
    num_cycles = floor(phi/(pi/2))
    # map phi to [0,pi/2]
    phi = abs(arcsin(sin(phi)))
    # formula from https://en.wikipedia.org/wiki/Carlson_symmetric_form
    integral = sin(phi)*elliprf(cos(phi)**2,1-k**2*sin(phi)**2,1)+1/3*n*sin(phi)**3*elliprj(cos(phi)**2,1-k**2*sin(phi)**2,1,1-n*sin(phi)**2)

    return where(num_cycles % 2 == 0, num_cycles*_ellippi(n,k)+integral, (num_cycles+1)*_ellippi(n,k)-integral)
    
def radial_solutions(a,p,e,x):
    constants = constants_of_motion(a,p,e,x)
    E, L, Q = constants
    r1, r2, r3, r4 = _radial_roots(a,p,e,constants)

    r_plus = 1+sqrt(1-a**2)
    r_minus = 1-sqrt(1-a**2)
    
    h_r = (r1-r2)/(r1-r3)
    h_plus = (r1-r2)*(r3-r_plus)/((r1-r3)*(r2-r_plus))
    h_minus = (r1-r2)*(r3-r_minus)/((r1-r3)*(r2-r_minus))
    
    # equation 13
    k_r = sqrt((r1-r2)*(r3-r4)/((r1-r3)*(r2-r4)))

    # Note: q_r = upsilon_r*lambda = 2pi*lambda/Lambda_r

    def r(q_r):
        # equation 27
        u_r = ellipk(k_r**2)*q_r/pi

        sn, cn, dn, psi_r = ellipj(u_r,k_r**2)
        return (r3*(r1-r2)*sn**2-r2*(r1-r3))/((r1-r2)*sn**2-(r1-r3))
    
    def t_r(q_r):
        sn, cn, dn, psi_r = ellipj(ellipk(k_r**2)*q_r/pi,k_r**2)
        return 2/sqrt((1-E**2)*(r1-r3)*(r2-r4))* \
        (
        E/2*(
            (r2-r3)*(r1+r2+r3+r4)*(_ellippiinc(psi_r,h_r,k_r)-q_r/pi*_ellippi(h_r,k_r)) \
            + (r1-r3)*(r2-r4)*(ellipeinc(psi_r,k_r**2)+h_r*sn*cn*sqrt(1-k_r**2*sn**2)/(h_r*sn**2-1) - q_r/pi*ellipe(k_r**2))
            ) 
        + 2*E*(r2-r3)*(_ellippiinc(psi_r,h_r,k_r)-q_r/pi*_ellippi(h_r,k_r)) 
        - 2/(r_plus-r_minus) * \
           (
            ((4*E-a*L)*r_plus-2*a**2*E)*(r2-r3)/((r3-r_plus)*(r2-r_plus))*(_ellippiinc(psi_r,h_plus,k_r)-q_r/pi*_ellippi(h_plus,k_r)) 
            - ((4*E-a*L)*r_minus-2*a**2*E)*(r2-r3)/((r3-r_minus)*(r2-r_minus))*(_ellippiinc(psi_r,h_minus,k_r)-q_r/pi*_ellippi(h_minus,k_r))
           )
        )
    
    def phi_r(q_r):
        u_r = ellipk(k_r**2)*q_r/pi
        sn, cn, dn, psi_r = ellipj(u_r,k_r**2)
        return -2*a/((r_plus-r_minus)*sqrt((1-E**2)*(r1-r3)*(r2-r4))) * \
                (
                (2*E*r_plus-a*L)*(r2-r3)/((r3-r_plus)*(r2-r_plus))*(_ellippiinc(psi_r,h_plus,k_r)-q_r/pi*_ellippi(h_plus,k_r)) 
                - (2*E*r_minus-a*L)*(r2-r3)/((r3-r_minus)*(r2-r_minus))*(_ellippiinc(psi_r,h_minus,k_r)-q_r/pi*_ellippi(h_minus,k_r))
                )
    
    return r, t_r, phi_r
        
def polar_solutions(a,p,e,x):
    constants = constants_of_motion(a,p,e,x)
    E, L, Q = constants
    epsilon0, z_minus, z_plus = _polar_roots(a,x,constants)
    # simplified form of epsilon0*z_plus
    e0zp = (a**2*(1-E**2)*(1-z_minus)+L**2)/(L**2*(1-z_minus))

    # equation 13
    k_theta = 0 if a == 0 else sqrt(z_minus/z_plus)

    # Note: q_theta = upsilon_theta*lambda = 2pi*lambda/Lambda_theta

    def theta(q_theta):
        u_theta = 2/pi*ellipk(k_theta**2)*(q_theta+pi/2)
        sn, cn, dn, ph = ellipj(u_theta,k_theta**2)
        return arccos(sqrt(z_minus)*sn)

    def t_theta(q_theta):
        u_theta = 2/pi*ellipk(k_theta**2)*(q_theta+pi/2)
        sn, cn, dn, psi_theta = ellipj(u_theta,k_theta**2)
        return a**2*E*z_plus/(L*sqrt(epsilon0*z_plus))*(2/pi*ellipe(k_theta**2)*(q_theta+pi/2)-ellipeinc(psi_theta,k_theta**2))
    
    def phi_theta(q_theta):
        sn, cn, dn, psi_theta = ellipj(2/pi*ellipk(k_theta**2)*(q_theta+pi/2),k_theta**2)
        return 1/sqrt(e0zp)*(_ellippiinc(psi_theta,z_minus,k_theta)-2/pi*_ellippi(z_minus,k_theta)*(q_theta+pi/2))
    
    return theta, t_theta, phi_theta

