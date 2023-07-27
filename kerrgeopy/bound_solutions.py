from .constants_of_motion import *
from .frequencies_from_constants import _ellippi
from .frequencies_from_constants import *
from scipy.special import ellipj, ellipeinc
from numpy import sin, cos, arcsin, arccos, floor, where

def _ellippiinc(phi,n,k):
    r"""
    Incomplete elliptic integral of the third kind defined as :math:`\Pi(\phi,n,k) = \int_0^{\phi} \frac{1}{1-n\sin^2\theta}\frac{1}{\sqrt{1-k^2\sin^2\theta}}d\theta`.

    :type phi: double
    :type n: double
    :type k: double

    :rtype: double
    """
    # Note: sign of n is reversed from the definition in Fujita and Hikida

    # count the number of half periods
    num_cycles = floor(phi/(pi/2))
    # map phi to [0,pi/2]
    phi = abs(arcsin(sin(phi)))
    # formula from https://en.wikipedia.org/wiki/Carlson_symmetric_form
    integral = sin(phi)*elliprf(cos(phi)**2,1-k**2*sin(phi)**2,1)+1/3*n*sin(phi)**3*elliprj(cos(phi)**2,1-k**2*sin(phi)**2,1,1-n*sin(phi)**2)

    return where(num_cycles % 2 == 0, num_cycles*_ellippi(n,k)+integral, (num_cycles+1)*_ellippi(n,k)-integral)
    
def radial_solutions(a,constants,radial_roots):
    r"""
    Computes the radial solutions :math:`r(q_r), t^{(r)}(q_r), \phi^{(r)}(q_r)` from equation 6 of Fujita and Hikida (arXiv:0906.1420). 
    :math:`q_r` is defined as :math:`q_r = \Upsilon_r \lambda = 2\pi \frac{\lambda}{\Lambda_r}`.
    Assumes the initial conditions :math:`r(0) = r_{\text{min}}` and :math:`\theta(0) = \theta_{\text{min}}`.

    :param a: dimensionless spin parameter
    :type a: double
    :param constants: tuple of constants :math:`(E,L,Q)`
    :type constants: tuple(double,double,double)
    :param radial_roots: tuple of roots :math:`(r_1,r_2,r_3,r_4)`
    :type radial_roots: tuple(double,double,double,double)

    :return: tuple of functions in the form :math:`(r, t^{(r)}, \phi^{(r)})`
    :rtype: tuple(function, function, function)
    """
    E, L, Q = constants
    r1, r2, r3, r4 = radial_roots

    r_plus = 1+sqrt(1-a**2)
    r_minus = 1-sqrt(1-a**2)
    
    h_r = (r1-r2)/(r1-r3)
    h_plus = (r1-r2)*(r3-r_plus)/((r1-r3)*(r2-r_plus))
    h_minus = (r1-r2)*(r3-r_minus)/((r1-r3)*(r2-r_minus))
    
    # equation 13
    k_r = sqrt((r1-r2)*(r3-r4)/((r1-r3)*(r2-r4)))

    def r(q_r):
        # equation 27
        u_r = ellipk(k_r**2)*q_r/pi

        sn, cn, dn, psi_r = ellipj(u_r,k_r**2)
        return (r3*(r1-r2)*sn**2-r2*(r1-r3))/((r1-r2)*sn**2-(r1-r3))
    
    def t_r(q_r):
        # equation 27
        u_r = ellipk(k_r**2)*q_r/pi
        sn, cn, dn, psi_r = ellipj(u_r,k_r**2)
        # equation 28
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
        # equation 27
        u_r = ellipk(k_r**2)*q_r/pi
        sn, cn, dn, psi_r = ellipj(u_r,k_r**2)
        # equation 28
        return -2*a/((r_plus-r_minus)*sqrt((1-E**2)*(r1-r3)*(r2-r4))) * \
                (
                (2*E*r_plus-a*L)*(r2-r3)/((r3-r_plus)*(r2-r_plus))*(_ellippiinc(psi_r,h_plus,k_r)-q_r/pi*_ellippi(h_plus,k_r)) 
                - (2*E*r_minus-a*L)*(r2-r3)/((r3-r_minus)*(r2-r_minus))*(_ellippiinc(psi_r,h_minus,k_r)-q_r/pi*_ellippi(h_minus,k_r))
                )
    
    return r, t_r, phi_r
        
def polar_solutions(a,constants,polar_roots):
    r"""
    Computes the polar solutions :math:`\theta(q_\theta), t^{(\theta)}(q_\theta), \phi^{(\theta)}(q_\theta)` from equation 6 of Fujita and Hikida (arXiv:0906.1420).
    :math:`q_\theta` is defined as :math:`q_\theta = \Upsilon_\theta \lambda = 2\pi \frac{\lambda}{\Lambda_\theta}`.
    Assumes the initial conditions :math:`r(0) = r_{\text{min}}` and :math:`\theta(0) = \theta_{\text{min}}`.

    :param a: dimensionless spin parameter
    :type a: double
    :param constants: tuple of constants :math:`(E,L,Q)`
    :type constants: tuple(double,double,double)
    :param polar_roots: tuple of roots :math:`(z_-,z_+)`
    :type polar_roots: tuple(double,double)

    :return: tuple of functions in the form :math:`(\theta, t^{(\theta)}, \phi^{(\theta)})`
    :rtype: tuple(function, function, function)
    """
    E, L, Q = constants
    z_minus, z_plus = polar_roots
    epsilon0 = a**2*(1-E**2)/L**2
    # simplified form of epsilon0*z_plus
    e0zp = (a**2*(1-E**2)*(1-z_minus)+L**2)/(L**2*(1-z_minus))
    # simplified form of a**2*sqrt(z_plus/epsilon0)
    a2sqrt_zp_over_e0 = L**2/((1-E**2)*sqrt(1-z_minus)) if a == 0 else a**2*z_plus/sqrt(epsilon0*z_plus)

    # equation 13
    k_theta = 0 if a == 0 else sqrt(z_minus/z_plus)

    def theta(q_theta):
        u_theta = 2/pi*ellipk(k_theta**2)*(q_theta+pi/2)
        sn, cn, dn, ph = ellipj(u_theta,k_theta**2)
        # equation 38
        return arccos(sqrt(z_minus)*sn)

    def t_theta(q_theta):
        u_theta = 2/pi*ellipk(k_theta**2)*(q_theta+pi/2)
        sn, cn, dn, psi_theta = ellipj(u_theta,k_theta**2)
        # equation 39
        return sign(L)*a2sqrt_zp_over_e0*E/L*(2/pi*ellipe(k_theta**2)*(q_theta+pi/2)-ellipeinc(psi_theta,k_theta**2))
    
    def phi_theta(q_theta):
        sn, cn, dn, psi_theta = ellipj(2/pi*ellipk(k_theta**2)*(q_theta+pi/2),k_theta**2)
        # equation 39
        return sign(L)*1/sqrt(e0zp)*(_ellippiinc(psi_theta,z_minus,k_theta)-2/pi*_ellippi(z_minus,k_theta)*(q_theta+pi/2))
    
    return theta, t_theta, phi_theta

