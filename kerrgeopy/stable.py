"""
Module implementing the stable bound orbit solutions of `Fujita and Hikida <https://doi.org/10.48550/arXiv.0906.1420>`_
"""
from .constants import *
from .constants import _standardize_params
from .frequencies import _ellippi, _ellippiinc
from .frequencies import *
from scipy.special import ellipj, ellipeinc
from .orbit import Orbit
from numpy import pi, arccos

class StableOrbit(Orbit):
    r"""
    Class representing a stable bound orbit in Kerr spacetime.

    :param a: dimensionless angular momentum
    :type a: double
    :param p: semi-latus rectum
    :type p: double
    :param e: orbital eccentricity
    :type e: double
    :param x: cosine of the orbital inclination
    :type x: double
    :param initial_phases: tuple of initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`, defaults to (0,0,0,0)
    :type initial_phases: tuple, optional
    :param M: mass of the primary in solar masses, optional
    :type M: double
    :param mu: mass of the smaller body in solar masses, optional
    :type mu: double

    :ivar a: dimensionless angular momentum
    :ivar p: semi-latus rectum
    :ivar e: orbital eccentricity
    :ivar x: cosine of the orbital inclination
    :ivar M: mass of the primary in solar masses
    :ivar mu: mass of the smaller body in solar masses
    :ivar E: dimensionless energy
    :ivar L: dimensionless angular momentum
    :ivar Q: dimensionless carter constant
    :ivar initial_position: tuple of initial position coordinates :math:`(t_0, r_0, \theta_0, \phi_0)`
    :ivar initial_velocity: tuple of initial four-velocity components :math:`(u^t_0, u^r_0, u^\theta_0, u^\phi_0)`
    :ivar upsilon_r: dimensionless radial orbital frequency in Mino time
    :ivar upsilon_theta: dimensionless polar orbital frequency in Mino time
    :ivar upsilon_phi: dimensionless azimuthal orbital frequency in Mino time
    :ivar gamma: dimensionless time dilation factor
    :ivar omega_r: dimensionless radial orbital frequency in Boyer-Lindquist time
    :ivar omega_theta: dimensionless polar orbital frequency in Boyer-Lindquist time
    :ivar omega_phi: dimensionless azimuthal orbital frequency in Boyer-Lindquist time
    """
    def __init__(self,a,p,e,x,initial_phases=(0,0,0,0),M=None,mu=None):
        a, x = _standardize_params(a,x)
        self.a, self.p, self.e, self.x, self.initial_phases, self.M, self.mu = a, p, e, x, initial_phases, M, mu
        constants = constants_of_motion(a,p,e,x)

        self.E, self.L, self.Q = constants
        self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma = mino_frequencies(a,p,e,x)
        self.omega_r, self.omega_theta, self.omega_phi = fundamental_frequencies(a,p,e,x)
        self.stable = True

        u_t, u_r, u_theta, u_phi = self.four_velocity()
        t, r, theta, phi = self.trajectory()
        self.initial_position = t(0), r(0), theta(0), phi(0)
        self.initial_velocity = u_t(0), u_r(0), u_theta(0), u_phi(0)

    @classmethod
    def from_constants(cls,a,E,L,Q,initial_phases=(0,0,0,0),M=None,mu=None):
        """
        Alternative constructor method that takes the constants of motion as arguments.

        :param a: spin parameter
        :type a: double
        :param E: dimensionless energy
        :type E: double
        :param L: dimensionless angular momentum
        :type L: double
        :param Q: dimensionless Carter constant
        :type Q: double
        :param M: mass of the primary in solar masses
        :type M: double, optional
        :param mu: mass of the smaller body in solar masses
        :type mu: double, optional
        """
        a, p, e, x = apex_from_constants(a,E,L,Q)

        return cls(a,p,e,x,initial_phases,M,mu)
        
    def mino_frequencies(self, units="natural"):
        r"""
        Computes orbital frequencies in Mino time. Returns dimensionless frequencies in geometrized units by default.
        M and mu must be defined in order to convert to physical units.

        :param units: units to return the frequencies in (options are "natural", "mks" and "cgs"), defaults to "natural"
        :type units: str, optional

        :return: tuple of orbital frequencies in the form :math:`(\Upsilon_r, \Upsilon_\theta, \Upsilon_\phi, \Gamma)`
        :rtype: tuple(double, double, double, double)
        """
        upsilon_r, upsilon_theta, upsilon_phi, gamma = self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma
        if units == "natural":
            return upsilon_r, upsilon_theta, upsilon_phi, gamma
        
        if self.M is None: raise ValueError("M must be specified to convert frequencies to physical units")
        
        if units == "mks" or units == "cgs":
            return time_in_seconds(upsilon_r,self.M), time_in_seconds(upsilon_theta,self.M), time_in_seconds(upsilon_phi,self.M), time2_in_seconds2(gamma,self.M)
        
        raise ValueError("units must be one of 'natural', 'mks', or 'cgs'")
    
    def fundamental_frequencies(self, units="natural"):
        r"""
        Computes orbital frequencies in Boyer-Lindquist time. Returns dimensionless frequencies in geometrized units by default.
        M and mu must be defined in order to convert to physical units.

        :param units: units to return the frequencies in (options are "natural", "mks", "cgs" and "mHz"), defaults to "natural"
        :type units: str, optional
        :return: tuple of orbital frequencies in the form :math:`(\Omega_r, \Omega_\theta, \Omega_\phi)`
        :rtype: tuple(double, double, double)
        """
        upsilon_r, upsilon_theta, upsilon_phi, gamma = self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma
        if units == "natural":
            return upsilon_r/gamma, upsilon_theta/gamma, upsilon_phi/gamma
        
        if self.M is None: raise ValueError("M must be specified to convert frequencies to physical units")

        if units == "mks" or units == "cgs":
            return frequency_in_Hz(upsilon_r/gamma,self.M), frequency_in_Hz(upsilon_theta/gamma,self.M), frequency_in_Hz(upsilon_phi/gamma,self.M)
        if units == "mHz":
            return frequency_in_mHz(upsilon_r/gamma,self.M), frequency_in_mHz(upsilon_theta/gamma,self.M), frequency_in_mHz(upsilon_phi/gamma,self.M)
        
        raise ValueError("units must be one of 'natural', 'mks', 'cgs', or 'mHz'")
        
    def trajectory(self,initial_phases=None,distance_units="natural",time_units="natural"):
        r"""
        Computes the time, radial, polar, and azimuthal coordinates of the orbit as a function of mino time.

        :param initial_phases: tuple of initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        :type initial_phases: tuple, optional
        :param distance_units: units to compute the radial component of the trajectory in (options are "natural", "mks", "cgs", "au" and "km"), defaults to "natural"
        :type distance_units: str, optional
        :param time_units: units to compute the time component of the trajectory in (options are "natural", "mks", "cgs", and "days"), defaults to "natural"
        :type time_units: str, optional

        :return: tuple of functions in the form :math:`(t(\lambda), r(\lambda), \theta(\lambda), \phi(\lambda))`
        :rtype: tuple(function, function, function, function)
        """
        if initial_phases is None: initial_phases = self.initial_phases

        return stable_trajectory(self.a,self.p,self.e,self.x,initial_phases,self.M,distance_units,time_units)

    
def radial_solutions(a,constants,radial_roots):
    r"""
    Computes the radial solutions :math:`r(q_r), t^{(r)}(q_r), \phi^{(r)}(q_r)` from equation 6 of `Fujita and Hikida <https://doi.org/10.48550/arXiv.0906.1420>`_.
    :math:`q_r` is defined as :math:`q_r = \Upsilon_r \lambda = 2\pi \frac{\lambda}{\Lambda_r}`.
    Assumes the initial conditions :math:`r(0) = r_{\text{min}}` and :math:`\theta(0) = \theta_{\text{min}}`.

    :param a: dimensionless spin parameter
    :type a: double
    :param constants: tuple of constants :math:`(E,L,Q)`
    :type constants: tuple(double,double,double)
    :param radial_roots: tuple of roots :math:`(r_1,r_2,r_3,r_4)`. Assumes that motion is between :math:`r_1` and :math:`r_2` and that roots are otherwise in decreasing order.
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
            (r2-r3)*(r1+r2+r3+r4)*(_ellippiinc(psi_r,h_r,k_r**2)-q_r/pi*_ellippi(h_r,k_r**2)) \
            + (r1-r3)*(r2-r4)*(ellipeinc(psi_r,k_r**2)+h_r*sn*cn*sqrt(1-k_r**2*sn**2)/(h_r*sn**2-1) - q_r/pi*ellipe(k_r**2))
            ) 
        + 2*E*(r2-r3)*(_ellippiinc(psi_r,h_r,k_r**2)-q_r/pi*_ellippi(h_r,k_r**2)) 
        - 2/(r_plus-r_minus) * \
           (
            ((4*E-a*L)*r_plus-2*a**2*E)*(r2-r3)/((r3-r_plus)*(r2-r_plus))*(_ellippiinc(psi_r,h_plus,k_r**2)-q_r/pi*_ellippi(h_plus,k_r**2)) 
            - ((4*E-a*L)*r_minus-2*a**2*E)*(r2-r3)/((r3-r_minus)*(r2-r_minus))*(_ellippiinc(psi_r,h_minus,k_r**2)-q_r/pi*_ellippi(h_minus,k_r**2))
           )
        )
    
    def phi_r(q_r):
        # equation 27
        u_r = ellipk(k_r**2)*q_r/pi
        sn, cn, dn, psi_r = ellipj(u_r,k_r**2)
        # equation 28
        return -2*a/((r_plus-r_minus)*sqrt((1-E**2)*(r1-r3)*(r2-r4))) * \
                (
                (2*E*r_plus-a*L)*(r2-r3)/((r3-r_plus)*(r2-r_plus))*(_ellippiinc(psi_r,h_plus,k_r**2)-q_r/pi*_ellippi(h_plus,k_r**2)) 
                - (2*E*r_minus-a*L)*(r2-r3)/((r3-r_minus)*(r2-r_minus))*(_ellippiinc(psi_r,h_minus,k_r**2)-q_r/pi*_ellippi(h_minus,k_r**2))
                )
    
    return r, t_r, phi_r
        
def polar_solutions(a,constants,polar_roots):
    r"""
    Computes the polar solutions :math:`\theta(q_\theta), t^{(\theta)}(q_\theta), \phi^{(\theta)}(q_\theta)` from equation 6 of `Fujita and Hikida <https://doi.org/10.48550/arXiv.0906.1420>`_.
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
        return sign(L)*1/sqrt(e0zp)*(_ellippiinc(psi_theta,z_minus,k_theta**2)-2/pi*_ellippi(z_minus,k_theta**2)*(q_theta+pi/2))
    
    return theta, t_theta, phi_theta

def stable_trajectory(a,p,e,x,initial_phases=(0,0,0,0),M=None,distance_units="natural",time_units="natural"):
    r"""
    Computes the time, radial, polar, and azimuthal coordinates of the orbit with the given parameters as a function of mino time.

    :param a: dimensionless spin parameter
    :type a: double
    :param p: semi-latus rectum
    :type p: double
    :param e: orbital eccentricity
    :type e: double
    :param x: cosine of the orbital inclination
    :type x: double
    :param initial_phases: tuple of initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`, defaults to (0,0,0,0)
    :type initial_phases: tuple, optional
    :param M: mass of the primary in solar masses
    :type M: double, optional
    :param distance_units: units to compute the radial component of the trajectory in (options are "natural", "mks", "cgs", "au" and "km"), defaults to "natural"
    :type distance_units: str, optional
    :param time_units: units to compute the time component of the trajectory in (options are "natural", "mks", "cgs", and "days"), defaults to "natural"
    :type time_units: str, optional

    :return: tuple of functions in the form :math:`(t(\lambda), r(\lambda), \theta(\lambda), \phi(\lambda))`
    :rtype: tuple(function, function, function, function)
    """
    if ((distance_units != "natural") or (time_units != "natural")) and M is None: raise ValueError("M must be specified to convert to physical units")

    upsilon_r, upsilon_theta, upsilon_phi, gamma = mino_frequencies(a,p,e,x)
    constants = constants_of_motion(a,p,e,x)
    radial_roots = stable_radial_roots(a,p,e,x,constants)
    polar_roots = stable_polar_roots(a,p,e,x,constants)

    r_phases, t_r, phi_r = radial_solutions(a,constants,radial_roots)
    theta_phases, t_theta, phi_theta = polar_solutions(a,constants,polar_roots)
    q_t0, q_r0, q_theta0, q_phi0 = initial_phases

    distance_conversion_func = {"natural": lambda x,M: x, "mks": distance_in_meters, "cgs": distance_in_cm, "au": distance_in_au,"km": distance_in_km}
    time_conversion_func = {"natural": lambda x,M: x, "mks": time_in_seconds, "cgs": time_in_seconds, "days": time_in_days}

    # Calculate normalization constants so that t = 0 and phi = 0 at lambda = 0 when q_t0 = 0 and q_phi0 = 0 
    C_t = t_r(q_r0)+t_theta(q_theta0)
    C_phi= phi_r(q_r0)+phi_theta(q_theta0)

    def t(mino_time):
        # equation 6
        return time_conversion_func[time_units](q_t0 + gamma*mino_time + t_r(upsilon_r*mino_time+q_r0) + t_theta(upsilon_theta*mino_time+q_theta0) - C_t, M)
    
    def r(mino_time):
        return distance_conversion_func[distance_units](r_phases(upsilon_r*mino_time+q_r0),M)
    
    def theta(mino_time):
        return theta_phases(upsilon_theta*mino_time+q_theta0)
    
    def phi(mino_time):
        # equation 6
        return q_phi0 + upsilon_phi*mino_time + phi_r(upsilon_r*mino_time+q_r0) + phi_theta(upsilon_theta*mino_time+q_theta0) - C_phi
    
    return t, r, theta, phi