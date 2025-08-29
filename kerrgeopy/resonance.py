"""Module containing functions for finding resonance orbits
"""
from .frequencies import *
from .frequencies import _ellippi
from .constants import *
from .constants import _standardize_params
from scipy.optimize import fsolve, newton, root_scalar
from scipy.special import ellipk, elliprc
from numpy import sqrt, pi, sign, exp

def valid_integers(r, theta, phi):
    '''
    Check if the three integers are valid 
    '''
    if r < 0:
        return False
    if (r!=0) & (theta!=0) & (phi!=0):
        if (r<abs(theta)) & (r<abs(phi)):
            return True
    elif (r==0) & (theta!=0) & (phi!=0):
        if (theta*phi>0) & (theta<phi):
            return True
    elif (r!=0) & (theta==0) & (phi!=0):
        if r<abs(phi):
            return True
    elif (r!=0) & (theta!=0) & (phi==0):
        if r<abs(theta):
            return True
    else:
        return False

# Frequency ratio functions
## near-separatrix functions
def _dr2_minus_dr3_ps(a, e, x, ps=None, constants=None):
    """
    d(r_2-r_3)/dp evaluated at separatrix.
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    ps : double
        orbital semi-latus rectum at the separatrix
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit at the separatrix
        
    Returns
    -------
    double
    
    """
    if not ps:
        ps = separatrix(a,e,x)
    if not constants:
        constants = constants_of_motion(a, ps, e, x)   
    r1, r2, r3, r4 = stable_radial_roots(a, ps, e, x, constants)
    r3 = r2
    S = r3+r4
    P = r3*r4
    A11 = (
        4 * S * (a**2 * (e**2 - 1) * (e**2 - ps - 1) + ps * (e**2 * (ps + 4) - ps**2 + 3*ps - 4))
        +8 * ps * (a**2 - ps) * (e**2 + ps - 1)
        -8 * P * (e**2 + ps - 1)**2
    )
    A12 = (
        4 * P * (a**2 * (e**2 - 1) * (e**2 - ps - 1) + ps * (e**2 * (ps + 4) - ps**2 + 3*ps - 4))
        +2 * S * (-a**4 * (e**2 - 1)**2 - 2*a**2 * (e**2 - 1) * ps * (ps + 4) - (ps - 4)**2 * ps**2)
        +4 * ps * (a**4 * (e**2 - 1) + a**2 * ps * (e**2 + ps + 3) + (ps - 4) * ps**2)
    )
    A21 = ps**2 - a**2 * (e**2 - 1) * (x**2 - 1)
    A22 = 2 * a**2 * ps * (x**2 - 1)
    B1 = (
        8 * (e**2 - 1) * P * (P - 2*S)
        +4 * ps**3 * (S - 2)**2
        +4 * a**4 * (e**2 * (-S) + 2*ps + S)
        +8 * ps * (-P * (e**2 * (S - 2) + 3*S + 2) + P**2 + 4*S**2)
        +12 * ps**2 * (P * (S + 2) - 2*(S - 2)*S)
        -4 * a**2 * (
            3 * ps**2 * (S + 2) - 
            (e**2 - 1) * (P * (S - 2) + 2*S**2) + 
            ps * (S * (-(e**2 * (S - 2)) + S + 6) + 4*P)
        )
    )
    B2 = -2 * (a**2 * (x**2 - 1) * (ps + S) + ps * P)
    det = A11*A22-A12*A21
    dP = (A22*B1-A12*B2)/det
    dS = (-A21*B1+A11*B2)/det
    return 1/(1+e)-(dP-r3*dS)/(S-2*r3)

def _C(a, e, x,  ps=None, constants=None):
    '''
    (r_1-r_4)/(r_1-r_3)/(r_2-r_4) evaluated at separatrix p=p_s
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    ps : double
        orbital semi-latus rectum at the separatrix
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit at the separatrix
        
    Returns
    -------
    double
    
    '''
    if not ps:
        ps = separatrix(a,e,x)
    if not constants:
        constants = constants_of_motion(a,ps,e,x)
    r1, r2, r3, r4  = stable_radial_roots(a, ps, e, x, constants)
    r3 = r2
    if e == 1:
        return 1/(r2-r4)
    return (r1-r4)/(r1-r3)/(r2-r4)

def _Frtheta(a, e, x, ps=None, constants=None):
    '''
    Frtheta= Upsilon_r/Upsilon_theta*K(k_r), evaluated at separatrix p=p_s
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    ps : double
        orbital semi-latus rectum at the separatrix
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit at the separatrix
        
    Returns
    -------
    double
    '''
    if not ps:
        ps = separatrix(a,e,x)
    if not constants:
        constants = constants_of_motion(a, ps, e, x)
    E, L, Q = constants
    r1, r2, r3, r4= stable_radial_roots(a, ps, e, x, constants)
    r3 = r2
    z_minus, z_plus = stable_polar_roots(a, ps, e, x, constants)
    if e == 1:
        return 2*sqrt(2*(r2-r4))/sqrt(L**2+Q)*pi/4
    else:
        return sqrt((r1-r3)*(r2-r4)/(a**2*z_plus))*ellipk(z_minus/z_plus)
    
def _FK(a, e, x, ps=None, constants=None):
    '''
    Terms that have a common factor K(k_r) in Upsilon_phi/Upsilon_r, evaluated at separatrix.
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    ps : double
        orbital semi-latus rectum at the separatrix
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit at the separatrix
        
    Returns
    -------
    double
    
    '''
    if not ps:
        ps = separatrix(a,e,x)
    if not constants:
        constants = constants_of_motion(a, ps, e, x)
    E, L, Q = constants
    r1, r2, r3, r4= stable_radial_roots(a, ps, e, x, constants)
    r3 = r2
    z_minus, z_plus = stable_polar_roots(a, ps, e, x, constants)
    
    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)
    
    term1 = 1/_Frtheta(a, e, x, ps, constants)
    
    if e == 1:
        term2 = 2*a/(pi*sqrt(2*(r2-r4)))*(2*E*r3-a*L)/(r3-r_plus)/(r3-r_minus)
        if x==0:
            return term1+term2, term1-term2
        else:
            return term1+sign(x)*term2
    else:
        term2 = 2*a/(pi*sqrt((1-E**2)*(r1-r3)*(r2-r4)))*(2*E*r3-a*L)/(r3-r_plus)/(r3-r_minus)
        if x == 0:
            return term1+term2, term1-term2
        else:
            e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))
            return  2/(pi*sqrt(e0zp))*_ellippi(z_minus, z_minus/z_plus)*term1+sign(x)*term2
        
def _FPlus(a, e, x, ps=None, constants=None):
    '''
    Terms that have a common factor Pi(h_plus, k_r) in Upsilon_phi/Upsilon_r, evaluated at separatrix.
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    ps : double
        orbital semi-latus rectum at the separatrix
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit at the separatrixs
        
    Returns
    -------
    double
    
    '''
    if not ps:
        ps = separatrix(a,e,x)
    if not constants:
        constants = constants_of_motion(a, ps, e, x)
    E, L, Q = constants
    r1, r2, r3, r4= stable_radial_roots(a, ps, e, x, constants)
    r3 = r2
    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)
    
    if e == 1:
        term1 = (
            -2*a/(pi*sqrt(2*(r2-r4)))
            *(2*E*r_plus-a*L)/(r3-r_plus)/(r_plus-r_minus)
            *r3/r_plus
        )
        term2 = (r2-r4)/(r2-r_plus)
    else:
        term1 = (
            -2*a/(pi*sqrt((1-E**2)*(r1-r3)*(r2-r4)))
            *(2*E*r_plus-a*L)/(r3-r_plus)/(r_plus-r_minus)
            *(r1-r3)/(r1-r_plus)
        )
        term2 = (r1-r_plus)/(r2-r_plus)*(r2-r4)/(r1-r4) 
    
    # return term1*(log(2)+1/2*log(term2))
    return term1*sqrt(term2)*elliprc(term2, 1)

def _FMinus(a, e, x, ps=None, constants=None):
    '''
    Terms that have a common factor Pi(h_minus, k_r) in Upsilon_phi/Upsilon_r, evaluated at separatrix.
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    ps : double
        orbital semi-latus rectum at the separatrix
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit at the separatrix
        
    Returns
    -------
    double
    
    '''
    if not ps:
        ps = separatrix(a,e,x)
    if not constants:
        constants = constants_of_motion(a, ps, e, x)
    E, L, Q = constants
    r1, r2, r3, r4= stable_radial_roots(a, ps, e, x, constants)
    r3 = r2
    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)
    
    if e == 1:
        term1 = (
            2*a/(pi*sqrt(2*(r2-r4)))
            *(2*E*r_minus-a*L)/(r3-r_minus)/(r_plus-r_minus)
            *r3/r_minus
            )
        term2 = (r2-r4)/(r2-r_minus)
    else:
        term1 = (
            2*a/(pi*sqrt((1-E**2)*(r1-r3)*(r2-r4)))
            *(2*E*r_minus-a*L)/(r3-r_minus)/(r_plus-r_minus)
            *(r1-r3)/(r1-r_minus)
        )
        term2 = (r1-r_minus)/(r2-r_minus)*(r2-r4)/(r1-r4)
    # return term1*(log(2)+1/2*log(term2))
    return term1*sqrt(term2)*elliprc(term2, 1)

def _phitheta_boundRatio(a, e, x, ps=None, constants=None, xSign=None):
    '''
    Maximum/minimum of phi-theta frequency ratio of prograde/retrograde orbits.
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    ps : double
        orbital semi-latus rectum at the separatrix
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit at the separatrix
    xSign : float
        sign of x (needed for polar orbits)
        
    Returns
    -------
    double
    
    '''
    if not ps:
        ps = separatrix(a,e,x)
    if not constants:
        constants = constants_of_motion(a, ps, e, x)
    E, L, Q = constants
    r1, r2, r3, r4= stable_radial_roots(a, ps, e, x, constants)
    r3 = r2
    z_minus, z_plus = stable_polar_roots(a, ps, e, x, constants)
    
    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)
    
    term = (
            2*a*ellipk(z_minus/z_plus)/(pi*sqrt(a**2*x**2*(1-E**2)+L**2+Q))
            *(2*E*r3-a*L)/(r3-r_plus)/(r3-r_minus)
        )
    if x == 0:
        if not xSign:
            return 1+term, 1-term
        else:
            if xSign == 1:
                return 1+term
            elif xSign == -1:
                return 1-term

    else:
        if e == 1:
            return 1+sign(x)*term
        else:
            e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))
            return  2/(pi*sqrt(e0zp))*_ellippi(z_minus, z_minus/z_plus)+sign(x)*term


## near-separatrix initial guesses

def _rtheta_nearSeparatrix_initialGuess(a, e, x, ratio, ps=None, constants=None):
    '''
    Initial guess of orbital semi-latus rectum for solving numerically r-theta resonance whose root is very close to the separatrix.
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    ratio: double
        r-theta frequency ratio
    ps : double
        orbital semi-latus rectum at the separatrix
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit at the separatrix
        
    Returns
    -------
    double
    
    '''
    if not ps:
        ps = separatrix(a,e,x)
    if e == 0:
        return ps
    if not constants:
        constants = constants_of_motion(a, ps, e, x)
    term = 16/_C(a,e,x,ps,constants)*exp(-2*_Frtheta(a,e,x,ps,constants)/abs(ratio))
    return (
        ps + 16/_dr2_minus_dr3_ps(a,e,x,ps,constants)/_C(a,e,x,ps,constants)
            *exp(-2*_Frtheta(a,e,x,ps,constants)/abs(ratio))
    )

def _rphi_nearSeparatrix_initialGuess(a, e, x, ratio, ps=None, constants=None):
    '''
    Initial guess of orbital semi-latus rectum for solving numerically r-phi resonance whose root is very close to the separatrix.
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    ratio: double
        r-phi frequency ratio
    ps : double
        orbital semi-latus rectum at the separatrix
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit at the separatrix
        
    Returns
    -------
    double
    
    '''
    if not ps:
        ps = separatrix(a,e,x)
    if e == 0:
        return ps
    if not constants:
        constants = constants_of_motion(a, ps, e, x)
    
    if x == 0:
        if ratio > 0:
            FK = _FK(a,e,x,ps,constants)[0]
        elif ratio < 0:
            FK = _FK(a,e,x,ps,constants)[1]
    else:
        FK = _FK(a,e,x,ps,constants)
    return (
        ps + 16/_dr2_minus_dr3_ps(a,e,x,ps,constants)/_C(a,e,x,ps,constants)
            *exp(2/FK*sign(ratio)*(_FPlus(a,e,x,ps,constants)+_FMinus(a,e,x,ps,constants)-1/ratio))
    )

def _phitheta_nearSeparatrix_initialGuess(a, e, x, ratio, ps=None, constants=None):
    '''
    Initial guess of orbital semi-latus rectum for solving numerically phi-theta resonance whose root is very close to the separatrix.
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    ratio: double
        phi-theta frequency ratio
    ps : double
        orbital semi-latus rectum at the separatrix  
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit at the separatrix
    Returns
    -------
    double
    
    '''
    if not ps:
        ps = separatrix(a,e,x)
    if e == 0:
        return ps
    if not constants:
        constants = constants_of_motion(a, ps, e, x)

    if x == 0:
        if ratio > 1:
            FK = _FK(a,e,x,ps,constants)[0]
        elif ratio < 1:
            FK = _FK(a,e,x,ps,constants)[1]
    else:
        FK = _FK(a,e,x,ps,constants)
        
    Frtheta = _Frtheta(a,e,x,ps,constants)
    if ratio == FK*Frtheta:
        return ps
    else:
        return (
            ps + 16/_dr2_minus_dr3_ps(a,e,x,ps,constants)/_C(a,e,x,ps,constants)
                *exp(2*Frtheta*(_FPlus(a,e,x,ps,constants)+_FMinus(a,e,x,ps,constants))
                    /abs(FK*Frtheta-ratio))
        ) 

## general frequency ratios
def _rtheta_frequencyRatio(a, p, e, x, constants=None):
    """Ratio of r-frequency and theta-frequency times sign(x) 

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit
        
    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)
    
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a,e,x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a,p,e,x):
        raise ValueError("Not a stable orbit")
    if p == separatrix(a,e,x):
        return 0
    
    constants = constants_of_motion(a,p,e,x)
    E, L, Q = constants
    if a == 0:
        return r_frequency(a,p,e,x,constants)/theta_frequency(a,p,e,x,constants)
    if e == 1:
        r1, r2, r3, r4 = stable_radial_roots(a, p, e, x, constants)
        k_r = sqrt((r3-r4)/(r2-r4))
        return 2*sqrt(2*(r2-r4))/sqrt(L**2+Q)*pi/4/ellipk(k_r**2)
    
    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x, constants)
    z_minus, z_plus= stable_polar_roots(a, p, e, x, constants)
    k_r = sqrt((r1-r2)*(r3-r4) / ((r1-r3)*(r2-r4)))
    k_theta = sqrt(z_minus/z_plus)
    return sqrt((r1-r3)*(r2-r4)/(a**2*z_plus))*ellipk(k_theta**2)/ellipk(k_r**2)

def _rphi_frequencyRatio(a, p, e, x, constants=None):
    """Ratio of r-frequency and phi-frequency times sign(x)

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination
    constants : tuple(double, double, double)
        dimensionless constants of motion for the orbit
        
    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)
    
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a,e,x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a,p,e,x):
        raise ValueError("Not a stable orbit")
    if p == separatrix(a,e,x):
        return 0
    
    if not constants:
        constants = constants_of_motion(a,p,e,x)
    E, L, Q = constants
    if a == 0:
        return r_frequency(a,p,e,x,constants)/abs(phi_frequency(a,p,e,x,constants))
    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x, constants)
    z_minus, z_plus = stable_polar_roots(a, p, e, x, constants)
    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)
    
    if e == 1:
        k_r = sqrt((r3-r4)/(r2-r4))
        h_plus =  (r3-r_plus)/(r2-r_plus)
        h_minus = (r3-r_minus)/(r2-r_minus)
        B_r = 2*a/(pi*(r_plus-r_minus)*sqrt(2*(r2-r4))
        ) * (
        (2*E*r_plus-a*L)/(r3-r_plus)
        * (ellipk(k_r**2) - (r2-r3)/(r2-r_plus) * _ellippi(h_plus, k_r**2))
        - (2*E*r_minus-a*L)/(r3-r_minus)
        * (ellipk(k_r**2) - (r2-r3)/(r2-r_minus) * _ellippi(h_minus, k_r**2))
        ) 
        ratio_rtheta = 2*sqrt(2*(r2-r4))/sqrt(L**2+Q)*pi/4/ellipk(k_r**2)
        if x == 0:
            return 1/(1/ratio_rtheta+B_r), 1/(1/ratio_rtheta-B_r)
        else:
            return 1/(1/ratio_rtheta+sign(x)*B_r)
        
    k_r = sqrt((r1-r2)*(r3-r4) / ((r1-r3)*(r2-r4)))
    h_plus =  (r1-r2)*(r3-r_plus) / ((r1-r3)*(r2-r_plus))
    h_minus = (r1-r2)*(r3-r_minus) / ((r1-r3)*(r2-r_minus))
    k_theta = sqrt(z_minus/z_plus)
    B_r = 2*a/(pi*(r_plus-r_minus)*sqrt((1-E**2)*(r1-r3)*(r2-r4))
    ) * (
    (2*E*r_plus-a*L)/(r3-r_plus)
    * (ellipk(k_r**2) - (r2-r3)/(r2-r_plus) * _ellippi(h_plus, k_r**2))
    - (2*E*r_minus-a*L)/(r3-r_minus)
    * (ellipk(k_r**2) - (r2-r3)/(r2-r_minus) * _ellippi(h_minus, k_r**2))
    ) 
    ratio_rtheta = sqrt((r1-r3)*(r2-r4)/(a**2*z_plus))*ellipk(k_theta**2)/ellipk(k_r**2)
    if x == 0:
        return 1/(1/ratio_rtheta+B_r), 1/(1/ratio_rtheta-B_r)
    else: 
        e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))
        B_theta = 2/(pi*sqrt(e0zp))*_ellippi(z_minus, z_minus/z_plus)
        return  1/(B_theta/ratio_rtheta+sign(x)*B_r)
    
def _phitheta_frequencyRatio(a, p, e, x, constants=None):
    """Ratio of r-frequency and phi-frequency 

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)
    
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a,e,x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a,p,e,x):
        raise ValueError("Not a stable orbit")
    if p == separatrix(a,e,x):
        return _phitheta_boundRatio(a, e, x)
    if not constants:
        constants = constants_of_motion(a,p,e,x)
    E, L, Q = constants
    if a == 0:
        return phi_frequency(a,p,e,x,constants)/abs(theta_frequency(a,p,e,x,constants))
    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x, constants)
    z_minus, z_plus = stable_polar_roots(a, p, e, x, constants)
    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)
    
    if e == 1:
        k_r = sqrt((r3-r4)/(r2-r4))
        h_plus =  (r3-r_plus)/(r2-r_plus)
        h_minus = (r3-r_minus)/(r2-r_minus)
        B_r = 2*a/(pi*(r_plus-r_minus)*sqrt(2*(r2-r4))
        ) * (
        (2*E*r_plus-a*L)/(r3-r_plus)
        * (ellipk(k_r**2) - (r2-r3)/(r2-r_plus) * _ellippi(h_plus, k_r**2))
        - (2*E*r_minus-a*L)/(r3-r_minus)
        * (ellipk(k_r**2) - (r2-r3)/(r2-r_minus) * _ellippi(h_minus, k_r**2))
        ) 
        ratio_rtheta = 2*sqrt(2*(r2-r4))/sqrt(L**2+Q)*pi/4/ellipk(k_r**2)
        if x == 0:
            return 1+B_r*ratio_rtheta, 1-B_r*ratio_rtheta
        else:
            return 1+sign(x)*B_r*ratio_rtheta

    k_r = sqrt((r1-r2)*(r3-r4) / ((r1-r3)*(r2-r4)))
    h_plus =  (r1-r2)*(r3-r_plus) / ((r1-r3)*(r2-r_plus))
    h_minus = (r1-r2)*(r3-r_minus) / ((r1-r3)*(r2-r_minus))
    k_theta = sqrt(z_minus/z_plus)
    B_r = 2*a/(pi*(r_plus-r_minus)*sqrt((1-E**2)*(r1-r3)*(r2-r4))
    ) * (
    (2*E*r_plus-a*L)/(r3-r_plus)
    * (ellipk(k_r**2) - (r2-r3)/(r2-r_plus) * _ellippi(h_plus, k_r**2))
    - (2*E*r_minus-a*L)/(r3-r_minus)
    * (ellipk(k_r**2) - (r2-r3)/(r2-r_minus) * _ellippi(h_minus, k_r**2))
    ) 
    ratio_rtheta = sqrt((r1-r3)*(r2-r4)/(a**2*z_plus))*ellipk(k_theta**2)/ellipk(k_r**2)
    if x == 0:
        return 1+B_r*ratio_rtheta, 1-B_r*ratio_rtheta
    else: 
        e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))
        B_theta = 2/(pi*sqrt(e0zp))*_ellippi(z_minus, k_theta**2)
        return  B_theta+sign(x)*B_r*ratio_rtheta

# Finding double resonance
## General solver
def _doubleResonance_solver(equation, pWF, pNS, ps):
    '''Try `scipy.newton` with two different initial guess to solve resonant condition for orbital semi-latus rectum,
    one of them is very near the separatrix.
    Also try using `scipy.root_scalar(method="brentq")`. If the root is very close to separatrix, approximate it to pNS.
    
    Parameters
    ----------
    equation: callable
        resonant condition equation needed to be solved
    pWF : double
        weak-field initial guess
    pNS: double
        near-separatrix initial guess
    ps: double
        orbital semi-latus rectum at the separatrix
    
    Returns
    -------
    p : double
        orbital semi-latus rectum
    '''
    exceptErrors = (RuntimeError, ValueError)
    if pWF > ps:
        try:
            return newton(equation, pWF)
        except exceptErrors:
            pass
    try:
        return newton(equation, pNS)
    except exceptErrors:
        pass
    try:    
        return root_scalar(equation, bracket=[ps, (pNS-ps)/16*exp(pi)+ps], method="brentq").root
    except exceptErrors:
        pass
    print("p is approximated to near-separatrix initial guess")
    return pNS


## r-theta resonance
def _rtheta_resonance_p(a, e, x, rInteger, thetaInteger):
    """Find p of r-theta resonant orbit with given (a,e,x,r-integer,theta-integer) such that
    r-frequency/theta-frequency = r-integer/theta-integer
    
    Parameters
    ----------
    rInteger: integer
        r-integer
    thetaInteger: integer
        theta-integer
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    p : double
        orbital semi-latus rectum
    """
    
    a, x = _standardize_params(a, x)
    if not valid_integers(rInteger, thetaInteger, 0):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported") 
    ratio = rInteger/thetaInteger
    if (x != 0) & (sign(x) != sign(ratio)):
            raise ValueError("Require sign(x) = sign(rInteger/thetaInteger)")
    ps = separatrix(a,e,x)
    pWF = 6/(1-ratio**2)
    pNS = _rtheta_nearSeparatrix_initialGuess(a,e,x,rInteger/thetaInteger,ps)
    return _doubleResonance_solver(lambda p: _rtheta_frequencyRatio(a, p, e, x)*abs(thetaInteger)-rInteger,pWF,pNS,ps)

def _rtheta_resonance_e(a, p, x, rInteger, thetaInteger):
    """Find e of r-theta resonant orbit with given (a,p,x,r-integer,theta-integer) such that
    r-frequency/theta-frequency = r-integer/theta-integer
    
    Parameters
    ----------
    rInteger: integer
        r-integer
    thetaInteger: integer
        theta-integer
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    x : double
        cosine of the orbital inclination

    Returns
    -------
    e : double
        orbital eccentricity
    """
    
    a, x = _standardize_params(a, x)
    if not valid_integers(rInteger, thetaInteger, 0):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported") 
    ratio = rInteger/thetaInteger
    if sign(x)*sign(ratio)<0:
            raise ValueError("Require sign(x) = sign(rInteger/thetaInteger)")
    p0 = _rtheta_resonance_p(a, 0, x, rInteger, thetaInteger)
    p1 = _rtheta_resonance_p(a, 1, x, rInteger, thetaInteger)
    if (p<p0)|(p1<p):
        print(f"p must be in [{p0:f},{p1:f}]")
        return nan
    if p == p0:
        return 0
    elif p == p1:
        return 1
    else:
        e_guess = (p-p0)/(p1-p0)
        return newton(lambda e: _rtheta_frequencyRatio(a, p, e, x)*abs(thetaInteger)-rInteger, e_guess)

def _rtheta_resonance_x(a, p, e, rInteger, thetaInteger):
    """Find x of r-theta resonant orbit with given (a,p,e,r-integer,theta-integer) such that
    r-frequency/theta-frequency = r-integer/theta-integer
    
    Parameters
    ----------
    rInteger: integer
        r-integer
    thetaInteger: integer
        theta-integer
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    
    Returns
    -------
    x : double
        cosine of the orbital inclination
    """
    
    a = abs(a)
    if not valid_integers(rInteger, thetaInteger, 0):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported") 
    ratio = rInteger/thetaInteger
    Lz_sign = sign(ratio)
    p0 = _rtheta_resonance_p(a, e, 0, rInteger, thetaInteger)
    p1 = _rtheta_resonance_p(a, e, Lz_sign, rInteger, thetaInteger)
    if (ratio>0)&((p<p1)|(p0<p)):
        print(f"p must be in [{p1:f},{p0:f}]")
        return nan
    if (ratio<0)&((p<p0)|(p1<p)):
        print(f"p must be in [{p0:f},{p1:f}]")
        return nan
    if p == p0:
        return 0
    elif p == p1:
        return 1
    else:
        x_guess = Lz_sign*(p-p0)/(p1-p0)
        return newton(lambda x: _rtheta_frequencyRatio(a, p, e, x)*abs(thetaInteger)-rInteger, x_guess)


## r-phi resonance
def _rphi_resonance_p(a, e, x, rInteger, phiInteger):
    """Find p of r-phi resonant orbit with given (a,e,x,r-integer,phi-integer) such that
    r-frequency/phi-frequency = r-integer/phi-integer
    
    Parameters
    ----------
    rInteger: integer
        r-integer
    phiInteger: integer
        phi-integer
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    p : double
        orbital semi-latus rectum
    """
    
    a, x = _standardize_params(a, x)
    if not valid_integers(rInteger, 0, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    ratio = rInteger/phiInteger
    if x == 0:
        if ratio > 0:
            resonantEquation = lambda p: _rphi_frequencyRatio(a, p, e, 0)[0]*abs(phiInteger)-rInteger
        elif ratio < 0:
            resonantEquation = lambda p: _rphi_frequencyRatio(a, p, e, 0)[1]*abs(phiInteger)-rInteger
    else:
        if sign(x)*sign(ratio)<0:
            raise ValueError("Require sign(x) = sign(rInteger/phiInteger)")
        resonantEquation = lambda p: _rphi_frequencyRatio(a, p, e, x)*abs(phiInteger)-rInteger
    ps = separatrix(a,e,x)
    pWF = 6/(1-ratio**2)
    pNS = _rphi_nearSeparatrix_initialGuess(a,e,x,rInteger/phiInteger,ps)
    return _doubleResonance_solver(resonantEquation,pWF,pNS,ps)

def _rphi_resonance_e(a, p, x, rInteger, phiInteger):
    """Find e of r-phi resonant orbit with given (a,p,x,r-integer,phi-integer) such that
    r-frequency/phi-frequency = r-integer/phi-integer
    
    Parameters
    ----------
    rInteger: integer
        r-integer
    phiInteger: integer
        phi-integer
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    x : double
        cosine of the orbital inclination

    Returns
    -------
    e : double
        orbital eccentricity
    """
    
    a, x = _standardize_params(a, x)
    if not valid_integers(rInteger, 0, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    ratio = rInteger/phiInteger
    p0 = _rphi_resonance_p(a, 0, x, rInteger, phiInteger)
    p1 = _rphi_resonance_p(a, 1, x, rInteger, phiInteger)
    if (p<p0)|(p1<p):
        print(f"p must be in [{p0:f},{p1:f}]")
        return nan
    if p == p0:
        return 0
    if p == p1:
        return 1
    e_guess = (p-p0)/(p1-p0)
    if x == 0:
        if ratio > 0:
            resonantEquation = lambda e: _rphi_frequencyRatio(a, p, e, x)[0]*abs(phiInteger)-rInteger
        elif ratio < 0:
            resonantEquation = lambda e: _rphi_frequencyRatio(a, p, e, x)[1]*abs(phiInteger)-rInteger
    else:
        if sign(x)*sign(ratio)<0:
            raise ValueError("Require sign(x) = sign(rInteger/phiInteger)")
        resonantEquation = lambda e: _rphi_frequencyRatio(a, p, e, x)*abs(phiInteger)-rInteger
    return newton(resonantEquation, e_guess)

def _rphi_resonance_x(a, p, e, rInteger, phiInteger):
    """Find x of r-phi resonant orbit with given (a,p,e,r-integer,phi-integer) such that
    r-frequency/phi-frequency = r-integer/phi-integer
    
    Parameters
    ----------
    rInteger: integer
        r-integer
    phiInteger: integer
        phi-integer
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    
    Returns
    -------
    x : double
        cosine of the orbital inclination
    """
    
    a = abs(a)
    if not valid_integers(rInteger, 0, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    ratio = rInteger/phiInteger
    Lz_sign = sign(ratio)
    p0 = _rphi_resonance_p(a, e, 0, rInteger, phiInteger)
    p1 = _rphi_resonance_p(a, e, Lz_sign, rInteger, phiInteger)
    if (ratio>0)&((p<p1)|(p0<p)):
        print(f"p must be in [{p1:f},{p0:f}]")
        return nan
    if (ratio<0)&((p<p0)|(p1<p)):
        print(f"p must be in [{p0:f},{p1:f}]")
        return nan
    if p == p0:
        return 0
    if p == p1:
        return 1
    x_guess = Lz_sign*(p-p0)/(p1-p0)
    resonantEquation = lambda x: _rphi_frequencyRatio(a, p, e, x)*abs(phiInteger)-rInteger
    return newton(resonantEquation, x_guess)


## phi-theta resonance
def _phitheta_resonance_p(a, e, x, thetaInteger, phiInteger):
    """Find p of phi-theta resonant orbit with given (a,e,x,theta-integer,phi-integer) such that
    phi-frequency/theta-frequency = phi-integer/theta-integer
    
    Parameters
    ----------
    thetaInteger: integer
        theta-integer
    phiInteger: integer
        phi-integer
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    p : double
        orbital semi-latus rectum
    """
    
    a, x = _standardize_params(a, x)
    if not valid_integers(0, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if (x!=0) & (sign(x)!=sign(thetaInteger)):
        raise ValueError("Require sign(x) = sign(phiInteger/thetaInteger-1)")
    ps = separatrix(a,e,x)
    bound = _phitheta_boundRatio(a, e, x, ps=ps, xSign=sign(thetaInteger))
    if phiInteger == bound*thetaInteger:
        return ps
    elif not (thetaInteger<phiInteger) & (phiInteger<bound*thetaInteger):
        print(f"No prograde (retrograde) resonace whose phi-theta ratio greater (lesser) than {bound}")
        return nan
    if x == 0:
        if phiInteger/thetaInteger > 1:
            resonantEquation = lambda p: _phitheta_frequencyRatio(a, p, e, 0)[0]*thetaInteger-phiInteger
        elif phiInteger/thetaInteger < 1:
            resonantEquation = lambda p: _phitheta_frequencyRatio(a, p, e, 0)[1]*thetaInteger-phiInteger
    else:
        resonantEquation = lambda p: _phitheta_frequencyRatio(a, p, e, x)*thetaInteger-phiInteger
    
    pWF = (2*a/abs(phiInteger/thetaInteger-1))**(2/3)
    pNS = _phitheta_nearSeparatrix_initialGuess(a,e,x,phiInteger/thetaInteger,ps)
    return _doubleResonance_solver(resonantEquation,pWF,pNS,ps)

def _phitheta_resonance_e(a, p, x, thetaInteger, phiInteger):
    """Find e of phi-theta resonant orbit with given (a,p,x,theta-integer,phi-integer) such that
    phi-frequency/theta-frequency = phi-integer/theta-integer
    
    Parameters
    ----------
    thetaInteger: integer
        theta-integer
    phiInteger: integer
        phi-integer
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    x : double
        cosine of the orbital inclination

    Returns
    -------
    e : double
        orbital eccentricity
    """
    
    a, x = _standardize_params(a, x)
    if not valid_integers(0, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    ratio = phiInteger/thetaInteger
    if (x!=0) & (x*(ratio-1)<0):
        raise ValueError("Require sign(x) = sign(phiInteger/thetaInteger-1)")
    ratio_bound_0 = _phitheta_boundRatio(a, 0, x, xSign=sign(ratio-1))
    ratio_bound_1 = _phitheta_boundRatio(a, 1, x, xSign=sign(ratio-1))
    
    if (thetaInteger<ratio_bound_1*thetaInteger) & (ratio_bound_1*thetaInteger<phiInteger):
        print(f"No prograde (retrograde) resonace whose phi-theta ratio greater (lesser) than {ratio_bound_1}")
        return nan
    elif (thetaInteger<phiInteger) & (phiInteger<ratio_bound_0*thetaInteger):
        e0 = 0
        p0 = _phitheta_resonance_p(a, e0, x, thetaInteger, phiInteger)
    elif (ratio_bound_0*thetaInteger<phiInteger) & (phiInteger<ratio_bound_1*thetaInteger):
        e0 = root_scalar(lambda e: _phitheta_boundRatio(a,e,x,xSign=sign(ratio-1))-ratio, bracket=[0,1], method="brentq").root
        p0 = separatrix(a,e0,x)
    p1 = _phitheta_resonance_p(a, 1, x, thetaInteger, phiInteger)
    if (p<p0)|(p1<p):
        print(f"p must be in [{p0},{p1}]")
        return nan
    elif p == p0:
        return e0
    elif p == p1:
        return 1
    else:
        e_guess = (1-e0)*(p-p0)/(p1-p0)+e0
    if x == 0:
        if ratio > 1:
            resonantEquation = lambda e: _phitheta_frequencyRatio(a, p, e, x)[0]*thetaInteger-phiInteger
        elif ratio < 1:
            resonantEquation = lambda e: _phitheta_frequencyRatio(a, p, e, x)[1]*thetaInteger-phiInteger
    else:
        resonantEquation = lambda e: _phitheta_frequencyRatio(a, p, e, x)*thetaInteger-phiInteger
    try:
        return newton(resonantEquation, e_guess)
    except (ValueError):
        print(f"p is very close to separatrix, it is between {p0} and {p1}")
        return nan
    
def _phitheta_resonance_x(a, p, e, thetaInteger, phiInteger):
    """Find x of phi-theta resonant orbit with given (a,p,e,theta-integer,phi-integer) such that
    phi-frequency/theta-frequency = phi-integer/theta-integer
    
    Parameters
    ----------
    thetaInteger: integer
        theta-integer
    phiInteger: integer
        phi-integer
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity

    Returns
    -------
    x : double
        cosine of the orbital inclination
    
    """
    
    a = abs(a)
    
    if not valid_integers(0, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    ratio = phiInteger/thetaInteger
    ratio_bound_0 = _phitheta_boundRatio(a, e, 0, xSign=sign(ratio-1))
    ratio_bound_1 = _phitheta_boundRatio(a, e, sign(ratio-1), xSign=sign(ratio-1))
    if ratio > 1:
        if phiInteger > thetaInteger*ratio_bound_1:
            raise ValueError("No resonance")
        elif thetaInteger*ratio_bound_0 < phiInteger < thetaInteger*ratio_bound_1:
            x0 = root_scalar(lambda x: _phitheta_boundRatio(a, e, x, xSign=1)*thetaInteger-phiInteger, bracket=[0,1], method="brentq").root
            p0 = separatrix(a, e, x0)
            x1 = 1
            p1 = _phitheta_resonance_p(a, e, x1, thetaInteger, phiInteger)
            x_guess = (1-x0)*(p-p0)/(p1-p0)+x0
        elif phiInteger < thetaInteger*ratio_bound_0:
            x0 = 0
            p0 = _phitheta_resonance_p(a, e, x0, thetaInteger, phiInteger)
            x1 = 1
            p1 = _phitheta_resonance_p(a, e, x1, thetaInteger, phiInteger)
            x_guess = (p-p0)/(p1-p0)
        if (p<p1)|(p0<p):
            print(f"p must be in [{p1},{p0}]")
            return nan   
    elif ratio < 1:
        if phiInteger > thetaInteger*ratio_bound_0:
            raise ValueError("No resonance")
        elif thetaInteger*ratio_bound_1 < phiInteger < thetaInteger*ratio_bound_0:
            x1 = root_scalar(lambda x: _phitheta_boundRatio(a, e, x, xSign=-1)*thetaInteger-phiInteger, bracket=[-1,0], method="brentq").root
            p1 = separatrix(a,e,x1)
            x0 = 0
            p0 = _phitheta_resonance_p(a, e, x0, thetaInteger, phiInteger)
            x_guess = -x1*(p-p1)/(p0-p1)+x1
        elif phiInteger < thetaInteger*ratio_bound_1:
            x1 = -1
            p1 = _phitheta_resonance_p(a, e, x1, thetaInteger, phiInteger)
            x0 = 0
            p0 = _phitheta_resonance_p(a, e, x0, thetaInteger, phiInteger)
            x_guess = -(p-p0)/(p1-p0)
        if (p<p0)|(p1<p): 
            print(f"p must be in [{p0},{p1}]")
            return nan
    if (p==p1): return x1
    if (p==p0): return x0
    try:
        return newton(lambda x: _phitheta_frequencyRatio(a,p,e,x)*thetaInteger-phiInteger, x_guess)
    except (ValueError):
        print(f"p is very close to separatrix, x is between {x0} and {x1}")
        return nan
    

# Finding triple resonance

## Given (a,e)
def _tripleResonance_px_rInteger(thetaInteger, phiInteger, a, e):
    """Find possible r-integer sastify
    
    r-frequency:theta-frequency:phi-frequency=r-integer:theta-integer:phi-integer.

    Parameters
    ----------
    thetaInteger : double
        theta-integer
    phiInteger : double
        phi-integer
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    
    Returns
    ------- 
    (double, double)
        The range of posible r-integers.
    """
        
    a = abs(a)
    
    if not valid_integers(0, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    phitheta_ratio = phiInteger/thetaInteger
    xSign = sign(phiInteger/thetaInteger-1)
    ratio_bound_0 = _phitheta_boundRatio(a,e,0,xSign=xSign)
    ratio_bound_1 = _phitheta_boundRatio(a,e,xSign)
        
    if phitheta_ratio > 1:
        if phiInteger > thetaInteger*ratio_bound_1:
            raise ValueError("No resonance")
        elif thetaInteger*ratio_bound_0 < phiInteger < thetaInteger*ratio_bound_1:
            p1 = _phitheta_resonance_p(a, e, xSign, thetaInteger, phiInteger)
            rtheta_ratio1 = _rtheta_frequencyRatio(a,p1,e,xSign)
            return [0, rtheta_ratio1*abs(thetaInteger)]
            
        elif phiInteger < thetaInteger*ratio_bound_0:
            p0 = _phitheta_resonance_p(a, e, 0, thetaInteger, phiInteger)
            p1 = _phitheta_resonance_p(a, e, xSign, thetaInteger, phiInteger)
            rtheta_ratio0 = _rtheta_frequencyRatio(a, p0, e, 0)
            rtheta_ratio1 = _rtheta_frequencyRatio(a, p1, e, xSign)
            return [abs(thetaInteger)*rtheta_ratio0, abs(thetaInteger)*rtheta_ratio1]

    elif phitheta_ratio < 1:
        if phiInteger > thetaInteger*ratio_bound_0:
            raise ValueError("No resonance")
        elif thetaInteger*ratio_bound_1 < phiInteger < thetaInteger*ratio_bound_0:
            p0 = _phitheta_resonance_p(a, e, 0, thetaInteger, phiInteger)
            rtheta_ratio0 = _rtheta_frequencyRatio(a, p0, e, 0)
            return [0, rtheta_ratio0*abs(thetaInteger)]
        elif phiInteger < thetaInteger*ratio_bound_1:
            p0 = _phitheta_resonance_p(a, e, 0, thetaInteger, phiInteger)
            p1 = _phitheta_resonance_p(a, e, xSign, thetaInteger, phiInteger)
            rtheta_ratio0 = _rtheta_frequencyRatio(a, p0, e, 0)
            rtheta_ratio1 = _rtheta_frequencyRatio(a, p1, e, xSign)
            return sorted([abs(thetaInteger)*rtheta_ratio1, abs(thetaInteger)*rtheta_ratio0])

def _tripleResonance_px_thetaInteger(rInteger, phiInteger, a, e):
    """Find possible theta-integer sastify
    
    r-frequency:theta-frequency:phi-frequency=r-integer:theta-integer:phi-integer.

    Parameters
    ----------
    rInteger : double
        r-integer
    phiInteger : double
        theta-integer
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    
    Returns
    ------- 
    (double, double)
        The range of posible theta-integers.
    """
        
    a = abs(a)
    
    if not valid_integers(rInteger, 0, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    p0 = _rphi_resonance_p(a, e, 0, rInteger, phiInteger)
    p1 = _rphi_resonance_p(a, e, sign(phiInteger), rInteger, phiInteger)
    rtheta_ratio0 = _rtheta_frequencyRatio(a, p0, e, 0)
    rtheta_ratio1 = _rtheta_frequencyRatio(a, p1, e, sign(phiInteger))
    
    return sorted([rInteger/rtheta_ratio0, rInteger/rtheta_ratio1])

def _tripleResonance_px_phiInteger(rInteger, thetaInteger, a, e):
    """Find possible phi-integer sastify
    
    r-frequency:theta-frequency:phi-frequency=r-integer:theta-integer:phi-integer.

    Parameters
    ----------
    rInteger : double
        r-integer
    thetaInteger : double
        theta-integer
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    
    Returns
    ------- 
    (double, double)
        The range of posible phi-integers.
    """
        
    a = abs(a)
    if not valid_integers(rInteger, thetaInteger, 0):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    p0 = _rtheta_resonance_p(a, e, 0, rInteger, thetaInteger)
    p1 = _rtheta_resonance_p(a, e, sign(thetaInteger), rInteger, thetaInteger)
    if thetaInteger > 0:
        rphi_ratio0 = _rphi_frequencyRatio(a, p0, e, 0)[0]
    elif thetaInteger < 0:
        rphi_ratio0 = _rphi_frequencyRatio(a, p0, e, 0)[1]
    rphi_ratio1 = _rphi_frequencyRatio(a, p1, e, sign(thetaInteger))
    
    return sorted([rInteger/rphi_ratio0, rInteger/rphi_ratio1])

def _tripleResonance_px(a, e, rInteger, thetaInteger, phiInteger):
    """Find (p, x) from resonant modes

    Parameters
    ----------
    rMode : double
        radial mode
    thetaMode : double
        polar-angle mode
    phiMode : double
        azimuthal-angle mode
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    
    Returns
    ------- 
    (p, x) : (double, double)
        Return orbital semi-latus rectum and cosine of the orbital inclination, respectively
    """
    
    a = abs(a)
    xSign = sign(thetaInteger)
    
    if not valid_integers(rInteger, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")  

    p0_rtheta = _rtheta_resonance_p(a, e, 0, rInteger, thetaInteger)
    p1_rtheta = _rtheta_resonance_p(a, e, xSign, rInteger, thetaInteger)
    p0_rphi = _rphi_resonance_p(a, e, 0, rInteger, phiInteger)
    p1_rphi = _rphi_resonance_p(a, e, xSign, rInteger, phiInteger)
                                   
    check_if_having_solution = (p0_rtheta-p0_rphi)*(p1_rtheta-p1_rphi)
    
    if (check_if_having_solution > 0) | (check_if_having_solution == nan):
        rs = _tripleResonance_px_rInteger(thetaInteger, phiInteger, a, e)
        thetas = _tripleResonance_px_thetaInteger(rInteger, phiInteger, a, e)
        phis = _tripleResonance_px_phiInteger(rInteger, thetaInteger, a, e)
        print("No triple resonance, try changing one of the integers:")
        print(f"Choose r-integer in {rs}")
        print(f"Choose theta-integer in {thetas}]")
        print(f"Choose phi-integer in {phis}")
        return [nan, nan]
    elif check_if_having_solution < 0:
        x0 = xSign/((p1_rphi-p1_rtheta)/(p0_rtheta-p0_rphi)+1)
        p0 = p0_rtheta-xSign*x0*(p0_rtheta-p1_rtheta)
        sol = fsolve(lambda X: [_rtheta_frequencyRatio(a, X[0], e, X[1])*abs(thetaInteger)-rInteger,
                                _rphi_frequencyRatio(a, X[0], e, X[1])*abs(phiInteger)-rInteger], [p0, x0])
        return sol
    else:
        if (p0_rtheta-p0_rphi) == 0:
            return [p0_rtheta, 0]
        elif (p1_rtheta-p1_rphi) == 0:
            return [p1_rtheta, xSign]

## Given (a,x)
def _tripleResonance_pe_rInteger(thetaInteger, phiInteger, a, x):
    """Find possible r-integer sastify
    
    r-frequency:theta-frequency:phi-frequency=r-integer:theta-integer:phi-integer.

    Parameters
    ----------
    thetaInteger : double
        theta-integer
    phiInteger : double
        phi-integer
    a : double
        dimensionless spin of the black hole
    x : double
        cosine of the orbital inclination
    
    Returns
    ------- 
    (double, double)
        The range of posible r-integers.
    """
        
    a = abs(a)
    
    if not valid_integers(0, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a, .5, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    phitheta_ratio = phiInteger/thetaInteger
    xSign = sign(phiInteger/thetaInteger-1)
    ratio_bound_0 = _phitheta_boundRatio(a,0,x,xSign=xSign)
    ratio_bound_1 = _phitheta_boundRatio(a,1,x,xSign=xSign)
    
    if (thetaInteger<ratio_bound_1*thetaInteger) & (ratio_bound_1*thetaInteger<phiInteger):
        print(f"No prograde (retrograde) resonace whose phi-theta ratio greater (lesser) than {ratio_bound_1}")
        return nan
    elif (thetaInteger<phiInteger) & (phiInteger<ratio_bound_0*thetaInteger):
        p0 = _phitheta_resonance_p(a, 0, x, thetaInteger, phiInteger)
        p1 = _phitheta_resonance_p(a, 1, x, thetaInteger, phiInteger)
        rtheta_ratio0 = _rtheta_frequencyRatio(a, p0, 0, x)
        rtheta_ratio1 = _rtheta_frequencyRatio(a, p1, 1, x)
        return [abs(thetaInteger)*rtheta_ratio0, abs(thetaInteger)*rtheta_ratio1]
        
    elif (ratio_bound_0*thetaInteger<phiInteger) & (phiInteger<ratio_bound_1*thetaInteger):
        p1 = _phitheta_resonance_p(a, 1, x, thetaInteger, phiInteger)
        rtheta_ratio1 = _rtheta_frequencyRatio(a, p1, 1, x)
        return [0, abs(thetaInteger)*rtheta_ratio1]

def _tripleResonance_pe_thetaInteger(rInteger, phiInteger, a, x):
    """Find possible theta-integer sastify
    
    r-frequency:theta-frequency:phi-frequency=r-integer:theta-integer:phi-integer.

    Parameters
    ----------
    rInteger : double
        r-integer
    phiInteger : double
        theta-integer
    a : double
        dimensionless spin of the black hole
    x : double
        cosine of orbital inclination
    
    Returns
    ------- 
    (double, double)
        The range of posible theta-integers.
    """
        
    a = abs(a)
    
    if not valid_integers(rInteger, 0, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if phiInteger*x<0:
        raise ValueError("Require phiInteger*x>=0")
    if not valid_params(a, 0.5, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    p0 = _rphi_resonance_p(a, 0, x, rInteger, phiInteger)
    p1 = _rphi_resonance_p(a, 1, x, rInteger, phiInteger)
    rtheta_ratio0 = sign(phiInteger)*_rtheta_frequencyRatio(a, p0, 0, x)
    rtheta_ratio1 = sign(phiInteger)*_rtheta_frequencyRatio(a, p1, 1, x)
    
    return sorted([rInteger/rtheta_ratio0, rInteger/rtheta_ratio1])

def _tripleResonance_pe_phiInteger(rInteger, thetaInteger, a, x):
    """Find possible phi-integer sastify
    
    r-frequency:theta-frequency:phi-frequency=r-integer:theta-integer:phi-integer.

    Parameters
    ----------
    rInteger : double
        r-integer
    thetaInteger : double
        theta-integer
    a : double
        dimensionless spin of the black hole
    x : double
        cosine of orbital inclination
    
    Returns
    ------- 
    (double, double)
        The range of posible theta-integers.
    """
        
    a = abs(a)
    
    if not valid_integers(rInteger, thetaInteger, 0):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if thetaInteger*x<0:
        raise ValueError("Require thetaInteger*x>=0")
    if not valid_params(a, 0.5, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    p0 = _rtheta_resonance_p(a, 0, x, rInteger, thetaInteger)
    p1 = _rtheta_resonance_p(a, 1, x, rInteger, thetaInteger)
    rphi_ratio0 = sign(thetaInteger)*_rphi_frequencyRatio(a, p0, 0, x)
    rphi_ratio1 = sign(thetaInteger)*_rphi_frequencyRatio(a, p1, 1, x)
    
    return sorted([rInteger/rphi_ratio0, rInteger/rphi_ratio1])

def _tripleResonance_pe(a, x, rInteger, thetaInteger, phiInteger):
    """Find (p, x) from resonant modes

    Parameters
    ----------
    rMode : double
        radial mode
    thetaMode : double
        polar-angle mode
    phiMode : double
        azimuthal-angle mode
    a : double
        dimensionless spin of the black hole
    x : double
        cosine of orbital inclination
    
    Returns
    ------- 
    (p, e) : (double, double)
        Return orbital semi-latus rectum and orbital eccentricity, respectively
    """
    
    a = abs(a)
    xSign = sign(thetaInteger)
    
    if not valid_integers(rInteger, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger|, thetaInteger*phiInteger>0 and thetaInteger<phiInteger")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a, 0.5, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")  

    p0_rtheta = _rtheta_resonance_p(a, 0, x, rInteger, thetaInteger)
    p1_rtheta = _rtheta_resonance_p(a, 1, x, rInteger, thetaInteger)
    p0_rphi = _rphi_resonance_p(a, 0, x, rInteger, phiInteger)
    p1_rphi = _rphi_resonance_p(a, 1, x, rInteger, phiInteger)
                                   
    check_if_having_solution = (p0_rtheta-p0_rphi)*(p1_rtheta-p1_rphi)
    
    if (check_if_having_solution > 0) | (check_if_having_solution == nan):
        rs = _tripleResonance_pe_rInteger(thetaInteger, phiInteger, a, x)
        thetas = _tripleResonance_pe_thetaInteger(rInteger, phiInteger, a, x)
        phis = _tripleResonance_pe_phiInteger(rInteger, thetaInteger, a, x)
        print("No triple resonance, try changing one of the integers:")
        print(f"Choose r-integer in {rs}")
        print(f"Choose theta-integer in {thetas}")
        print(f"Choose phi-integer in {phis}")
        return [nan, nan]
    elif check_if_having_solution < 0:
        e0 = 1/((p1_rphi-p1_rtheta)/(p0_rtheta-p0_rphi)+1)
        p0 = p0_rtheta-e0*(p0_rtheta-p1_rtheta)
        sol = fsolve(lambda X: [_rtheta_frequencyRatio(a, X[0], X[1], x)*abs(thetaInteger)-rInteger,
                                _rphi_frequencyRatio(a, X[0], X[1], x)*abs(phiInteger)-rInteger], [p0, e0])
        return sol
    else:
        if (p0_rtheta-p0_rphi) == 0:
            return [p0_rtheta, 0]
        elif (p1_rtheta-p1_rphi) == 0:
            return [p1_rtheta, 1]


# Generic function
def findResonance(a, p=None, e=None, x=None, integers=[0,0,0]):
    rInteger, thetaInteger, phiInteger = integers
    if (rInteger!=0) & (thetaInteger!=0) & (phiInteger!=0):
        if (p==None) & (e!=None) & (x==None):
            return _tripleResonance_px(a, e, rInteger, thetaInteger, phiInteger)
        elif (p==None) & (e==None) & (x!=None):
            return _tripleResonance_pe(a, x, rInteger, thetaInteger, phiInteger)
        else: 
            raise ValueError("Only support finding triple resonances with given (a,x) or (a,e)")
        
    elif (rInteger==0) & (thetaInteger!=0) & (phiInteger!=0):
        if (p==None) & (e!=None) & (x!=None):
            return _phitheta_resonance_p(a, e, x, thetaInteger, phiInteger)
        elif (p!=None) & (e==None) & (x!=None):
            return _phitheta_resonance_e(a, p, x, thetaInteger, phiInteger)
        elif (p!=None) & (e!=None) & (x==None):
            return _phitheta_resonance_x(a, p, e, thetaInteger, phiInteger)
        else: 
            raise ValueError("Require 2 of (p, e, x)")
        
    elif (rInteger!=0) & (thetaInteger==0) & (phiInteger!=0):
        if (p==None) & (e!=None) & (x!=None):
            return _rphi_resonance_p(a, e, x, rInteger, phiInteger)
        elif (p!=None) & (e==None) & (x!=None):
            return _rphi_resonance_e(a, p, x, rInteger, phiInteger)
        elif (p!=None) & (e!=None) & (x==None):
            return _rphi_resonance_x(a, p, e, rInteger, phiInteger)
        else: 
            raise ValueError("Require 2 of (p, e, x)")
        
    elif (rInteger!=0) & (thetaInteger!=0) & (phiInteger==0):
        if (p==None) & (e!=None) & (x!=None):
            return _rtheta_resonance_p(a, e, x, rInteger, thetaInteger)
        elif (p!=None) & (e==None) & (x!=None):
            return _rtheta_resonance_e(a, p, x, rInteger, thetaInteger)
        elif (p!=None) & (e!=None) & (x==None):
            return _rtheta_resonance_x(a, p, e, rInteger, thetaInteger)
        else: 
            raise ValueError("Require 2 of (p, e, x)")
        
    else:
        raise ValueError("At least 2 of the integers are non-zero")        