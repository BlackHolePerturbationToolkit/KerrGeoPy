from kerrgeopy.constants import constants_of_motion
from kerrgeopy.frequencies import orbital_frequencies

class BlackHole:
    def __init__(self,a,M):
        self.a = a
        self.M = M

class Orbit:
    def __init__(self,a,p,e,x):
        self.a, self.p, self.e, self.x = a, p, e, x
        self.E, self.L, self.Q = constants_of_motion(a,p,e,x)
        self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma = orbital_frequencies(a,p,e,x)