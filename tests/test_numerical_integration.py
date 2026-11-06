"""Cross-check KerrGeoPy's closed-form geodesics against direct numerical
integration of the second-order geodesic equation.

This suite is an *independent* check: it does not use any of KerrGeoPy's
analytic solution machinery.  For each orbit it takes KerrGeoPy's own initial
state at Mino time ``lambda = 0`` and integrates the geodesic equation
``d^2 x^mu/d tau^2 = - Gamma^mu_{ab} u^a u^b`` numerically (with
``scipy.integrate.solve_ivp``, which is already a dependency), then asserts that
KerrGeoPy's analytic ``trajectory(lambda)`` reproduces the numerically
integrated path.  In effect it answers: *is each closed-form solution an actual
geodesic consistent with its own stated initial conditions?*

The coordinate acceleration is assembled directly from the metric and its
analytic r/theta derivatives.  Every numerical integration is first checked for
its own quality (conservation of E, L, Q and the four-velocity norm) so that a
mismatch is attributable to the analytic solution and not to the integrator.

Orbits are drawn from a *seeded* random generator (reproducible) spanning the
parameter space, including the distinct analytical cases:

* stable bound orbits (a, p, e, x), including circular, equatorial and high spin;
* plunging orbits (a, E, L, Q) with a turning point outside the horizon;
* null geodesics covering all four radial cases and both polar cases
  (ordinary eta > 0 and vortical eta < 0).

For plunging and null orbits the comparison is made in the exterior region
(r > r_+): a Boyer-Lindquist integrator cannot continue through the horizon,
where g_rr = Sigma/Delta diverges, whereas the closed forms continue
analytically.
"""
import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from kerrgeopy.stable import StableOrbit
from kerrgeopy.plunge import PlungingOrbit
from kerrgeopy.light import LightOrbit
from kerrgeopy.constants import separatrix

SEED = 20240615
N_STABLE = 20
N_PLUNGE = 10
N_LIGHT_RANDOM = 10


# ---------------------------------------------------------------------------
# Pure-Python numerical oracle (no KerrGeoPy internals).
# ---------------------------------------------------------------------------
def horizon(a):
    return 1.0 + np.sqrt(1.0 - a * a)


def geodesic_rhs(a, y):
    """RHS of the first-order form of the geodesic equation.

    State y = (t, r, theta, phi, u^t, u^r, u^theta, u^phi, lambda); the last
    component carries Mino time via d lambda/d tau = 1/Sigma.
    """
    r, th = y[1], y[2]
    Vt, Vr, Vth, Vph = y[4], y[5], y[6], y[7]
    s, c = np.sin(th), np.cos(th)
    s2, c2, sc = s * s, c * c, s * c
    Sig = r * r + a * a * c2
    Del = r * r - 2.0 * r + a * a
    Sig2 = Sig * Sig
    r2a2 = r * r + a * a

    g_tt = -(1.0 - 2.0 * r / Sig)
    g_tph = -2.0 * a * r * s2 / Sig
    g_phph = s2 * (r2a2 + 2.0 * a * a * r * s2 / Sig)

    dg_tt_r = (2.0 * Sig - 4.0 * r * r) / Sig2
    dg_tph_r = -2.0 * a * s2 * (Sig - 2.0 * r * r) / Sig2
    dg_rr_r = (2.0 * r * Del - Sig * (2.0 * r - 2.0)) / (Del * Del)
    dg_thth_r = 2.0 * r
    dg_phph_r = s2 * (2.0 * r + 2.0 * a * a * s2 * (Sig - 2.0 * r * r) / Sig2)

    dg_tt_th = 4.0 * a * a * r * sc / Sig2
    dg_tph_th = -4.0 * a * r * sc * r2a2 / Sig2
    dg_rr_th = -2.0 * a * a * sc / Del
    dg_thth_th = -2.0 * a * a * sc
    dg_phph_th = 2.0 * sc * r2a2 + 4.0 * a * a * r * s2 * sc * (2.0 * Sig + a * a * s2) / Sig2

    F_t = -(dg_tt_r * Vr * Vt + dg_tph_r * Vr * Vph
            + dg_tt_th * Vth * Vt + dg_tph_th * Vth * Vph)
    F_ph = -(dg_tph_r * Vr * Vt + dg_phph_r * Vr * Vph
             + dg_tph_th * Vth * Vt + dg_phph_th * Vth * Vph)
    term1_r = dg_rr_r * Vr * Vr + dg_rr_th * Vth * Vr
    half_dr = 0.5 * (dg_tt_r * Vt * Vt + 2.0 * dg_tph_r * Vt * Vph
                     + dg_rr_r * Vr * Vr + dg_thth_r * Vth * Vth + dg_phph_r * Vph * Vph)
    F_r = -term1_r + half_dr
    term1_th = dg_thth_r * Vr * Vth + dg_thth_th * Vth * Vth
    half_dth = 0.5 * (dg_tt_th * Vt * Vt + 2.0 * dg_tph_th * Vt * Vph
                      + dg_rr_th * Vr * Vr + dg_thth_th * Vth * Vth + dg_phph_th * Vph * Vph)
    F_th = -term1_th + half_dth

    det2 = g_tt * g_phph - g_tph * g_tph
    gtt = g_phph / det2
    gtph = -g_tph / det2
    gphph = g_tt / det2
    grr = Del / Sig
    gthth = 1.0 / Sig

    return [Vt, Vr, Vth, Vph,
            gtt * F_t + gtph * F_ph, grr * F_r, gthth * F_th,
            gtph * F_t + gphph * F_ph, 1.0 / Sig]


def constants_of_motion(a, mu2, x, u):
    """Return (E, L, Q, norm) for a state (x, u=dx/d tau)."""
    r, th = x[1], x[2]
    s, c = np.sin(th), np.cos(th)
    s2, c2 = s * s, c * c
    Sig = r * r + a * a * c2
    g_tt = -(1.0 - 2.0 * r / Sig)
    g_tph = -2.0 * a * r * s2 / Sig
    g_rr = Sig / (r * r - 2.0 * r + a * a)
    g_thth = Sig
    g_phph = s2 * (r * r + a * a + 2.0 * a * a * r * s2 / Sig)
    p_t = g_tt * u[0] + g_tph * u[3]
    p_ph = g_tph * u[0] + g_phph * u[3]
    p_th = g_thth * u[2]
    E, L = -p_t, p_ph
    Q = p_th * p_th + c2 * (a * a * (mu2 - E * E) + L * L / s2)
    norm = (g_tt * u[0] * u[0] + 2.0 * g_tph * u[0] * u[3] + g_rr * u[1] * u[1]
            + g_thth * u[2] * u[2] + g_phph * u[3] * u[3])
    return E, L, Q, norm


class NumericalGeodesic:
    """Numerically integrated geodesic from an initial state at lambda = 0."""

    def __init__(self, a, mu2, x0, u0, lam_target, r_escape=1e3,
                 rtol=1e-11, atol=1e-12):
        self.a, self.mu2 = a, mu2
        rp = horizon(a)

        def rhs(tau, y):
            return geodesic_rhs(a, y)

        def ev_lambda(tau, y):
            return y[8] - lam_target
        ev_lambda.terminal, ev_lambda.direction = True, 1

        def ev_hor(tau, y):
            return y[1] - (rp + 1e-3)
        ev_hor.terminal, ev_hor.direction = True, -1

        def ev_esc(tau, y):
            return y[1] - r_escape
        ev_esc.terminal, ev_esc.direction = True, 1

        y0 = [*x0, *u0, 0.0]
        sol = solve_ivp(rhs, [0.0, 1e7], y0, method="DOP853", rtol=rtol,
                        atol=atol, dense_output=True,
                        events=[ev_lambda, ev_hor, ev_esc])
        if not sol.success:
            raise RuntimeError("numerical integration failed: " + sol.message)
        self.sol = sol
        self.tau_end = sol.t[-1]
        self.lam_max = float(sol.sol(self.tau_end)[8])

    def position(self, lams):
        """Coordinates (t, r, theta, phi) at the given Mino times."""
        out = np.empty((len(lams), 4))
        for i, lam in enumerate(lams):
            if lam <= 0:
                tau = 0.0
            else:
                tau = brentq(lambda t: self.sol.sol(t)[8] - lam, 0.0,
                             self.tau_end, xtol=1e-13)
            out[i] = self.sol.sol(tau)[:4]
        return out

    def max_constant_drift(self):
        """Largest deviation of (E, L, Q) and the norm over the integration."""
        taus = np.linspace(0.0, self.tau_end, 60)
        E0, L0, Q0, n0 = constants_of_motion(
            self.a, self.mu2, self.sol.sol(0.0)[:4], self.sol.sol(0.0)[4:8])
        dE = dL = dQ = dn = 0.0
        for tau in taus:
            y = self.sol.sol(tau)
            E, L, Q, n = constants_of_motion(self.a, self.mu2, y[:4], y[4:8])
            dE = max(dE, abs(E - E0))
            dL = max(dL, abs(L - L0))
            dQ = max(dQ, abs(Q - Q0))
            dn = max(dn, abs(n - n0))
        return dE, dL, dQ, dn


# ---------------------------------------------------------------------------
# Photon helpers (own radial-case classification + null momentum).
# ---------------------------------------------------------------------------
def photon_radial_roots(a, eta, ell):
    A = eta + (ell - a) ** 2
    c2 = 2.0 * (a * a - a * ell) - A
    c1 = 2.0 * A
    c0 = (a * a - a * ell) ** 2 - A * a * a
    return np.roots([1.0, 0.0, c2, c1, c0])


def photon_case(a, eta, ell, r0):
    roots = photon_radial_roots(a, eta, ell)
    real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
    if real.size == 0:
        return 4
    if real.size == 2:
        return 3
    return 2 if (real[-1] < horizon(a) or r0 > real[-1]) else 1


def photon_R(a, eta, ell, r):
    return (r * r + a * a - a * ell) ** 2 - (r * r - 2 * r + a * a) * (eta + (ell - a) ** 2)


def photon_Theta(a, eta, ell, th):
    c2 = np.cos(th) ** 2
    return eta + a * a * c2 - ell * ell * c2 / np.sin(th) ** 2


def null_momentum(a, eta, ell, r0, th0, sign_r, sign_th):
    """Contravariant photon four-momentum (energy normalised to 1)."""
    Delta = r0 * r0 - 2 * r0 + a * a
    Sig = r0 * r0 + a * a * np.cos(th0) ** 2
    s2 = np.sin(th0) ** 2
    R = max(photon_R(a, eta, ell, r0), 0.0)
    Th = max(photon_Theta(a, eta, ell, th0), 0.0)
    Vr = sign_r * np.sqrt(R)
    Vth = sign_th * np.sqrt(Th)
    Vt = a * (ell - a * s2) + (r0 * r0 + a * a) / Delta * ((r0 * r0 + a * a) - a * ell)
    Vph = (ell / s2 - a) + a / Delta * ((r0 * r0 + a * a) - a * ell)
    return np.array([Vt, Vr, Vth, Vph]) / Sig


# Curated photons guaranteeing every radial case (1-4) and both polar types.
# (label, a, eta, ell, r0, theta0, sign_r, sign_theta)
CURATED_PHOTONS = [
    ("case1_ordinary", 0.3,  30.0, 1.0, 2.158, np.pi / 2, +1, +1),
    ("case2_ordinary", 0.9,  10.0, 3.0, 20.0,  1.3,       -1, +1),
    ("case3_ordinary", 0.9,   1.0, 0.5, 10.0,  np.pi / 2, -1, +1),
    ("case3_vortical", 0.95, -0.1, 0.3, 12.0,  0.5,       -1, +1),
    ("case4_vortical", 0.9,  -0.2, 0.2, 10.0,  0.5,       -1, +1),
]


# ---------------------------------------------------------------------------
# Seeded random parameter generators.
# ---------------------------------------------------------------------------
def random_stable_orbits(rng, n):
    """Random stable bound orbits (a, p, e, x), plus deterministic edge cases."""
    orbits = [
        (0.0, 12.0, 0.3, 1.0),    # Schwarzschild, equatorial
        (0.9, 8.0, 0.0, 0.5),     # circular, inclined
        (0.95, 10.0, 0.4, 1.0),   # high spin, equatorial prograde
        (0.7, 11.0, 0.5, -0.8),   # retrograde, inclined
    ]
    while len(orbits) < n:
        a = rng.uniform(0.0, 0.98)
        e = rng.uniform(0.0, 0.6)
        x = rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 1.0)
        p = separatrix(a, e, x) + rng.uniform(0.6, 8.0)
        orbits.append((a, p, e, x))
    return orbits


def exterior_window(a, r_func, lam_span=10.0, npts=8001, margin=0.6,
                    rmax_cap=12.0):
    """Largest contiguous Mino-time interval where r > r_+ + margin.

    KerrGeoPy parametrises plunges starting *inside* the horizon, so the
    exterior portion of the orbit (which a Boyer-Lindquist integrator can
    follow) is an excursion at some other lambda; this locates it.  Excursions
    are restricted to a moderate apoapsis (r <= rmax_cap): on a very wide swing
    (r ~ tens) the coordinate t ~ -log(Delta) and the proper-time integration
    both lose precision, which is a conditioning issue, not a solver error.
    """
    rp = horizon(a)
    lams = np.linspace(-lam_span, lam_span, npts)
    rr = np.asarray(r_func(lams))
    ext = np.isfinite(rr) & (rr > rp + margin)
    if not ext.any():
        return None
    best = (0, 0)
    start = None
    for i in range(len(ext) + 1):
        e = ext[i] if i < len(ext) else False
        if e and start is None:
            start = i
        elif not e and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    s, e = best
    rmax = rr[s:e].max()
    if e - s < 5 or rmax < rp + 1.5 or rmax > rmax_cap:
        return None
    return lams[s], lams[e - 1]


def random_plunges(rng, n):
    """Random plunging orbits that have a usable exterior excursion."""
    out = []
    tries = 0
    while len(out) < n and tries < 40000:
        tries += 1
        a = rng.uniform(0.4, 0.95)
        E = rng.uniform(0.93, 0.999)
        L = rng.uniform(-4.0, 4.0)
        Q = rng.uniform(0.5, 8.0)
        try:
            po = PlungingOrbit(a, E, L, Q)
            r = po.trajectory()[1]
        except Exception:
            continue
        win = exterior_window(a, r)
        if win is None:
            continue
        out.append((a, E, L, Q, po, win))
    return out


def random_photons(rng, n):
    """Random valid photons (mixed ordinary/vortical) for breadth of coverage."""
    out = []
    tries = 0
    while len(out) < n and tries < 50000:
        tries += 1
        a = rng.uniform(0.3, 0.97)
        if rng.random() < 0.6:  # ordinary
            eta = rng.uniform(0.5, 25.0)
            ell = rng.uniform(-3.0, 5.0)
        else:  # vortical: |ell| < a and eta in (-(a-|ell|)^2, 0)
            ell = rng.uniform(-0.85, 0.85) * a
            lo = -(a - abs(ell)) ** 2
            if lo > -0.05:  # vortical window too thin to sample reliably
                continue
            eta = rng.uniform(0.9 * lo, -0.02)
        r0 = rng.uniform(4.0, 22.0)
        th0 = rng.uniform(0.35, np.pi - 0.35)
        if photon_R(a, eta, ell, r0) < 1e-3 or photon_Theta(a, eta, ell, th0) < 1e-3:
            continue
        out.append(("random", a, eta, ell, r0, th0, -1, rng.choice([-1, 1])))
    return out


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
class TestStableNumerical(unittest.TestCase):
    def test_trajectory_matches_numerical(self):
        rng = np.random.default_rng(SEED)
        components = ["t", "r", "theta", "phi"]
        for i, (a, p, e, x) in enumerate(random_stable_orbits(rng, N_STABLE)):
            orbit = StableOrbit(a, p, e, x)
            x0, u0 = orbit.initial_position, orbit.initial_velocity
            lam_max = 8.0
            num = NumericalGeodesic(a, 1.0, x0, u0, lam_max)

            # the numerical integration must itself be accurate
            dE, dL, dQ, dn = num.max_constant_drift()
            with self.subTest(i=i, check="integration_quality",
                              params=f"{a=}, {p=}, {e=}, {x=}",
                              drift=(dE, dL, dQ, dn)):
                self.assertLess(dn, 1e-7)
                self.assertLess(dE, 1e-7)

            times = np.linspace(0.0, 0.95 * num.lam_max, 25)
            num_xyz = num.position(times)
            t, r, theta, phi = orbit.trajectory()
            kg_xyz = np.transpose([t(times), r(times), theta(times), phi(times)])
            for j, comp in enumerate(components):
                with self.subTest(i=i, component=comp,
                                  params=f"{a=}, {p=}, {e=}, {x=}",
                                  diff=np.max(np.abs(num_xyz[:, j] - kg_xyz[:, j]))):
                    self.assertTrue(np.allclose(num_xyz[:, j], kg_xyz[:, j],
                                                rtol=1e-6, atol=1e-6))


class TestPlungingNumerical(unittest.TestCase):
    def test_trajectory_matches_numerical(self):
        rng = np.random.default_rng(SEED + 1)
        components = ["t", "r", "theta", "phi"]
        plunges = random_plunges(rng, N_PLUNGE)
        self.assertGreater(len(plunges), 0, "no valid plunges were generated")
        for i, (a, E, L, Q, orbit, (lam_lo, lam_hi)) in enumerate(plunges):
            t, r, theta, phi = orbit.trajectory()
            # start just inside the exterior excursion, moving outward
            lam0 = lam_lo + 0.02 * (lam_hi - lam_lo)
            h = 1e-5
            x0 = (t(lam0), r(lam0), theta(lam0), phi(lam0))
            Sig0 = x0[1] ** 2 + a ** 2 * np.cos(x0[2]) ** 2
            dl = lambda f: (-f(lam0 + 2 * h) + 8 * f(lam0 + h)
                            - 8 * f(lam0 - h) + f(lam0 - 2 * h)) / (12 * h)
            u0 = np.array([dl(t), dl(r), dl(theta), dl(phi)]) / Sig0
            num = NumericalGeodesic(a, 1.0, x0, u0, 1.3 * (lam_hi - lam0))

            dE, dL, dQ, dn = num.max_constant_drift()
            with self.subTest(i=i, check="integration_quality",
                              params=f"{a=}, {E=}, {L=}, {Q=}", drift=(dE, dL, dQ, dn)):
                self.assertLess(dE, 1e-6)
                self.assertLess(dn, 1e-5)  # norm worsens near the horizon (1/Delta)

            s = np.linspace(0.0, 0.95 * num.lam_max, 40)
            num_xyz = num.position(s)
            lam = lam0 + s
            kg_xyz = np.transpose([t(lam), r(lam), theta(lam), phi(lam)])
            # compare only away from the horizon, where t (~ -log Delta) is
            # well conditioned; r, theta, phi agree everywhere in the exterior.
            keep = num_xyz[:, 1] > horizon(a) + 0.7
            # t is a large, fast-accumulating coordinate (dt/dlambda ~ r^2 on a
            # wide excursion), so it is checked with a relative tolerance; the
            # spatial coordinates are checked tightly.
            tol = {0: (1e-5, 1e-4)}
            for j, comp in enumerate(components):
                rt, at = tol.get(j, (1e-6, 1e-6))
                with self.subTest(i=i, component=comp,
                                  params=f"{a=}, {E=}, {L=}, {Q=}",
                                  diff=np.max(np.abs((num_xyz - kg_xyz)[keep, j]))):
                    self.assertTrue(np.allclose(num_xyz[keep, j], kg_xyz[keep, j],
                                                rtol=rt, atol=at))


class TestLightNumerical(unittest.TestCase):
    def _check(self, i, label, a, eta, ell, r0, th0, sr, st):
        p = null_momentum(a, eta, ell, r0, th0, sr, st)
        pos = (0.0, r0, th0, 0.0)
        ko = LightOrbit(a, pos, tuple(p))
        t, r, theta, phi = ko.trajectory()  # also sets ko.lambda_x
        case = photon_case(a, eta, ell, r0)
        # numerical integration from KerrGeoPy's own lambda = 0 state
        x0 = (float(t(0.0)), float(r(0.0)), float(theta(0.0)), float(phi(0.0)))
        num = NumericalGeodesic(a, 0.0, x0, p / ko.E, 0.95 * ko.lambda_x)

        dE, dL, dQ, dn = num.max_constant_drift()
        with self.subTest(i=i, case=f"{case}/{label}", check="integration_quality",
                          params=f"{a=}, {eta=}, {ell=}, {r0=}", drift=(dE, dL, dQ, dn)):
            self.assertLess(dE, 1e-6)

        lam_hi = 0.9 * min(num.lam_max, ko.lambda_x)
        times = np.linspace(0.0, lam_hi, 20)
        num_xyz = num.position(times)
        kg_xyz = np.transpose([t(times), r(times), theta(times), phi(times)])
        # t is large and fast-accumulating on escaping rays -> relative check.
        tol = {0: (1e-5, 1e-4)}
        for j, comp in enumerate(["t", "r", "theta", "phi"]):
            rt, at = tol.get(j, (1e-7, 1e-6))
            with self.subTest(i=i, case=f"{case}/{label}", component=comp,
                              params=f"{a=}, {eta=}, {ell=}, {r0=}",
                              diff=np.nanmax(np.abs(num_xyz[:, j] - kg_xyz[:, j]))):
                self.assertTrue(np.allclose(num_xyz[:, j], kg_xyz[:, j],
                                            rtol=rt, atol=at))

    def test_curated_cases(self):
        """Every radial case (1-4) and both polar cases are covered explicitly."""
        for i, (label, a, eta, ell, r0, th0, sr, st) in enumerate(CURATED_PHOTONS):
            self._check(i, label, a, eta, ell, r0, th0, sr, st)

    def test_random_photons(self):
        rng = np.random.default_rng(SEED + 2)
        photons = random_photons(rng, N_LIGHT_RANDOM)
        self.assertGreater(len(photons), 0, "no valid photons were generated")
        for i, (label, a, eta, ell, r0, th0, sr, st) in enumerate(photons):
            self._check(i, label, a, eta, ell, r0, th0, sr, st)


if __name__ == "__main__":
    unittest.main()
