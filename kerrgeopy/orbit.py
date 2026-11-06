"""Module containing the Orbit class"""
from .spacetime import KerrSpacetime
from .initial_conditions import *
from .units import *
from .constants import (
    scale_constants,
    apex_from_constants,
    stable_polar_roots,
    stable_radial_roots,
)
from .frequencies import mino_frequencies, fundamental_frequencies
from .plot_utils import is_visible, plot, animate
from numpy import cos, pi
import numpy as np


class Orbit:
    r"""Class representing an orbit in Kerr spacetime defined using initial conditions.

    Parameters
    ----------
    a : double
        spin parameter
    initial_position : tuple(double,double,double,double)
        initial position of the orbit :math:`(t_0,r_0,\theta_0,\phi_0)`
    initial_velocity : tuple(double,double,double,double)
        initial four-velocity of the orbit
        :math:`(u^t_0,u^r_0,u^\theta_0,u^\phi_0)`
    M : double, optional
        mass of the primary in solar masses
    mu : double, optional
        mass of the smaller body in solar masses

    Attributes
    ----------
    a
        spin parameter
    initial_position
        initial position of the orbit :math:`(t_0,r_0,\theta_0,\phi_0)`
    initial_velocity
        initial four-velocity of the orbit
        :math:`(u^t_0,u^r_0,u^\theta_0,u^\phi_0)`
    E
        dimensionless energy
    L
        dimensionless angular momentum
    Q
        dimensionless carter constant
    stable
        boolean indicating whether the orbit is stable
    upsilon_r
        dimensionless radial orbital frequency in Mino time
    upsilon_theta
        dimensionless polar orbital frequency in Mino time
    """

    def __init__(self, a, initial_position, initial_velocity, M=None, mu=None):
        self.a, self.initial_position, self.initial_velocity, self.M, self.mu = (
            a,
            initial_position,
            initial_velocity,
            M,
            mu,
        )

        # check if initial four-velocity is valid
        spacetime = KerrSpacetime(a)
        initial_norm = spacetime.norm(*initial_position, initial_velocity)
        if initial_norm >= 0:
            raise ValueError("Initial velocity is not timelike")
        if abs(initial_norm + 1) > 1e-6:
            raise ValueError("Initial velocity is not normalized")

        E, L, Q = constants_from_initial_conditions(
            a, initial_position, initial_velocity
        )
        self.E, self.L, self.Q = E, L, Q

        if is_stable(a, initial_position, initial_velocity):
            self.stable = True
            a, p, e, x = apex_from_constants(a, E, L, Q)
            self.a, self.p, self.e, self.x = a, p, e, x
            (
                self.upsilon_r,
                self.upsilon_theta,
                self.upsilon_phi,
                self.gamma,
            ) = mino_frequencies(a, p, e, x)
            self.omega_r, self.omega_theta, self.omega_phi = fundamental_frequencies(
                a, p, e, x
            )
            self.initial_phases = stable_orbit_initial_phases(
                a, initial_position, initial_velocity
            )
        else:
            if a == 0:
                raise ValueError("Schwarzschild plunges are not currently supported")
            self.stable = False
            self.upsilon_r, self.upsilon_theta = plunging_mino_frequencies(a, E, L, Q)
            self.initial_phases = plunging_orbit_initial_phases(
                a, initial_position, initial_velocity
            )

    def trajectory_deltas(self, initial_phases=None):
        r"""Computes the trajectory deltas :math:`t_r(q_r)`, :math:`t_\theta(q_\theta)`,
        :math:`\phi_r(q_r)` and :math:`\phi_\theta(q_\theta)`

        Parameters
        ----------
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`

        Returns
        -------
        tuple(function, function, function, function)
            tuple of trajectory deltas :math:`(t_r(q_r),
            t_\theta(q_\theta), \phi_r(q_r),\phi_\theta(q_\theta))`
        """
        if initial_phases is None:
            initial_phases = self.initial_phases
        q_t0, q_r0, q_theta0, q_phi0 = initial_phases

        if self.stable:
            from .stable import radial_solutions, polar_solutions

            constants = (self.E, self.L, self.Q)
            radial_roots = stable_radial_roots(
                self.a, self.p, self.e, self.x, constants
            )
            polar_roots = stable_polar_roots(self.a, self.p, self.e, self.x, constants)
            r, t_r, phi_r = radial_solutions(self.a, constants, radial_roots)
            theta, t_theta, phi_theta = polar_solutions(self.a, constants, polar_roots)
        else:
            radial_roots = plunging_radial_roots(self.a, self.E, self.L, self.Q)
            if np.iscomplex(radial_roots[3]):
                from .plunge import (
                    plunging_radial_solutions_complex,
                    plunging_polar_solutions,
                )

                # adjust q_theta0 so that initial conditions are consistent with stable orbits
                q_theta0 = q_theta0 + pi / 2
                r, t_r, phi_r = plunging_radial_solutions_complex(
                    self.a, self.E, self.L, self.Q
                )
                theta, t_theta, phi_theta = plunging_polar_solutions(
                    self.a, self.E, self.L, self.Q
                )
            else:
                from .stable import radial_solutions, polar_solutions

                constants = (self.E, self.L, self.Q)
                r, t_r, phi_r = radial_solutions(self.a, constants, radial_roots)
                theta, t_theta, phi_theta = polar_solutions(
                    self.a, constants, radial_roots
                )

        return (
            lambda q_r: t_r(q_r + q_r0),
            lambda q_theta: t_theta(q_theta + q_theta0),
            lambda q_r: phi_r(q_r + q_r0),
            lambda q_theta: phi_theta(q_theta + q_theta0),
        )

    def trajectory(
        self, initial_phases=None, distance_units="natural", time_units="natural"
    ):
        r"""Computes the components of the trajectory as a function of Mino time

        Parameters
        ----------
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        distance_units : str, optional
            units to compute the radial component of the trajectory in
            (options are "natural", "mks", "cgs", "au" and "km"),
            defaults to "natural"
        time_units : str, optional
            units to compute the time component of the trajectory in
            (options are "natural", "mks", "cgs", and "days"), defaults
            to "natural"

        Returns
        -------
        tuple(function, function, function, function)
            tuple of functions :math:`(t(\lambda), r(\lambda),
            \theta(\lambda), \phi(\lambda))`
        """
        if initial_phases is None:
            initial_phases = self.initial_phases
        if self.stable:
            from .stable import stable_trajectory

            return stable_trajectory(
                self.a,
                self.p,
                self.e,
                self.x,
                initial_phases,
                self.M,
                distance_units,
                time_units,
            )
        else:
            from .plunge import plunging_trajectory

            return plunging_trajectory(
                self.a,
                self.E,
                self.L,
                self.Q,
                initial_phases,
                self.M,
                distance_units,
                time_units,
            )

    def constants_of_motion(self, units="natural"):
        """Computes the energy, angular momentum, and carter constant for the orbit.
        Computes dimensionless constants in geometried units by default.
        M and mu must be defined in order to convert to physical units.

        Parameters
        ----------
        units : str, optional
            units to return the constants of motion in (options are
            "natural", "mks" and "cgs"), defaults to "natural"

        Returns
        -------
        tuple(double, double, double)
            tuple of the form (E, L, Q)
        """
        constants = self.E, self.L, self.Q
        if units == "natural":
            return constants

        if self.M is None or self.mu is None:
            raise ValueError(
                "M and mu must be specified to convert constants of motion to physical units"
            )

        if units == "mks":
            E, L, Q = scale_constants(constants, 1, self.mu / self.M)
            return (
                energy_in_joules(E, self.M),
                angular_momentum_in_mks(L, self.M),
                carter_constant_in_mks(Q, self.M),
            )

        if units == "cgs":
            E, L, Q = scale_constants(constants, 1, self.mu / self.M)
            return (
                energy_in_ergs(E, self.M),
                angular_momentum_in_cgs(L, self.M),
                carter_constant_in_cgs(Q, self.M),
            )

        raise ValueError("units must be one of 'natural', 'mks', or 'cgs'")

    def four_velocity(self, initial_phases=None):
        r"""Computes the four velocity of the orbit as a function of Mino time using
        the geodesic equation.

        Parameters
        ----------
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`

        Returns
        -------
        tuple(function, function, function, function)
            components of the four velocity (i.e.
            :math:`u^t,u^r,u^\theta,u^\phi`)
        """
        if initial_phases is None:
            initial_phases = self.initial_phases
        t, r, theta, phi = self.trajectory(initial_phases)
        spacetime = KerrSpacetime(self.a)
        constants = self.E, self.L, self.Q

        return spacetime.four_velocity(
            t,
            r,
            theta,
            phi,
            constants,
            self.upsilon_r,
            self.upsilon_theta,
            initial_phases,
        )

    def _four_velocity_norm(self, initial_phases=None):
        r"""Computes the norm of the four velocity of the orbit as a function of Mino time

        Parameters
        ----------
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`

        Returns
        -------
        function
            norm of the four velocity :math:`g_{\mu\nu}u^\mu u^\nu`
        """
        if initial_phases is None:
            initial_phases = self.initial_phases
        t, r, theta, phi = self.trajectory(initial_phases)
        spacetime = KerrSpacetime(self.a)
        constants = self.E, self.L, self.Q
        t_prime, r_prime, theta_prime, phi_prime = self.four_velocity(
            initial_phases=initial_phases
        )

        def norm(time):
            u = [t_prime(time), r_prime(time), theta_prime(time), phi_prime(time)]
            return spacetime.norm(t(time), r(time), theta(time), phi(time), u)

        return norm

    def _numerical_four_velocity_norm(self, dx=1e-6, initial_phases=None):
        r"""Computes the norm of the four velocity of the orbit as a function of Mino time

        Parameters
        ----------
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`

        Returns
        -------
        function
            norm of the four velocity :math:`g_{\mu\nu}u^\mu u^\nu`
        """
        if initial_phases is None:
            initial_phases = self.initial_phases
        t, r, theta, phi = self.trajectory(initial_phases)
        spacetime = KerrSpacetime(self.a)
        constants = self.E, self.L, self.Q
        t_prime, r_prime, theta_prime, phi_prime = self.numerical_four_velocity(
            dx=dx, initial_phases=initial_phases
        )

        def norm(time):
            u = [t_prime(time), r_prime(time), theta_prime(time), phi_prime(time)]
            return spacetime.norm(t(time), r(time), theta(time), phi(time), u)

        return norm

    def numerical_four_velocity(self, dx=1e-6, initial_phases=None):
        r"""Computes the four velocity of the orbit as a function of Mino time using
        numerical differentiation.

        Parameters
        ----------
        dx : double, optional
            step size, defaults to 1e-6
        initial_phases : tuple(double,double,double,double), optional
            initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`, defaults
            to None

        Returns
        -------
        tuple(function, function, function, function)
            components of the four velocity (i.e.
            :math:`u^t,u^r,u^\theta,u^\phi`)
        """
        if initial_phases is None:
            initial_phases = self.initial_phases
        t, r, theta, phi = self.trajectory(initial_phases)

        def u_t(mino_time):
            sigma = r(mino_time) ** 2 + self.a**2 * cos(theta(mino_time)) ** 2
            return (
                -t(mino_time + 2 * dx)
                + 8 * t(mino_time + dx)
                - 8 * t(mino_time - dx)
                + t(mino_time - 2 * dx)
            ) / (12 * dx * sigma)

        def u_r(mino_time):
            sigma = r(mino_time) ** 2 + self.a**2 * cos(theta(mino_time)) ** 2
            return (
                -r(mino_time + 2 * dx)
                + 8 * r(mino_time + dx)
                - 8 * r(mino_time - dx)
                + r(mino_time - 2 * dx)
            ) / (12 * dx * sigma)

        def u_theta(mino_time):
            sigma = r(mino_time) ** 2 + self.a**2 * cos(theta(mino_time)) ** 2
            return (
                -theta(mino_time + 2 * dx)
                + 8 * theta(mino_time + dx)
                - 8 * theta(mino_time - dx)
                + theta(mino_time - 2 * dx)
            ) / (12 * dx * sigma)

        def u_phi(mino_time):
            sigma = r(mino_time) ** 2 + self.a**2 * cos(theta(mino_time)) ** 2
            return (
                -phi(mino_time + 2 * dx)
                + 8 * phi(mino_time + dx)
                - 8 * phi(mino_time - dx)
                + phi(mino_time - 2 * dx)
            ) / (12 * dx * sigma)

        return u_t, u_r, u_theta, u_phi

    def plot(
        self,
        lambda0=0,
        lambda1=10,
        elevation=30,
        azimuth=-60,
        initial_phases=None,
        grid=True,
        axes=True,
        lw=1,
        color="red",
        tau=np.inf,
        point_density=200,
    ):
        r"""Creates a plot of the orbit

        Parameters
        ----------
        lambda0 : double, optional
            starting mino time
        lambda1 : double, optional
            ending mino time
        elevation : double, optional
            camera elevation angle in degrees, defaults to 30
        azimuth : double, optional
            camera azimuthal angle in degrees, defaults to -60
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        grid : bool, optional
            if true, grid lines are shown on plot
        axes : bool, optional
            if true, axes are shown on plot
        lw : double, optional
            linewidth of the orbital trajectory, defaults to 1
        color : str, optional
            color of the orbital trajectory, defaults to "red"
        tau : double, optional
            time constant for the exponential decay of the linewidth,
            defaults to infinity
        point_density : int, optional
            number of points to plot per unit of mino time, defaults to
            200

        Returns
        -------
        matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
            matplotlib figure and axes
        """
        if initial_phases is None:
            initial_phases = self.initial_phases
        trajectory = self.trajectory(initial_phases=initial_phases)

        return plot(
            self.a,
            trajectory,
            lambda0,
            lambda1,
            elevation,
            azimuth,
            grid,
            axes,
            lw,
            color,
            tau,
            point_density,
        )

    def is_visible(self, points, elevation, azimuth):
        """Determines if a point is visible from a given viewing angle or obscured
        by the black hole. Viewing angles are defined as in
        https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html and
        black hole is centered at the origin.

        Parameters
        ----------
        points : array_like
            list of points given in cartesian coordinates
        elevation : double
            camera elevation angle in degrees
        azimuth : double
            camera azimuthal angle in degrees

        Returns
        -------
        np.array
            boolean array indicating whether each point is visible
        """
        return is_visible(self.a, points, elevation, azimuth)

    def animate(
        self,
        filename,
        lambda0=0,
        lambda1=10,
        elevation=30,
        azimuth=-60,
        initial_phases=None,
        grid=True,
        axes=True,
        color="red",
        tau=2,
        tail_length=5,
        lw=2,
        azimuthal_pan=None,
        elevation_pan=None,
        roll=None,
        speed=1,
        background_color=None,
        axis_limit=None,
        plot_components=False,
    ):
        r"""Saves an animation of the orbit as an mp4 file.
        Note that this function requires ffmpeg to be installed and may take several
        minutes to run depending on the length of the animation.

        Parameters
        ----------
        filename : str
            filename to save the animation to
        lambda0 : double, optional
            starting mino time, defaults to 0
        lambda1 : double, optional
            ending mino time, defaults to 10
        elevation : double, optional
            camera elevation angle in degrees, defaults to 30
        azimuth : double, optional
            camera azimuthal angle in degrees, defaults to -60
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        grid : bool, optional
            sets visibility of the grid, defaults to True
        axes : bool, optional
            sets visibility of axes, defaults to True
        color : str, optional
            color of the orbital tail, defaults to "red"
        tau : double, optional
            time constant for the exponential decay in the opacity of
            the tail, defaults to 2
        tail_length : double, optional
            length of the tail in units of mino time, defaults to 5
        lw : double, optional
            linewidth of the orbital trajectory, defaults to 2
        azimuthal_pan : function, optional
            function defining the azimuthal angle of the camera in
            degrees as a function of mino time, defaults to None
        elevation_pan : function, optional
            function defining the elevation angle of the camera in
            degrees as a function of mino time, defaults to None
        roll : function, optional
            function defining the roll angle of the camera in degrees as
            a function of mino time, defaults to None
        axis_limit : function, optional
            sets the axis limit as a function of mino time, defaults to
            None
        speed : double, optional
            playback speed of the animation in units of mino time per
            second (must be a multiple of 1/8), defaults to 1
        background_color : str, optional
            color of the background, defaults to None
        plot_components : bool, optional
            if true, plots the components of the trajectory in addition
            to the trajectory itself, defaults to False
        """
        if initial_phases is None:
            initial_phases = self.initial_phases
        trajectory = self.trajectory(initial_phases=initial_phases)
        if self.stable:
            txt = lambda t: fr"$a = {self.a}\quad p = {self.p}\quad e = {self.e}\quad x = {self.x:.3f}\quad \lambda = {t:.2f}$"
        else:
            txt = lambda t: fr"$a = {self.a}\quad E = {self.E:.3f}\quad L = {self.L:.3f}\quad Q = {self.Q:.3f}\quad \lambda = {t:.2f}$"

        animate(
            self.a,
            trajectory,
            filename,
            lambda0,
            lambda1,
            elevation,
            azimuth,
            grid,
            axes,
            color,
            tau,
            tail_length,
            lw,
            azimuthal_pan,
            elevation_pan,
            roll,
            speed,
            background_color,
            axis_limit,
            plot_components,
            txt,
        )