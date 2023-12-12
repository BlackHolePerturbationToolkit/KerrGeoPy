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
from numpy import sin, cos, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm


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
        lambda_range = lambda1 - lambda0
        num_pts = int(lambda_range * point_density)
        time = np.linspace(lambda0, lambda1, num_pts)

        t, r, theta, phi = self.trajectory(initial_phases=initial_phases)

        # compute trajectory in cartesian coordinates
        trajectory_x = r(time) * sin(theta(time)) * cos(phi(time))
        trajectory_y = r(time) * sin(theta(time)) * sin(phi(time))
        trajectory_z = r(time) * cos(theta(time))
        trajectory = np.column_stack((trajectory_x, trajectory_y, trajectory_z))

        # create sphere with radius equal to event horizon radius
        event_horizon = 1 + sqrt(1 - self.a**2)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        x_sphere = event_horizon * np.outer(np.cos(u), np.sin(v))
        y_sphere = event_horizon * np.outer(np.sin(u), np.sin(v))
        z_sphere = event_horizon * np.outer(np.ones(np.size(u)), np.cos(v))

        # replace z values for points behind the black hole with nan so they are not plotted
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/masked_demo.html
        visible = self.is_visible(trajectory, elevation, azimuth)
        trajectory_z_visible = trajectory_z.copy()
        trajectory_z_visible[~visible] = np.nan

        # compute linewidths using exponential decay
        decay = np.flip(0.1 + lw * np.exp(-(time - time[0]) / tau))

        # https://stackoverflow.com/questions/19390895/matplotlib-plot-with-variable-line-width
        points = np.array(
            [
                [[x, y, z]]
                for x, y, z in zip(trajectory_x, trajectory_y, trajectory_z_visible)
            ]
        )
        # create a segment connecting every pair of consecutive points
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        tail = Line3DCollection(segments, linewidth=decay, color=color)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        # plot black hole
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color="black")
        # plot orbit
        ax.add_collection(tail)
        # plot smaller body
        ax.scatter(
            trajectory_x[-1], trajectory_y[-1], trajectory_z[-1], color="black", s=20
        )

        # set axis limits
        x_values = np.concatenate((trajectory_x, x_sphere.flatten()))
        y_values = np.concatenate((trajectory_y, y_sphere.flatten()))
        z_values = np.concatenate((trajectory_z, z_sphere.flatten()))
        ax.set_xlim([x_values.min(), x_values.max()])
        ax.set_ylim([y_values.min(), y_values.max()])
        ax.set_zlim([z_values.min(), z_values.max()])
        # set viewing angle
        ax.view_init(elevation, azimuth)
        # set equal aspect ratio and orthogonal projection
        ax.set_aspect("equal")
        # https://matplotlib.org/stable/gallery/mplot3d/projections.html
        ax.set_proj_type("ortho")

        # turn off grid and axes if specified
        if not grid:
            ax.grid(False)
        if not axes:
            ax.axis("off")

        return fig, ax

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
        # compute event horizon radius
        event_horizon = 1 + sqrt(1 - self.a**2)

        # convert viewing angles to radians
        elevation_rad = elevation * pi / 180
        azimuth_rad = azimuth * pi / 180

        # https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
        view_plane_normal = [
            cos(elevation_rad) * cos(azimuth_rad),
            cos(elevation_rad) * sin(azimuth_rad),
            sin(elevation_rad),
        ]

        normal_component = points.dot(view_plane_normal)
        # compute the projection of each trajectory point onto the viewing plane
        projection = points - np.transpose(
            normal_component
            * np.transpose(np.broadcast_to(view_plane_normal, (len(points), 3)))
        )
        # find points in front of the viewing plane or outside the event horizon when projected onto the viewing plane
        return (
            (normal_component >= 0)
            | (np.linalg.norm(projection, axis=1) > event_horizon)
        ) & (np.linalg.norm(points) > event_horizon)

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
        lambda_range = lambda1 - lambda0
        point_density = 240  # number of points per unit of mino time
        num_pts = int(lambda_range * point_density)  # total number of points
        time = np.linspace(lambda0, lambda1, num_pts)
        speed_multiplier = int(speed * 8)
        num_frames = int(num_pts / speed_multiplier)
        # compute trajectory
        t, r, theta, phi = self.trajectory(initial_phases)

        fig = plt.figure(figsize=((18, 12) if plot_components else (12, 12)))
        if plot_components:
            ax_dict = fig.subplot_mosaic(
                """
            OOOOTT
            OOOORR
            OOOOΘΘ
            OOOOΦΦ
            """,
                per_subplot_kw={
                    "O": {"projection": "3d"},
                    "T": {"facecolor": "none"},
                    "R": {"facecolor": "none"},
                    "Θ": {"facecolor": "none"},
                    "Φ": {"facecolor": "none"},
                },
            )
            ax = ax_dict["O"]

            ax_dict["T"].set_ylabel("$t$")
            ax_dict["R"].set_ylabel("$r$")
            ax_dict["Θ"].set_ylabel(r"$\theta$")
            ax_dict["Φ"].set_ylabel(r"$\phi$")
            (t_plot,) = ax_dict["T"].plot(time, t(time))
            (r_plot,) = ax_dict["R"].plot(time, r(time))
            (theta_plot,) = ax_dict["Θ"].plot(time, theta(time))
            (phi_plot,) = ax_dict["Φ"].plot(time, phi(time))
            # add text with parameters and time
            text = ax.text2D(
                0.05,
                0.95,
                "",
                transform=ax.transAxes,
                fontsize=20,
                bbox=dict(facecolor="none", pad=10.0),
            )

        else:
            ax = fig.add_subplot(projection="3d")

        eh = 1 + sqrt(1 - self.a**2)  # event horizon radius

        # compute trajectory in cartesian coordinates
        trajectory_x = r(time) * sin(theta(time)) * cos(phi(time))
        trajectory_y = r(time) * sin(theta(time)) * sin(phi(time))
        trajectory_z = r(time) * cos(theta(time))
        trajectory = np.column_stack((trajectory_x, trajectory_y, trajectory_z))

        # create sphere with radius equal to event horizon radius
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        x_sphere = eh * np.outer(np.cos(u), np.sin(v))
        y_sphere = eh * np.outer(np.sin(u), np.sin(v))
        z_sphere = eh * np.outer(np.ones(np.size(u)), np.cos(v))

        # plot black hole
        black_hole_color = "#333" if background_color == "black" else "black"
        ax.plot_surface(
            x_sphere,
            y_sphere,
            z_sphere,
            color=black_hole_color,
            shade=(background_color == "black"),
            zorder=0,
        )
        # create orbital tail
        decay = np.flip(
            0.1 + 0.9 * np.exp(-(time - time[0]) / tau)
        )  # exponential decay
        tail = Line3DCollection([], color=color, linewidths=lw, zorder=1)
        ax.add_collection(tail)
        # plot smaller body
        body = ax.scatter([], [], [], c="black")

        # set axis limits so that the black hole is centered
        x_values = np.concatenate((trajectory_x, x_sphere.flatten()))
        y_values = np.concatenate((trajectory_y, y_sphere.flatten()))
        z_values = np.concatenate((trajectory_z, z_sphere.flatten()))
        limit = abs(
            max(
                x_values.min(),
                y_values.min(),
                z_values.min(),
                x_values.max(),
                y_values.max(),
                z_values.max(),
                key=abs,
            )
        )
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        # set equal aspect ratio and orthogonal projection
        ax.set_aspect("equal")
        # https://matplotlib.org/stable/gallery/mplot3d/projections.html
        ax.set_proj_type("ortho")

        # turn off grid and axes if specified
        if not grid:
            ax.grid(False)
        if not axes:
            ax.axis("off")

        # remove margins
        fig.tight_layout()

        # set background color if specified
        if background_color is not None:
            fig.set_facecolor(background_color)
            ax.set_facecolor(background_color)
            # make the panes transparent so that the background color shows through the grid
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

        # start progress bar
        with tqdm(total=num_frames, ncols=80) as pbar:

            def draw_frame(i, body, tail):
                # update progress bar
                pbar.update(1)

                j = speed_multiplier * i
                j0 = max(0, j - tail_length * point_density)
                current_time = time[j]

                # update camera angles
                updated_azimuth = (
                    azimuthal_pan(current_time)
                    if azimuthal_pan is not None
                    else azimuth
                )
                updated_elevation = (
                    elevation_pan(current_time)
                    if elevation_pan is not None
                    else elevation
                )
                updated_roll = roll(current_time) if roll is not None else 0
                ax.view_init(updated_elevation, updated_azimuth, updated_roll)

                # update axis limits
                if axis_limit is not None:
                    updated_limit = axis_limit(current_time)
                    ax.set_xlim(-updated_limit, updated_limit)
                    ax.set_ylim(-updated_limit, updated_limit)
                    ax.set_zlim(-updated_limit, updated_limit)

                # filter out points behind the black hole
                visible = self.is_visible(
                    trajectory[j0:j], updated_elevation, updated_azimuth
                )
                trajectory_z_visible = trajectory_z[j0:j].copy()
                trajectory_z_visible[~visible] = np.nan
                # create segments connecting every consecutive pair of points
                points = np.array(
                    [
                        [[x, y, z]]
                        for x, y, z in zip(
                            trajectory_x[j0:j], trajectory_y[j0:j], trajectory_z_visible
                        )
                    ]
                )
                segments = (
                    np.concatenate([points[:-1], points[1:]], axis=1)
                    if len(points) > 1
                    else []
                )
                # update tail
                tail.set_segments(segments)
                tail.set_alpha(decay[-(j - j0) :])
                # update body
                body._offsets3d = (
                    [trajectory_x[j]],
                    [trajectory_y[j]],
                    [trajectory_z[j]],
                )

                # update plots
                if plot_components:
                    t_plot.set_data(time[:j], t(time[:j]))
                    r_plot.set_data(time[:j], r(time[:j]))
                    theta_plot.set_data(time[:j], theta(time[:j]))
                    phi_plot.set_data(time[:j], phi(time[:j]))
                    # set text
                    if self.stable:
                        text.set_text(
                            f"$a = {self.a}\quad p = {self.p}\quad e = {self.e}\quad x = {self.x:.3f}\quad \lambda = {current_time:.2f}$"
                        )
                    else:
                        text.set_text(
                            f"$a = {self.a}\quad E = {self.E:.3f}\quad L = {self.L:.3f}\quad Q = {self.Q:.3f}\quad \lambda = {current_time:.2f}$"
                        )

            # save to file
            ani = FuncAnimation(fig, draw_frame, num_frames, fargs=(body, tail))
            FFwriter = FFMpegWriter(fps=30)
            # savefig overrides the facecolor so we need to set it again
            if background_color is not None:
                ani.save(
                    filename,
                    savefig_kwargs={"facecolor": background_color},
                    writer=FFwriter,
                )
            else:
                ani.save(filename, writer=FFwriter)
            # close figure so it doesn't show up in notebook
            plt.close(fig)
