"""Module containing reusable utility functions for plotting orbits and creating animations"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import numpy as np
from numpy import sin, cos, sqrt, pi
from tqdm import tqdm

def is_visible(a, points, elevation, azimuth):
    """Determines if a point is visible from a given viewing angle or obscured
    by the black hole. Viewing angles are defined as in
    https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html and
    black hole is centered at the origin.

    Parameters
    ----------
    a: double
        spin parameter of the black hole
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
    event_horizon = 1 + sqrt(1 - a**2)

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

def plot(
    a,
    trajectory,
    lambda0=0,
    lambda1=10,
    elevation=30,
    azimuth=-60,
    grid=True,
    axes=True,
    lw=1,
    color="red",
    tau=np.inf,
    point_density=200,
    axes_limits=None,
):
    r"""Creates a plot of the orbit

    Parameters
    ----------
    a: double
        spin parameter of the black hole
    trajectory: tuple
        tuple of functions representing the trajectory
        :math:`(t(\lambda),r(\lambda),\theta(\lambda),\phi(\lambda))`
    lambda0 : double, optional
        starting mino time
    lambda1 : double, optional
        ending mino time
    elevation : double, optional
        camera elevation angle in degrees, defaults to 30
    azimuth : double, optional
        camera azimuthal angle in degrees, defaults to -60
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
    axes_limits : tuple, optional
        limits for the axes (x_min, x_max, y_min, y_max, z_min, z_max), if None, limits are set automatically based on the trajectory and event horizon

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        matplotlib figure and axes
    """
    lambda_range = lambda1 - lambda0
    num_pts = int(lambda_range * point_density)
    time = np.linspace(lambda0, lambda1, num_pts)

    t, r, theta, phi = trajectory

    # compute trajectory in cartesian coordinates
    trajectory_x = r(time) * sin(theta(time)) * cos(phi(time))
    trajectory_y = r(time) * sin(theta(time)) * sin(phi(time))
    trajectory_z = r(time) * cos(theta(time))
    trajectory = np.column_stack((trajectory_x, trajectory_y, trajectory_z))

    # create sphere with radius equal to event horizon radius
    event_horizon = 1 + sqrt(1 - a**2)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x_sphere = event_horizon * np.outer(np.cos(u), np.sin(v))
    y_sphere = event_horizon * np.outer(np.sin(u), np.sin(v))
    z_sphere = event_horizon * np.outer(np.ones(np.size(u)), np.cos(v))

    # replace z values for points behind the black hole with nan so they are not plotted
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/masked_demo.html
    visible = is_visible(a, trajectory, elevation, azimuth)
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
    if axes_limits is None:
        x_values = np.concatenate((trajectory_x, x_sphere.flatten()))
        y_values = np.concatenate((trajectory_y, y_sphere.flatten()))
        z_values = np.concatenate((trajectory_z, z_sphere.flatten()))
        axes_limits = (x_values.min(), x_values.max(), y_values.min(), y_values.max(), z_values.min(), z_values.max())
    ax.set_xlim(axes_limits[0:2])
    ax.set_ylim(axes_limits[2:4])
    ax.set_zlim(axes_limits[4:6])
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

def animate(
    a,
    trajectory,
    filename,
    lambda0=0,
    lambda1=10,
    elevation=30,
    azimuth=-60,
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
    txt=lambda t: fr"\lambda = {t:.2f}",
    axes_limits=None,
):
    r"""Saves an animation of the orbit as an mp4 file.
    Note that this function requires ffmpeg to be installed and may take several
    minutes to run depending on the length of the animation.

    Parameters
    ----------
    a: double
        spin parameter of the black hole
    trajectory: tuple
        tuple of functions representing the trajectory
        :math:`(t(\lambda),r(\lambda),\theta(\lambda),\phi(\lambda))`
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
    txt: function, optional
        function that takes mino time as input and outputs a string to
        be displayed on the plot, defaults to a function that displays
        the mino time with 2 decimal places
    axes_limits : tuple, optional
        limits for the axes (x_min, x_max, y_min, y_max, z_min, z_max), if None, limits are set automatically based on the trajectory and event horizon
    """
    lambda_range = lambda1 - lambda0
    point_density = 240  # number of points per unit of mino time
    num_pts = int(lambda_range * point_density)  # total number of points
    time = np.linspace(lambda0, lambda1, num_pts)
    speed_multiplier = int(speed * 8)
    num_frames = int(num_pts / speed_multiplier)
    # compute trajectory
    t, r, theta, phi = trajectory

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

    eh = 1 + sqrt(1 - a**2)  # event horizon radius

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
    if axes_limits is None:
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
        axes_limits = (-limit, limit) * 3
    ax.set_xlim(axes_limits[0:2])
    ax.set_ylim(axes_limits[2:4])
    ax.set_zlim(axes_limits[4:6])
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
            visible = is_visible(
                a, trajectory[j0:j], updated_elevation, updated_azimuth
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
                text.set_text(txt(current_time))

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
