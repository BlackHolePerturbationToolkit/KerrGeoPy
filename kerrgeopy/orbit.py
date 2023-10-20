"""
Module containing the Orbit class
"""
from .spacetime import KerrSpacetime
from .initial_conditions import *
from .units import *
from .constants import scale_constants, apex_from_constants
from .frequencies import mino_frequencies, fundamental_frequencies
from numpy import sin, cos, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

class Orbit:
    r"""
    Class representing an orbit in Kerr spacetime defined using initial conditions.

    :param a: spin parameter
    :type a: double
    :param initial_position: initial position of the orbit :math:`(t_0,r_0,\theta_0,\phi_0)`
    :type initial_position: tuple(double,double,double,double)
    :param initial_velocity: initial four-velocity of the orbit :math:`(u^t_0,u^r_0,u^\theta_0,u^\phi_0)`
    :type initial_velocity: tuple(double,double,double,double)
    :param M: mass of the primary in solar masses
    :type M: double, optional
    :param mu: mass of the smaller body in solar masses
    :type mu: double, optional

    :ivar a: spin parameter
    :ivar initial_position: initial position of the orbit :math:`(t_0,r_0,\theta_0,\phi_0)`
    :ivar initial_velocity: initial four-velocity of the orbit :math:`(u^t_0,u^r_0,u^\theta_0,u^\phi_0)`
    :ivar E: dimensionless energy
    :ivar L: dimensionless angular momentum
    :ivar Q: dimensionless carter constant
    :ivar stable: boolean indicating whether the orbit is stable
    :ivar upsilon_r: dimensionless radial orbital frequency in Mino time
    :ivar upsilon_theta: dimensionless polar orbital frequency in Mino time
    """
    def __init__(self,a,initial_position,initial_velocity, M=None, mu=None):
        self.a, self.initial_position, self.initial_velocity, self.M, self.mu = a, initial_position, initial_velocity, M, mu

        # check if initial four-velocity is valid
        spacetime = KerrSpacetime(a)
        initial_norm =  spacetime.norm(*initial_position,initial_velocity)
        if initial_norm >= 0: raise ValueError("Initial velocity is not timelike")
        if abs(initial_norm + 1) > 1e-6: raise ValueError("Initial velocity is not normalized")

        E, L, Q = constants_from_initial_conditions(a,initial_position,initial_velocity)
        self.E, self.L, self.Q = E, L, Q

        if is_stable(a,initial_position,initial_velocity):
            self.stable = True
            a, p, e, x = apex_from_constants(a,E,L,Q)
            self.a, self.p, self.e, self.x = a, p, e, x
            self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma = mino_frequencies(a,p,e,x)
            self.omega_r, self.omega_theta, self.omega_phi = fundamental_frequencies(a,p,e,x)
            self.initial_phases = stable_orbit_initial_phases(a,initial_position,initial_velocity)
        else:
            if a == 0: raise ValueError("Schwarzschild plunges are not currently supported")
            self.stable = False
            self.upsilon_r, self.upsilon_theta = plunging_mino_frequencies(a,E,L,Q)
            self.initial_phases = plunging_orbit_initial_phases(a,initial_position,initial_velocity)

    def trajectory(self,initial_phases=None,distance_units="natural",time_units="natural"):
        r"""
        Computes the components of the trajectory as a function of Mino time
        
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
        if self.stable:
            from .stable import stable_trajectory
            return stable_trajectory(self.a,self.p,self.e,self.x,initial_phases,self.M,distance_units,time_units)
        else:
            from .plunge import plunging_trajectory
            return plunging_trajectory(self.a,self.E,self.L,self.Q,initial_phases,self.M,distance_units,time_units)
        
    def constants_of_motion(self, units="natural"):
        """
        Computes the energy, angular momentum, and carter constant for the orbit. Computes dimensionless constants in geometried units by default.
        M and mu must be defined in order to convert to physical units.

        :param units: units to return the constants of motion in (options are "natural", "mks" and "cgs"), defaults to "natural"
        :type units: str, optional

        :return: tuple of the form (E, L, Q)
        :rtype: tuple(double, double, double)
        """
        constants = self.E, self.L, self.Q
        if units == "natural":
            return constants
        
        if self.M is None or self.mu is None: raise ValueError("M and mu must be specified to convert constants of motion to physical units")
        
        if units == "mks":
            E, L, Q = scale_constants(constants,1,self.mu/self.M)
            return energy_in_joules(E,self.M), angular_momentum_in_mks(L,self.M), carter_constant_in_mks(Q,self.M)
        
        if units == "cgs":
            E, L, Q = scale_constants(constants,1,self.mu/self.M)
            return energy_in_ergs(E,self.M), angular_momentum_in_cgs(L,self.M), carter_constant_in_cgs(Q,self.M)
        
        raise ValueError("units must be one of 'natural', 'mks', or 'cgs'")
    

    def four_velocity(self,initial_phases=None):
        r"""
        Computes the four velocity of the orbit as a function of Mino time using the geodesic equation.

        :param initial_phases: tuple of initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        :type initial_phases: tuple, optional

        :return: components of the four velocity (i.e. :math:`u^t,u^r,u^\theta,u^\phi`)
        :rtype: tuple(function, function, function, function)
        """
        if initial_phases is None: initial_phases = self.initial_phases
        t, r, theta, phi = self.trajectory(initial_phases)
        spacetime = KerrSpacetime(self.a)
        constants = self.E, self.L, self.Q

        return spacetime.four_velocity(t,r,theta,phi,constants,self.upsilon_r,self.upsilon_theta,initial_phases)
    
    def _four_velocity_norm(self,initial_phases=None):
        r"""
        Computes the norm of the four velocity of the orbit as a function of Mino time

        :param initial_phases: tuple of initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        :type initial_phases: tuple, optional

        :return: norm of the four velocity :math:`g_{\mu\nu}u^\mu u^\nu`
        :rtype: function
        """
        if initial_phases is None: initial_phases = self.initial_phases
        t, r, theta, phi = self.trajectory(initial_phases)
        spacetime = KerrSpacetime(self.a)
        constants = self.E, self.L, self.Q
        t_prime, r_prime, theta_prime, phi_prime = self.four_velocity(initial_phases=initial_phases)

        def norm(time):
            u = [t_prime(time),r_prime(time),theta_prime(time),phi_prime(time)]
            return spacetime.norm(t(time),r(time),theta(time),phi(time),u)

        return norm
    
    def _numerical_four_velocity_norm(self,dx=1e-6,initial_phases=None):
        r"""
        Computes the norm of the four velocity of the orbit as a function of Mino time

        :param initial_phases: tuple of initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        :type initial_phases: tuple, optional

        :return: norm of the four velocity :math:`g_{\mu\nu}u^\mu u^\nu`
        :rtype: function
        """
        if initial_phases is None: initial_phases = self.initial_phases
        t, r, theta, phi = self.trajectory(initial_phases)
        spacetime = KerrSpacetime(self.a)
        constants = self.E, self.L, self.Q
        t_prime, r_prime, theta_prime, phi_prime = self.numerical_four_velocity(dx=dx,initial_phases=initial_phases)

        def norm(time):
            u = [t_prime(time),r_prime(time),theta_prime(time),phi_prime(time)]
            return spacetime.norm(t(time),r(time),theta(time),phi(time),u)

        return norm
    
    def numerical_four_velocity(self,dx=1e-6,initial_phases=None):
        r"""
        Computes the four velocity of the orbit as a function of Mino time using numerical differentiation

        :param dx: step size, defaults to 1e-6
        :type dx: double, optional
        :param initial_phases: initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`, defaults to None
        :type initial_phases: tuple(double,double,double,double), optional

        :return: components of the four velocity (i.e. :math:`u^t,u^r,u^\theta,u^\phi`)
        :rtype: tuple(function, function, function, function)
        """
        if initial_phases is None: initial_phases = self.initial_phases
        t, r, theta, phi = self.trajectory(initial_phases)

        def u_t(mino_time):
            sigma = r(mino_time)**2 + self.a**2*cos(theta(mino_time))**2
            return (t(mino_time+dx)-t(mino_time-dx))/(2*dx*sigma)
        def u_r(mino_time):
            sigma = r(mino_time)**2 + self.a**2*cos(theta(mino_time))**2
            return (r(mino_time+dx)-r(mino_time-dx))/(2*dx*sigma)
        def u_theta(mino_time):
            sigma = r(mino_time)**2 + self.a**2*cos(theta(mino_time))**2
            return (theta(mino_time+dx)-theta(mino_time-dx))/(2*dx*sigma)
        def u_phi(mino_time):
            sigma = r(mino_time)**2 + self.a**2*cos(theta(mino_time))**2
            return (phi(mino_time+dx)-phi(mino_time-dx))/(2*dx*sigma)
        return u_t, u_r, u_theta, u_phi

    def plot(self,lambda0=0, lambda1=10, elevation=30 ,azimuth=-60, initial_phases=None, grid=True, axes=True, lw=1,color="red",tau=np.inf,point_density=200):
        r"""
        Creates a plot of the orbit

        :param lambda0: starting mino time
        :type lambda0: double, optional
        :param lambda1: ending mino time
        :type lambda1: double, optional
        :param elevation: camera elevation angle in degrees, defaults to 30
        :type elevation: double, optional
        :param azimuth: camera azimuthal angle in degrees, defaults to -60
        :type azimuth: double, optional
        :param initial_phases: tuple of initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        :type initial_phases: tuple, optional
        :param grid: if true, grid lines are shown on plot
        :type grid: bool, optional
        :param axes: if true, axes are shown on plot
        :type axes: bool, optional
        :param lw: linewidth of the orbital trajectory, defaults to 1
        :type lw: double, optional
        :param color: color of the orbital trajectory, defaults to "red"
        :type color: str, optional
        :param tau: time constant for the exponential decay of the linewidth, defaults to infinity
        :type tau: double, optional
        :param point_density: number of points to plot per unit of mino time, defaults to 200
        :type point_density: int, optional

        :return: matplotlib figure and axes
        :rtype: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if initial_phases is None: initial_phases = self.initial_phases
        lambda_range = lambda1 - lambda0
        num_pts = int(lambda_range*point_density)
        time = np.linspace(lambda0,lambda1,num_pts)

        t, r, theta, phi = self.trajectory(initial_phases=initial_phases)

        # compute trajectory in cartesian coordinates
        trajectory_x = r(time)*sin(theta(time))*cos(phi(time))
        trajectory_y = r(time)*sin(theta(time))*sin(phi(time))
        trajectory_z = r(time)*cos(theta(time))
        trajectory = np.column_stack((trajectory_x,trajectory_y,trajectory_z))

        # create sphere with radius equal to event horizon radius
        event_horizon = 1+sqrt(1-self.a**2)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        x_sphere = event_horizon * np.outer(np.cos(u), np.sin(v))
        y_sphere = event_horizon * np.outer(np.sin(u), np.sin(v))
        z_sphere = event_horizon * np.outer(np.ones(np.size(u)), np.cos(v))

        # replace z values for points behind the black hole with nan so they are not plotted
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/masked_demo.html
        visible = self.is_visible(trajectory,elevation,azimuth)
        trajectory_z_visible = trajectory_z.copy()
        trajectory_z_visible[~visible] = np.nan

        # compute linewidths using exponential decay
        decay = np.flip(0.1+lw*np.exp(-(time-time[0])/tau))

        # https://stackoverflow.com/questions/19390895/matplotlib-plot-with-variable-line-width
        points = np.array([[[x, y, z]] for x, y, z in zip(trajectory_x,trajectory_y,trajectory_z_visible)])
        # create a segment connecting every pair of consecutive points
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        tail = Line3DCollection(segments, linewidth=decay, color=color)

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(projection='3d')
        
        # plot black hole
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='black')
        # plot orbit
        ax.add_collection(tail)
        # plot smaller body
        ax.scatter(trajectory_x[-1],trajectory_y[-1],trajectory_z[-1],color="black",s=20)

        # set axis limits
        x_values = np.concatenate((trajectory_x,x_sphere.flatten()))
        y_values = np.concatenate((trajectory_y,y_sphere.flatten()))
        z_values = np.concatenate((trajectory_z,z_sphere.flatten()))
        ax.set_xlim([x_values.min(),x_values.max()])
        ax.set_ylim([y_values.min(),y_values.max()])
        ax.set_zlim([z_values.min(),z_values.max()])
        # set viewing angle
        ax.view_init(elevation,azimuth)
        # set equal aspect ratio and orthogonal projection
        ax.set_aspect("equal")
        # https://matplotlib.org/stable/gallery/mplot3d/projections.html
        ax.set_proj_type('ortho')

        # turn off grid and axes if specified
        if not grid: ax.grid(False)
        if not axes: ax.axis("off")

        return fig, ax
    
    def is_visible(self,points,elevation,azimuth):
        """
        Determines if a point is visible from a given viewing angle or obscured by the black hole. 
        Viewing angles are defined as in https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html and black hole is centered at the origin.

        :param points: list of points given in cartesian coordinates
        :type points: array_like
        :param elevation: camera elevation angle in degrees
        :type elevation: double
        :param azimuth: camera azimuthal angle in degrees
        :type azimuth: double

        :return: boolean array indicating whether each point is visible
        :rtype: np.array
        """
        # compute event horizon radius
        event_horizon = 1+sqrt(1-self.a**2)

        # convert viewing angles to radians
        elevation_rad = elevation*pi/180
        azimuth_rad = azimuth*pi/180

        # https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
        view_plane_normal = [cos(elevation_rad)*cos(azimuth_rad),cos(elevation_rad)*sin(azimuth_rad),sin(elevation_rad)]

        normal_component = points.dot(view_plane_normal)
        # compute the projection of each trajectory point onto the viewing plane
        projection = points-np.transpose(normal_component*np.transpose(np.broadcast_to(view_plane_normal,(len(points),3))))
        # find points in front of the viewing plane or outside the event horizon when projected onto the viewing plane
        return ((normal_component >= 0) | (np.linalg.norm(projection,axis=1) > event_horizon)) & (np.linalg.norm(points) > event_horizon)
    
    def animate(self,filename,lambda0=0, lambda1=10, elevation=30 ,azimuth=-60, initial_phases=None, grid=True, axes=True, color="red", tau=2, tail_length=5, 
                lw=2, azimuthal_pan=None, elevation_pan=None, speed=1, background_color=None):
        r"""
        Saves an animation of the orbit as an mp4 file. 
        Note that this function requires ffmpeg to be installed and may take several minutes to run depending on the length of the animation.

        :param filename: filename to save the animation to
        :type filename: str
        :param lambda0: starting mino time, defaults to 0
        :type lambda0: double, optional
        :param lambda1: ending mino time, defaults to 10
        :type lambda1: double, optional
        :param elevation: camera elevation angle in degrees, defaults to 30
        :type elevation: double, optional
        :param azimuth: camera azimuthal angle in degrees, defaults to -60
        :type azimuth: double, optional
        :param initial_phases: tuple of initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        :type initial_phases: tuple, optional
        :param grid: sets visibility of the grid, defaults to True
        :type grid: bool, optional
        :param axes: sets visibility of axes, defaults to True
        :type axes: bool, optional
        :param color: color of the orbital tail, defaults to "red"
        :type color: str, optional
        :param tau: time constant for the exponential decay in the opacity of the tail, defaults to infinity
        :type tau: double, optional
        :param tail_length: length of the tail in units of mino time, defaults to 5
        :type tail_length: double, optional
        :param lw: linewidth of the orbital trajectory, defaults to 2
        :type lw: double, optional
        :param azimuthal_pan: function defining the azimuthal angle of the camera in degrees as a function of mino time, defaults to None
        :type azimuthal_pan: function, optional
        :param elevation_pan: function defining the elevation angle of the camera in degrees as a function of mino time, defaults to None
        :type elevation_pan: function, optional
        :param speed: playback speed of the animation in units of mino time per second (must be a multiple of 1/8), defaults to 1
        :type speed: double, optional
        :param background_color: color of the background, defaults to None
        :type background_color: str, optional
        """
        lambda_range = lambda1 - lambda0
        point_density = 240 # number of points per unit of mino time
        num_pts = int(lambda_range*point_density) # total number of points
        time = np.linspace(lambda0,lambda1,num_pts)
        speed_multiplier = int(speed*8)
        num_frames = int(num_pts/speed_multiplier)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        eh = 1+sqrt(1-self.a**2)

        t, r, theta, phi = self.trajectory(initial_phases)

        ax.view_init(elevation,azimuth)
        trajectory_x = r(time)*sin(theta(time))*cos(phi(time))
        trajectory_y = r(time)*sin(theta(time))*sin(phi(time))
        trajectory_z = r(time)*cos(theta(time))
        trajectory = np.column_stack((trajectory_x,trajectory_y,trajectory_z))

        # create sphere with radius equal to event horizon radius
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        x_sphere = eh * np.outer(np.cos(u), np.sin(v))
        y_sphere = eh * np.outer(np.sin(u), np.sin(v))
        z_sphere = eh * np.outer(np.ones(np.size(u)), np.cos(v))

        # plot black hole
        black_hole_color = "#333" if background_color == "black" else "black"
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color=black_hole_color,shade=(background_color == "black"), zorder=0)
        # create tail
        decay = np.flip(0.1+0.9*np.exp(-(time-time[0])/tau)) # exponential decay
        tail = Line3DCollection([], color=color, linewidths=lw, zorder=1)
        ax.add_collection(tail)
        # plot smaller body
        body = ax.scatter([],[],[],c="black")

        # set axis limits so that the black hole is centered
        x_values = np.concatenate((trajectory_x,x_sphere.flatten()))
        y_values = np.concatenate((trajectory_y,y_sphere.flatten()))
        z_values = np.concatenate((trajectory_z,z_sphere.flatten()))
        limit = abs(max(x_values.min(),y_values.min(),z_values.min(),x_values.max(),y_values.max(),z_values.max(),key=abs))
        ax.set_xlim(-limit,limit)
        ax.set_ylim(-limit,limit)
        ax.set_zlim(-limit,limit)
        # set equal aspect ratio and orthogonal projection
        ax.set_aspect("equal")
        # https://matplotlib.org/stable/gallery/mplot3d/projections.html
        ax.set_proj_type('ortho')
        # set viewing angle
        ax.view_init(elevation,azimuth)
         # turn off grid and axes if specified
        if not grid or background_color is not None: ax.grid(False)
        if not axes or background_color is not None: ax.axis("off")
        # set background color if specified
        if background_color is not None: ax.set_facecolor(background_color)

        # start progress bar
        with tqdm(total=num_frames,ncols=80) as pbar:
            def draw_frame(i,body,tail):
                # update progress bar
                pbar.update(1)

                j = speed_multiplier*i
                j0 = max(0,j-tail_length*point_density)
                mino_time = time[j]

                # update camera angles
                new_azimuth = azimuthal_pan(mino_time) if azimuthal_pan is not None else azimuth
                new_elevation = elevation_pan(mino_time) if elevation_pan is not None else elevation
  
                ax.view_init(new_elevation,new_azimuth)

                # filter out points behind the black hole
                visible = self.is_visible(trajectory[j0:j],new_elevation,new_azimuth)
                trajectory_z_visible = trajectory_z[j0:j].copy()
                trajectory_z_visible[~visible] = np.nan
                # create segments connecting every consecutive pair of points
                points = np.array([[[x, y, z]] for x, y, z in zip(trajectory_x[j0:j],trajectory_y[j0:j],trajectory_z_visible)])
                segments = np.concatenate([points[:-1], points[1:]], axis=1) if len(points) > 1 else []
                # update tail
                tail.set_segments(segments)
                tail.set_alpha(decay[-(j-j0):])
                #tail.set_array(decay[-(j-j0):])
                # update body
                body._offsets3d = ([trajectory_x[j]],[trajectory_y[j]],[trajectory_z[j]])
                                
            # save to file
            ani = FuncAnimation(fig,draw_frame,num_frames,fargs=(body,tail))
            FFwriter = FFMpegWriter(fps=30)
            ani.save(filename, writer = FFwriter)
            # close figure so it doesn't show up in notebook
            plt.close(fig)