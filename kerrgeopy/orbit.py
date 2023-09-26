"""
Module containing the Orbit class
"""
from .spacetime import KerrSpacetime
from .initial_conditions import *
from .units import *
from .constants import scale_constants
from .frequencies import mino_frequencies, fundamental_frequencies
from numpy import sin, cos, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

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
            self.stable = False
            self.upsilon_r, self.upsilon_theta = plunging_mino_frequencies(a,E,L,Q)
            self.initial_phases = plunging_orbit_initial_phases(a,initial_position,initial_velocity)
    
    def orbital_parameters(self):
        """
        Returns the orbital parameters :math:`(a,p,e,x)` of the orbit. Raises a ValueError if the orbit is not stable.

        :return: tuple of orbital parameters :math:`(a,p,e,x)`
        :rtype: tuple(double,double,double,double)
        """
        if not self.stable: raise ValueError("Orbit is not stable")
    
        return self.a, self.p, self.e, self.x

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
            from .stable_orbit import StableOrbit
            orbit = StableOrbit(self.a,self.p,self.e,self.x,M=self.M,mu=self.mu)
            return orbit.trajectory(initial_phases,distance_units,time_units)
        else:
            from .plunging_orbit import PlungingOrbit
            orbit = PlungingOrbit(self.a,self.E,self.L,self.Q,M=self.M,mu=self.mu)
            return orbit.trajectory(initial_phases,distance_units,time_units)
        
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
        Computes the four velocity of the orbit as a function of Mino time

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
    
    def four_velocity_norm(self,initial_phases=None):
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
        t_prime, r_prime, theta_prime, phi_prime = self.four_velocity(initial_phases)

        def norm(time):
            u = [t_prime(time),r_prime(time),theta_prime(time),phi_prime(time)]
            return spacetime.norm(t(time),r(time),theta(time),phi(time),u)

        return norm

    def plot(self,lambda0=0, lambda1=20, elevation=30 ,azimuth=-60, initial_phases=None, grid=True, axes=True, thickness=1):
        r"""
        Creates a plot of the orbit

        :param lambda0: starting mino time
        :type lambda0: double, optional
        :param lambda1: ending mino time
        :type lambda1: double, optional
        :param elevation: camera elevation angle in degrees
        :type elevation: double, optional
        :param azimuth: camera azimuthal angle in degrees
        :type azimuth: double, optional
        :param initial_phases: tuple of initial phases :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        :type initial_phases: tuple, optional
        :param grid: if true, grid lines are shown on plot
        :type grid: bool, optional
        :param axes: if true, axes are shown on plot
        :type axes: bool, optional
        :param thickness: line thickness of the orbit
        :type thickness: double, optional

        :return: matplotlib figure and axes
        :rtype: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if initial_phases is None: initial_phases = self.initial_phases
        lambda_range = lambda1 - lambda0
        point_density = 500
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
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = event_horizon * np.outer(np.cos(u), np.sin(v))
        y_sphere = event_horizon * np.outer(np.sin(u), np.sin(v))
        z_sphere = event_horizon * np.outer(np.ones(np.size(u)), np.cos(v))

        # convert viewing angles to radians
        elevation_rad = elevation*pi/180
        azimuth_rad = azimuth*pi/180

        # https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
        view_plane_normal = [cos(elevation_rad)*cos(azimuth_rad),cos(elevation_rad)*sin(azimuth_rad),sin(elevation_rad)]
        # matplotlib has no ray tracer so points behind the black hole must be filtered out manually
        # for each trajectory point compute the component normal to the viewing plane
        normal_component = np.apply_along_axis(lambda x: np.dot(view_plane_normal,x),1,trajectory)
        # compute the projection of each trajectory point onto the viewing plane
        projection = trajectory-np.transpose(normal_component*np.transpose(np.broadcast_to(view_plane_normal,(num_pts,3))))
        # find points in front of the viewing plane or outside the event horizon when projected onto the viewing plane
        condition = (np.linalg.norm(trajectory,axis=1) > event_horizon) & ((normal_component >= 0) | (np.linalg.norm(projection,axis=1) > event_horizon))
        x_visible = trajectory_x[condition]
        y_visible = trajectory_y[condition]
        z_visible = trajectory_z[condition]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        # plot black hole
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='black',shade=False)
        # plot orbit
        ax.scatter(x_visible,y_visible,z_visible,color="red",s=thickness)

        # set viewing angle
        ax.view_init(elevation,azimuth)
        # set equal aspect ratio and orthogonal projection
        ax.set_box_aspect([np.ptp(x_visible),np.ptp(y_visible),np.ptp(z_visible)])
        # https://matplotlib.org/stable/gallery/mplot3d/projections.html
        ax.set_proj_type('ortho')

        # turn off grid and axes if specified
        if not grid: ax.grid(False)
        if not axes: ax.axis("off")

        return fig, ax
    
    def is_visible(self,point,elevation,azimuth):
        """
        Determines if a point is visible from a given viewing angle or obscured by the black hole. 
        Viewing angles are defined as in https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html and black hole is centered at the origin.

        :param point: cartersian coordinates of point to test
        :type point: array_like
        :param elevation: camera elevation angle in degrees
        :type elevation: double
        :param azimuth: camera azimuthal angle in degrees
        :type azimuth: double
        """
        # compute event horizon radius
        event_horizon = 1+sqrt(1-self.a**2)

        # convert viewing angles to radians
        elevation_rad = elevation*pi/180
        azimuth_rad = azimuth*pi/180

        # https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
        view_plane_normal = [cos(elevation_rad)*cos(azimuth_rad),cos(elevation_rad)*sin(azimuth_rad),sin(elevation_rad)]

        # compute the component normal to the viewing plane
        normal_component = np.dot(view_plane_normal,point)
        # compute the projection of the point onto the viewing plane
        projection = point-normal_component*view_plane_normal

        # test if point is outside the event horizon and either in front of the viewing plane or outside the event horizon when projected onto the viewing plane
        return True if (np.linalg.norm(point) > event_horizon) & ((normal_component >= 0) | (np.linalg.norm(projection) > event_horizon)) else False
    
    def animate(self,filename,lambda0=0, lambda1=10, elevation=30 ,azimuth=-60, initial_phases=None, grid=True, axes=True, thickness=2, tail_length="long"):
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
        :param thickness: thickness of the tail of the orbit, defaults to 2
        :type thickness: double, optional
        :param tail: sets the length of the tail (options are "short", "long" and "none"), defaults to "long"
        :type tail: str, optional
        """
        lambda_range = lambda1 - lambda0
        point_density = 200
        num_pts = int(lambda_range*point_density)
        time = np.linspace(lambda0,lambda1,num_pts)

        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(projection='3d')
        eh = 1+sqrt(1-self.a**2)

        body = ax.scatter([],[],[],c="black")
        tail = ax.scatter([],[],[],c="red", s=thickness)

        t, r, theta, phi = self.trajectory(initial_phases)

        ax.view_init(elevation,azimuth)
        x = r(time)*sin(theta(time))*cos(phi(time))
        y = r(time)*sin(theta(time))*sin(phi(time))
        z = r(time)*cos(theta(time))
        trajectory = np.column_stack((x,y,z))

        azimuth_rad = azimuth*pi/180
        elevation_rad = elevation*pi/180

        # https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
        view_plane_normal = [cos(elevation_rad)*cos(azimuth_rad),cos(elevation_rad)*sin(azimuth_rad),sin(elevation_rad)]

        # matplotlib has no ray tracer so points behind the black hole must be filtered out manually
        # for each trajectory point compute the component normal to the viewing plane
        normal_component = np.apply_along_axis(lambda x: np.dot(view_plane_normal,x),1,trajectory)
        # compute the projection of each trajectory point onto the viewing plane
        projection = trajectory-np.transpose(normal_component*np.transpose(np.broadcast_to(view_plane_normal,(num_pts,3))))
        # find points in front of the viewing plane or outside the event horizon when projected onto the viewing plane
        condition = (np.dot(trajectory,view_plane_normal) >= 0) | (np.linalg.norm(projection,axis=1) > eh)

        # create sphere with radius equal to event horizon radius
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = eh * np.outer(np.cos(u), np.sin(v))
        y_sphere = eh * np.outer(np.sin(u), np.sin(v))
        z_sphere = eh * np.outer(np.ones(np.size(u)), np.cos(v))

        # plot black hole
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='black',shade=False,zorder=0)

        # set axis limits
        ax.set_xlim([x.min(),x.max()])
        ax.set_ylim([y.min(),y.max()])
        ax.set_zlim([z.min(),z.max()])

        # set equal aspect ratio and orthogonal projection
        ax.set_box_aspect([np.ptp(x),np.ptp(y),np.ptp(z)])
        # https://matplotlib.org/stable/gallery/mplot3d/projections.html
        ax.set_proj_type('ortho')

         # turn off grid and axes if specified
        if not grid: ax.grid(False)
        if not axes: ax.axis("off")

        def animate(i,body,tail):
            # adjust length of tail
            start = 0
            if tail_length == "short": start = max(0,i-50)
            elif tail_length == "none": start = i

            condition_slice = condition[start:i]
            body._offsets3d = ([x[i]],[y[i]],[z[i]])
            tail._offsets3d = (x[start:i][condition_slice],y[start:i][condition_slice],z[start:i][condition_slice])
            
        # save to file
        ani = FuncAnimation(fig,animate,num_pts,fargs=(body,tail))
        FFwriter = FFMpegWriter(fps=60)
        ani.save(filename, writer = FFwriter)