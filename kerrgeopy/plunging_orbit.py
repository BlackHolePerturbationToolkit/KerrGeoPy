from .plunge import *
import matplotlib.pyplot as plt

class PlungingOrbit:
    def __init__(self,a,E,L,Q):
        self.a, self.E, self.L, self.Q = a, E, L, Q
    
    def trajectory(self):
        a, E, L, Q = self.a, self.E, self.L, self.Q
        r, t_r, phi_r = plunging_radial_solutions_complex(a,E,L,Q)
        theta, t_theta, phi_theta = plunging_polar_solutions(a,E,L,Q)

        def t(mino_time):
            return t_r(mino_time) + t_theta(mino_time) + a*L*mino_time
        
        def phi(mino_time):
            return phi_r(mino_time) + phi_theta(mino_time) - a*E*mino_time
        
        return t, r, theta, phi
    
    def plot(self,lambda0=0, lambda1=20, elevation=30 ,azimuth=-60, grid=True, axes=True, thickness=1,point_density=500):
        """
        Creates a plot of the orbit

        :param lambda0: starting mino time
        :type lambda0: double, optional
        :param lambda1: ending mino time
        :type lambda1: double, optional
        :param elevation: camera elevation angle in degrees
        :type elevation: double, optional
        :param azimuth: camera azimuthal angle in degrees
        :type azimuth: double, optional
        :param initial_phases: tuple of initial phases, defaults to (0,0,0,0)
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
        lambda_range = lambda1 - lambda0
        num_pts = lambda_range*point_density
        time = np.linspace(lambda0,lambda1,num_pts)

        t, r, theta, phi = self.trajectory()

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
        condition = (np.dot(trajectory,view_plane_normal) >= 0) | (np.linalg.norm(projection,axis=1) > event_horizon)
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