from .plunge import *
from .plunge import _plunging_radial_roots
from .geodesics import *
from .orbit import Orbit
import matplotlib.pyplot as plt

class PlungingOrbit(Orbit):
    def __init__(self,a,E,L,Q):
        self.a, self.E, self.L, self.Q = a, E, L, Q
        self.upsilon_r, self.upsilon_theta = plunging_mino_frequencies(a,E,L,Q)

    def trajectory(self,initial_phases=(0,0,0,0)):
        a, E, L, Q = self.a, self.E, self.L, self.Q
        radial_roots = _plunging_radial_roots(a,E,L,Q)
        if np.iscomplex(radial_roots[3]):
            return self._complex_trajectory(initial_phases)
        else:
            return self._real_trajectory(initial_phases)
    
    def _complex_trajectory(self,initial_phases=(0,0,0,0)):
        a, E, L, Q = self.a, self.E, self.L, self.Q
        upsilon_r, upsilon_theta = self.upsilon_r, self.upsilon_theta
        r_phases, t_r, phi_r = plunging_radial_solutions_complex(a,E,L,Q)
        theta_phases, t_theta, phi_theta = plunging_polar_solutions(a,E,L,Q)
        q_t0, q_r0, q_theta0, q_phi0 = initial_phases

        q_r0 = q_r0+pi
        # Calculate normalization constants so that t = 0 and phi = 0 at lambda = 0 when q_t0 = 0 and q_phi0 = 0 
        C_t = t_r(q_r0)+t_theta(q_theta0)
        C_phi= phi_r(q_r0)+phi_theta(q_theta0)

        def t(mino_time):
            return t_r(upsilon_r*mino_time+q_r0) + t_theta(upsilon_theta*mino_time+q_theta0) + a*L*mino_time - C_t + q_t0
        
        def r(mino_time):
            return r_phases(upsilon_r*mino_time+q_r0)
        
        def theta(mino_time):
            return theta_phases(upsilon_theta*mino_time+q_theta0)
        
        def phi(mino_time):
            return phi_r(upsilon_r*mino_time+q_r0) + phi_theta(upsilon_theta*mino_time+q_theta0) - a*E*mino_time - C_phi + q_phi0
        
        return t, r, theta, phi
    
    def _real_trajectory(self,initial_phases=(0,0,0,0)):
        a, E, L, Q = self.a, self.E, self.L, self.Q
        constants = (E,L,Q)
        Z = Polynomial([Q,-(Q+a**2*(1-E**2)+L**2),a**2*(1-E**2)])
        radial_roots = _plunging_radial_roots(a,E,L,Q)
        polar_roots = Z.roots()
        if len(polar_roots) == 1:
            z_minus = polar_roots[0]
            z_plus = polar_roots[0]
        elif len(polar_roots) == 2:
            z_minus, z_plus = polar_roots

        upsilon_r, upsilon_theta, upsilon_phi, gamma = mino_frequencies(a,constants,radial_roots,polar_roots)
        r_phases, t_r, phi_r = radial_solutions(a,constants,radial_roots)
        theta_phases, t_theta, phi_theta = polar_solutions(a,constants,polar_roots)
        q_t0, q_r0, q_theta0, q_phi0 = initial_phases

        # Calculate normalization constants so that t = 0 and phi = 0 at lambda = 0 when q_t0 = 0 and q_phi0 = 0 
        C_t = t_r(q_r0)+t_theta(q_theta0)
        C_phi= phi_r(q_r0)+phi_theta(q_theta0)

        def t(mino_time):
            # equation 6
            return q_t0 + gamma*mino_time + t_r(upsilon_r*mino_time+q_r0) + t_theta(upsilon_theta*mino_time+q_theta0) - C_t
        
        def r(mino_time):
            return r_phases(upsilon_r*mino_time+q_r0)
        
        def theta(mino_time):
            return theta_phases(upsilon_theta*mino_time+q_theta0)
        
        def phi(mino_time):
            # equation 6
            return q_phi0 + upsilon_phi*mino_time + phi_r(upsilon_r*mino_time+q_r0) + phi_theta(upsilon_theta*mino_time+q_theta0) - C_phi
        
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