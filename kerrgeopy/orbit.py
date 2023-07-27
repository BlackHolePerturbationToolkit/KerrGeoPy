from numpy import sin, cos, sqrt, pi
from numpy.polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

class Orbit:
    r"""
    Class representing an orbit in Kerr spacetime.

    :param a: spin parameter
    :type a: double
    :param init_position: initial position of the orbit :math:`(t_0,r_0,\theta_0,\phi_0)`
    :type init_position: tuple(double,double,double,double)
    :param init_velocity: initial four-velocity of the orbit :math:`(\frac{dt}{d\tau}(0), \frac{dr}{d\tau}(0),\frac{d\theta}{d\tau}(0),\frac{d\phi}{d\tau}(0))`
    :type init_velocity: tuple(double,double,double,double)

    :ivar a: spin parameter
    :ivar init_position: initial position of the orbit :math:`(t_0,r_0,\theta_0,\phi_0)`
    :ivar E: dimensionless energy
    :ivar L: dimensionless angular momentum
    :ivar Q: dimensionless carter constant
    """
    def __init__(self,a,init_position,init_velocity):
        self.a = a
        self.init_position = init_position
        self.init_velocity = init_velocity

        t0, r0, theta0, phi0 = init_position
        dt0, dr0, dtheta0, dphi0 = init_velocity
        sigma = r0**2+a**2*cos(theta0)**2
        delta = r0**2-2*r0+a**2

        # solve for E and L by writing the t and phi equations as a matrix equation
        A = np.array(
            [[(r0**2+a**2)**2/delta-a**2*(1-cos(theta0)**2), a-a*(r0**2+a**2)/delta],
            [a*(r0**2+a**2)/delta-a, 1/(1-cos(theta0)**2)-a**2/delta]]
        )
        b = np.array([sigma*dt0,sigma*dphi0])
        E, L = np.linalg.solve(A,b)

        # solve for Q by substituting E and L back into the theta equation
        Q = ((sigma*dtheta0)**2 + cos(theta0)**2*(self.a**2*(1-self.E**2)*(1-cos(theta0)**2)+self.L**2))/(1-cos(theta0)**2)

        self.E, self.L, self.Q = E, L, Q

        # standard form of the radial polynomial R(r)
        R = Polynomial([-a**2*Q, 2*L**2+2*Q+2*a**2*E**2-4*a*E*L, a**2*E**2-L**2-Q-a**2, 2, E**2-1])
        radial_roots = R.roots()
        # get the real roots and the complex roots
        real_roots = np.sort(np.real(radial_roots[np.isreal(radial_roots)]))
        complex_roots = radial_roots[np.iscomplex(radial_roots)]

        r_minus = 1-sqrt(1-a**2)

        if len(complex_roots) == 4: raise ValueError("Not a physical orbit")
        if len(complex_roots) == 2:
            self.plunging = True
        else:
            r4, r3, r2, r1 = real_roots
            if (r2 < r_minus) & (r0 < r1) & (r0 > r2): 
                self.plunging = True
            elif (r2 > r_minus) & (r0 < r1) & (r0 > r2):
                self.plunging = False
            elif (r2 > r_minus) & (r0 < r3) & (r0 > r4):
                self.plunging = True
        
    # def trajectory(init_phases=(0,0,0,0)):
    #     pass

    def plot(self,lambda0=0, lambda1=20, elevation=30 ,azimuth=-60, initial_phases=(0,0,0,0), grid=True, axes=True, thickness=1):
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
        point_density = 500
        num_pts = lambda_range*point_density
        time = np.linspace(lambda0,lambda1,num_pts)

        t, r, theta, phi = self.trajectory(initial_phases)

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
    
    def animate(self,filename,lambda0=0, lambda1=10, elevation=30 ,azimuth=-60, initial_phases=(0,0,0,0), grid=True, axes=True, thickness=2, tail="long"):
        """
        Saves an animation of the orbit as an mp4 file

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
        :param initial_phases: tuple of initial phases, defaults to (0,0,0,0)
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
        num_pts = int(2e3)
        time = np.linspace(lambda0,lambda1,num_pts)

        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(projection='3d')
        eh = 1+sqrt(1-self.a**2)

        body = ax.scatter([],[],[],c="black")
        tail = ax.scatter([],[],[],c="red",s=thickness)

        t, r, theta, phi = self.trajectory(initial_phases)

        ax.view_init(35,-60)
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
            if tail == "short": start = max(0,i-50)
            elif tail == "none": start = i

            condition_slice = condition[start:i]
            body._offsets3d = ([x[i]],[y[i]],[z[i]])
            tail._offsets3d = (x[start:i][condition_slice],y[start:i][condition_slice],z[start:i][condition_slice])
            
        # save to file
        ani = FuncAnimation(fig,animate,num_pts,fargs=(body,tail))
        FFwriter = FFMpegWriter(fps=60)
        ani.save(filename, writer = FFwriter)