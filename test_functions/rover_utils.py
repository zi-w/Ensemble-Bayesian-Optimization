from itertools import izip

import numpy as np
import scipy.interpolate as si


class Trajectory:

    def __init__(self):
        pass

    def set_params(self, start, goal, params):
        raise NotImplemented

    def get_points(self, t):
        raise NotImplemented

    @property
    def param_size(self):
        raise NotImplemented


class PointBSpline(Trajectory):
    """
    dim : number of dimensions of the state space
    num_points : number of internal points used to represent the trajectory.
                    Note, internal points are not necessarily on the trajectory.
    """

    def __init__(self, dim, num_points):
        self.tck = None
        self.d = dim
        self.npoints = num_points

    """
    Set fit the parameters of the spline from a set of points. If values are given for start or goal,
    the start or endpoint of the trajectory will be forces on those points, respectively.
    """

    def set_params(self, params, start=None, goal=None):

        points = params.reshape((-1, self.d)).T

        if start is not None:
            points = np.hstack((start[:, None], points))

        if goal is not None:
            points = np.hstack((points, goal[:, None]))

        self.tck, u = si.splprep(points, k=3)

        if start is not None:
            for a, sv in izip(self.tck[1], start):
                a[0] = sv

        if goal is not None:
            for a, gv in izip(self.tck[1], goal):
                a[-1] = gv

    def get_points(self, t):
        assert self.tck is not None, "Parameters have to be set with set_params() before points can be queried."
        return np.vstack(si.splev(t, self.tck)).T

    @property
    def param_size(self):
        return self.d * self.npoints


def simple_rbf(x, point):
    return (1 - np.exp(-np.sum(((x - point) / 0.25) ** 2)))


class RoverDomain:
    """
    Rover domain defined on R^d
    cost_fn : vectorized function giving a scalar cost to states
    start : a start state for the rover
    goal : a goal state
    traj : a parameterized trajectory object offering an interface
            to interpolate point on the trajectory
    s_range : the min and max of the state with s_range[0] in R^d are
                the mins and s_range[1] in R^d are the maxs
    """

    def __init__(self, cost_fn,
                 start,
                 goal,
                 traj,
                 s_range,
                 start_miss_cost=None,
                 goal_miss_cost=None,
                 force_start=True,
                 force_goal=True,
                 only_add_start_goal=True,
                 rnd_stream=None):
        self.cost_fn = cost_fn
        self.start = start
        self.goal = goal
        self.traj = traj
        self.s_range = s_range
        self.rnd_stream = rnd_stream
        self.force_start = force_start
        self.force_goal = force_goal

        self.goal_miss_cost = goal_miss_cost
        self.start_miss_cost = start_miss_cost

        if self.start_miss_cost is None:
            self.start_miss_cost = simple_rbf
        if self.goal_miss_cost is None:
            self.goal_miss_cost = simple_rbf

        if self.rnd_stream is None:
            self.rnd_stream = np.random.RandomState(np.random.randint(0, 2 ** 32 - 1))

    # return the negative cost which need to be optimized
    def __call__(self, params, n_samples=1000):
        self.set_params(params)

        return -self.estimate_cost(n_samples=n_samples)

    def set_params(self, params):
        self.traj.set_params(params + self.rnd_stream.normal(0, 1e-4, params.shape),
                             self.start if self.force_start else None,
                             self.goal if self.force_goal else None)

    def estimate_cost(self, n_samples=1000):
        # get points on the trajectory
        points = self.traj.get_points(np.linspace(0, 1.0, n_samples, endpoint=True))
        # compute cost at each point
        costs = self.cost_fn(points)

        # estimate (trapezoidal) the integral of the cost along traj
        avg_cost = 0.5 * (costs[:-1] + costs[1:])
        l = np.linalg.norm(points[1:] - points[:-1], axis=1)
        total_cost = np.sum(l * avg_cost)

        if not self.force_start:
            total_cost += self.start_miss_cost(points[0], self.start)
        if not self.force_goal:
            total_cost += self.goal_miss_cost(points[-1], self.goal)

        return total_cost

    @property
    def input_size(self):
        return self.traj.param_size


class AABoxes:
    def __init__(self, lows, highs):
        self.l = lows
        self.h = highs

    def contains(self, X):
        if X.ndim == 1:
            X = X[None, :]

        lX = self.l.T[None, :, :] <= X[:, :, None]
        hX = self.h.T[None, :, :] > X[:, :, None]

        return (lX.all(axis=1) & hX.all(axis=1))


class NegGeom:
    def __init__(self, geometry):
        self.geom = geometry

    def contains(self, X):
        return ~self.geom.contains(X)


class UnionGeom:
    def __init__(self, geometries):
        self.geoms = geometries

    def contains(self, X):
        return np.any(np.hstack([g.contains(X) for g in self.geoms]), axis=1, keepdims=True)


class ConstObstacleCost:
    def __init__(self, geometry, cost):
        self.geom = geometry
        self.c = cost

    def __call__(self, X):
        return self.c * self.geom.contains(X)


class ConstCost:
    def __init__(self, cost):
        self.c = cost

    def __call__(self, X):
        if X.ndim == 1:
            X = X[None, :]
        return np.ones((X.shape[0], 1)) * self.c


class AdditiveCosts:
    def __init__(self, fns):
        self.fns = fns

    def __call__(self, X):
        return np.sum(np.hstack([f(X) for f in self.fns]), axis=1)


class GMCost:
    def __init__(self, centers, sigmas, weights=None):
        self.c = centers
        self.s = sigmas
        if self.s.ndim == 1:
            self.s = self.s[:, None]
        self.w = weights
        if weights is None:
            self.w = np.ones(centers.shape[0])

    def __call__(self, X):
        if X.ndim == 1:
            X = X[None, :]

        return np.exp(-np.sum(((X[:, :, None] - self.c.T[None, :, :]) / self.s.T[None, :, :]) ** 2, axis=1)).dot(self.w)


def plot_2d_rover(roverdomain, ngrid_points=100, ntraj_points=100, colormap='RdBu', draw_colorbar=False):
    import matplotlib.pyplot as plt
    # get a grid of points over the state space
    points = [np.linspace(mi, ma, ngrid_points, endpoint=True) for mi, ma in zip(*roverdomain.s_range)]
    grid_points = np.meshgrid(*points)
    points = np.hstack([g.reshape((-1, 1)) for g in grid_points])

    # compute the cost at each point on the grid
    costs = roverdomain.cost_fn(points)

    # get the cost of the current trajectory
    traj_cost = roverdomain.estimate_cost()

    # get points on the current trajectory
    traj_points = roverdomain.traj.get_points(np.linspace(0., 1.0, ntraj_points, endpoint=True))

    # set title to be the total cost
    plt.title('traj cost: {0}'.format(traj_cost))
    print('traj cost: {0}'.format(traj_cost))
    # plot cost function
    cmesh = plt.pcolormesh(grid_points[0], grid_points[1], costs.reshape((ngrid_points, -1)), cmap=colormap)
    if draw_colorbar:
        plt.gcf().colorbar(cmesh)
    # plot traj
    plt.plot(traj_points[:, 0], traj_points[:, 1], 'g')
    # plot start and goal
    plt.plot([roverdomain.start[0], roverdomain.goal[0]], (roverdomain.start[1], roverdomain.goal[1]), 'ok')

    return cmesh


def generate_verts(rectangles):
    poly3d = []
    all_faces = []
    vertices = []
    for l, h in zip(rectangles.l, rectangles.h):
        verts = [[l[0], l[1], l[2]], [l[0], h[1], l[2]], [h[0], h[1], l[2]], [h[0], l[1], l[2]],
                 [l[0], l[1], h[2]], [l[0], h[1], h[2]], [h[0], h[1], h[2]], [h[0], l[1], h[2]]]

        faces = [[0, 1, 2, 3], [0, 3, 7, 4], [3, 2, 6, 7], [7, 6, 5, 4], [1, 5, 6, 2], [0, 4, 5, 1]]

        vert_ind = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [4, 3, 7], [7, 3, 2], [2, 6, 7],
                    [7, 5, 4], [7, 6, 5], [2, 5, 6], [2, 1, 5], [0, 1, 4], [1, 4, 5]]

        plist = [[verts[vert_ind[ix][iy]] for iy in range(len(vert_ind[0]))] for ix in range(len(vert_ind))]
        faces = [[verts[faces[ix][iy]] for iy in range(len(faces[0]))] for ix in range(len(faces))]

        poly3d = poly3d + plist
        vertices = vertices + verts
        all_faces = all_faces + faces

    return poly3d, vertices, all_faces


def plot_3d_forest_rover(roverdomain, rectangles, ntraj_points=100):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    # get the cost of the current trajectory
    traj_cost = roverdomain.estimate_cost()

    # get points on the current trajectory
    traj_points = roverdomain.traj.get_points(np.linspace(0., 1.0, ntraj_points, endpoint=True))

    # convert the rectangles into lists of vertices for matplotlib
    poly3d, verts, faces = generate_verts(rectangles)

    ax = plt.gcf().add_subplot(111, projection='3d')

    # plot start and goal
    ax.scatter((roverdomain.start[0], roverdomain.goal[0]),
               (roverdomain.start[1], roverdomain.goal[1]),
               (roverdomain.start[2], roverdomain.goal[2]), c='k')

    # plot traj
    seg = (zip(traj_points[:-1, :], traj_points[1:, :]))
    ax.add_collection3d(Line3DCollection(seg, colors=[(0, 1., 0, 1.)] * len(seg)))

    # plot rectangles
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors=(0.7, 0.7, 0.7, 1.), linewidth=0.5))

    # set limits of axis to be the same as domain
    s_range = roverdomain.s_range
    ax.set_xlim(s_range[0][0], s_range[1][0])
    ax.set_ylim(s_range[0][1], s_range[1][1])
    ax.set_zlim(s_range[0][2], s_range[1][2])


def main():
    import matplotlib.pyplot as plt
    center = np.array([[1., 1.], [1., 0.0]])
    sigma = np.ones(2) * 0.5
    cost_fn = GMCost(center, sigma)
    start = np.zeros(2) + 0.1
    goal = np.ones(2) * 1 - 0.1

    traj = PointBSpline(dim=2, num_points=3)
    p = np.array([[0.1, 0.5], [0.3, 1.3], [0.75, 1.2]])
    traj.set_params(start, goal, p.flatten())

    domain = RoverDomain(cost_fn,
                         start=start,
                         goal=goal,
                         traj=traj,
                         s_range=np.array([[0., 0.], [2., 2.]]))

    plt.figure()
    plot_2d_rover(domain)
    plt.plot(p[:, 0], p[:, 1], '*g')
    plt.show()


if __name__ == "__main__":
    main()
