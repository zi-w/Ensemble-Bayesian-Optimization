import os

import numpy as np

from ebo_core.bo import global_minimize
from gp_tools.gp import DenseKernelGP
from gp_tools.representation import DenseL1Kernel


class SimpleQuadratic(object):
    def __init__(self, dx, z, k):
        self.dx = dx
        self.z = z
        self.k = k

    def __call__(self, x):
        x = np.squeeze(x)
        assert len(x) == self.dx
        f = 0
        for i in range(self.dx):
            f -= ((x[i] - 0.5) ** 2.0) / self.dx
        return f


class SampledGpFunc(object):
    def __init__(self, x_range, dx, z, k, n, sigma):
        self.dx = dx
        self.z = z
        self.k = k
        self.sigma = sigma
        self.x_range = x_range
        kern = DenseL1Kernel(self.z, self.k)

        X = np.random.uniform(x_range[0], x_range[1], (n, dx))
        kxx = kern(X) + sigma ** 2 * np.eye(X.shape[0])
        y = np.random.multivariate_normal(np.zeros(n), kxx).T

        self.gp = DenseKernelGP(X, y, sigma=sigma, kern=kern)
        self.gp.fit()
        self.get_max()

    def get_max(self):
        x = self.x_range[0].copy()
        all_cat = np.unique(self.z)
        for a in all_cat:
            active = self.z == a
            k1 = DenseL1Kernel(self.z[active], self.k[active])
            af = lambda x: np.array(k1(x, self.gp.X[:, active])).dot(self.gp.alpha)
            x[active] = np.squeeze(global_minimize(af, self.x_range[:, active], 10000))

        self.argmax = x
        self.f_max = -np.squeeze(np.array(self.gp.kern(x, self.gp.X)).dot(self.gp.alpha))

    def __call__(self, x):
        if x.ndim == 1:
            n = 1
        else:
            n = x.shape[0]
        kXn = np.array(self.gp.kern(x, self.gp.X))
        mu = kXn.dot(self.gp.alpha)
        f = mu
        f += np.random.normal(size=n) * self.sigma
        return -np.squeeze(f)


def sample_z(dx):
    z = np.zeros(dx, dtype=int)
    cnt = 1
    samecnt = 1
    for i in range(1, dx):
        if (samecnt < 3 and np.random.rand() < 0.5) or (samecnt < 4 and np.random.rand() < 0.1):
            z[i] = z[i - 1]
            samecnt += 1
        else:
            z[i] = cnt
            cnt += 1
            samecnt = 1
    return z


def save_sampled_gp_funcs(dx, n=50, nfunc=1, isplot=1, dirnm='mytests'):
    import cPickle as pic
    for i in range(nfunc):
        sigma = 0.01
        z = sample_z(dx)
        k = np.array([10] * dx)
        x_range = np.matlib.repmat([[0.], [1.]], 1, dx)
        f = SampledGpFunc(x_range, dx, z, k, n, sigma)
        filenm = os.path.join(dirnm, str(i) + '_' + str(dx) + '_f.pk')
        pic.dump(f, open(filenm, 'wb'))

    if isplot:
        plot_f(f)

    return f


def plot_f(f, filenm='test_function.eps'):
    # only for 2D functions
    import matplotlib.pyplot as plt
    import matplotlib
    font = {'size': 20}
    matplotlib.rc('font', **font)

    delta = 0.005
    x = np.arange(0.0, 1.0, delta)
    y = np.arange(0.0, 1.0, delta)
    nx = len(x)
    X, Y = np.meshgrid(x, y)

    xx = np.array((X.ravel(), Y.ravel())).T
    yy = f(xx)

    plt.figure()
    plt.contourf(X, Y, yy.reshape(nx, nx), levels=np.linspace(yy.min(), yy.max(), 40))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.colorbar()
    plt.scatter(f.argmax[0], f.argmax[1], s=180, color='k', marker='+')
    plt.savefig(filenm)
