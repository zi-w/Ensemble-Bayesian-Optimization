import numpy as np
from ebo_core.gibbs import GibbsSampler
from scipy.optimize import minimize


class bo(object):
    def __init__(self, f, X, y, x_range, eval_only, extra, options):
        self.f = f
        self.options = options
        self.eval_only = eval_only

        if eval_only:
            self.newX = extra
        else:
            self.x_range = x_range

            self.well_defined = X.shape[0] > 0
            self.solver = GibbsSampler(X, y, options)

            self.opt_n = options['opt_n']
            self.dx = options['dx']
            self.n_bo = extra
            self.opt_n = np.maximum(self.opt_n, self.n_bo * 2)

        self.max_value = self.options['max_value']
        self.n_bo_top_percent = self.options['n_bo_top_percent']

    def learn(self):
        self.gp, self.z, self.k = self.solver.run(self.options['gibbs_iter'])

    def run(self):
        if self.eval_only:
            ynew = [self.f(x) for x in self.newX]
            return ynew

        # return random inputs if X is empty
        if not self.well_defined:
            xnew = np.random.uniform(self.x_range[0], self.x_range[1], (self.n_bo, self.dx))
            acfnew = [self.max_value] * self.n_bo
            return xnew, acfnew, self.solver.z, self.solver.k

        # learn and optimize
        self.learn()
        # initialization
        xnew = np.empty((self.n_bo, self.dx))
        xnew[0] = np.random.uniform(self.x_range[0], self.x_range[1])
        # optimize group by group
        all_cat = np.unique(self.z)
        for a in np.random.permutation(all_cat):
            active = self.z == a
            af = lambda x: acfun(x, xnew[0], active, self.max_value, self.gp)
            xnew[:, active] = global_minimize(af, self.x_range[:, active], \
                                              self.opt_n, self.n_bo, self.n_bo_top_percent)
        mu, var = self.gp.predict(xnew)
        acfnew = np.squeeze((self.max_value - mu) / np.sqrt(var))
        return xnew, acfnew, self.z, self.k


def global_minimize(f, x_range, n, n_bo=1, n_bo_top_percent=1.0):
    dx = x_range.shape[1]
    tx = np.random.uniform(x_range[0], x_range[1], (n, dx))
    ty = f(tx)
    x0 = tx[ty.argmin()]  # x0 is a 2d array of size 1*dx
    res = minimize(f, x0, bounds=x_range.T, method='L-BFGS-B')
    tx = np.vstack((tx, res.x))
    ty = np.hstack((ty, res.fun))
    inds = ty.argsort()
    thres = np.ceil(n_bo * n_bo_top_percent).astype(int)
    inds_of_inds = np.hstack((range(thres), np.random.permutation(range(thres, len(inds)))))
    inds = inds[inds_of_inds[:n_bo]]
    return tx[inds, :]


def acfun(X, fixX, active_dims, maxval, gp):
    if len(X.shape) > 1:
        nX = np.matlib.repmat(fixX, X.shape[0], 1)
        nX[:, active_dims] = X
    else:
        nX = fixX
        nX[active_dims] = X
    mu, var = gp.predict(nX)
    assert (var > 0).all(), 'error in acfun: variance <= 0??'

    return np.squeeze((maxval - mu) / np.sqrt(var))
