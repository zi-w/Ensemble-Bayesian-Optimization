import os
import time

import ebo_core.helper as helper
import numpy as np
from ebo_core.mondrian import MondrianTree
from ebo_core.mypool import MyPool
from builtins import range

try:
    import cPickle as pickle
except:
    import pickle
from gp_tools.representation import DenseL1Kernel
import scipy.linalg

'''
Emsemble Bayesian Optimization
'''


class ebo(object):
    def __init__(self, f, options):
        check_valid_options(options)
        self.f = f
        self.options = options
        print(options)
        # initialization
        if 'X' in options and 'y' in options:
            X, y = options['X'], options['y']
        else:
            X, y = np.empty((0, options['dx'])), np.empty((0, 1))

        self.X, self.y = X.copy(), y.copy()
        options['X'], options['y'] = None, None
        # parallel pool
        self.pool = MyPool(options['pid'], options['datadir'], options['useAzure'], options['thresAzure'])
        # hyper parameters
        if 'z' in options and 'k' in options:
            self.z, self.k = options['z'], options['k']
        else:
            self.z, self.k = None

        # for timing
        self.timing = []
        self.variance = options['gp_sigma'] ** 2

    def get_params(self):
        all_params = ['x_range', 'T', 'B', 'dim_limit', 'min_leaf_size', 'max_n_leaves', 'n_bo', 'n_top']
        return [self.options[t] for t in all_params]

    def run(self):
        x_range, T, B, dim_limit, min_leaf_size, max_n_leaves, n_bo, n_top = self.get_params()
        tstart = self.X.shape[0] // B
        for t in range(tstart, T):
            # search space partition
            ref = self.y.min() if self.y.shape[0] > 0 else None
            self.tree = MondrianTree(self.X, self.y, x_range, max_n_leaves, reference=ref)
            leaves = self.tree.grow_tree(min_leaf_size)
            tot_eval = np.ceil(2.0 * B)

            # this might be dangerous if high dimension and R>1
            tot_volumn = np.array([n.volumn for n in leaves]).sum()
            parameters = [[0, n.X, n.y, n.x_range, False,
                           np.maximum(n_bo, np.ceil((tot_eval * n.volumn // tot_volumn)).astype(int)), self.options] for
                          n in leaves]

            # run bo learning in parallel
            start = time.time()
            res = self.pool.map(parameters, 'iter' + str(t))
            elapsed = time.time() - start

            self.timing.append((self.X.shape[0], elapsed))
            # allocate worker budget
            newX, newacf, z_all, k_all = zip(*res)
            # sync hyper parameters

            if self.options['gibbs_iter'] != 0:
                self.z = helper.mean_z(np.array(z_all), dim_limit)
                self.k = np.mean(k_all, axis=0).astype(int)
            # get newX
            newX = np.vstack(newX)
            newacf = np.hstack(newacf)

            newX = self.choose_newX(newX, newacf, n_top, B)

            # map again to evaluate the selected inputs
            parameters = [[self.f, None, None, None, True, [x], self.options] for x in newX]

            newY = self.pool.map(parameters, 'eval' + str(t), not self.options['func_cheap'])
            # update X, y
            self.X = np.vstack((self.X, newX))
            self.y = np.vstack((self.y, newY))

            self.print_step(newX, t)

            # save
            self.save()

        self.pause()

    def choose_newX(self, newX, newacf, n_top, B):
        print('start choosing newX')
        start = time.time()
        inds = newacf.argsort()
        if 'heuristic' in self.options and self.options['heuristic']:
            n_top = np.ceil(B / 2.).astype(int)
            inds_of_inds = np.hstack((range(n_top), np.random.permutation(range(n_top, len(inds)))))
            newX = newX[inds[inds_of_inds[:B]]]
            return newX

        good_inds = [inds[0]] * B
        len_inds = len(inds)
        jbest = 0
        maxjbest = 0
        next_ind = 1
        all_candidates = np.arange(1, len_inds)
        kern = DenseL1Kernel(self.z, self.k)
        rec = []
        while next_ind < B:
            jnext = maxjbest + n_top
            candidates = all_candidates[:jnext]

            assert len(candidates) > 0, 'B > number of selections?'
            maxlogdet = -np.float('inf')
            jbest = -1
            curX = newX[good_inds[:next_ind]]

            Ky = kern(curX) + self.variance * np.eye(curX.shape[0])
            # compute K + sigma^2I inverse
            factor = scipy.linalg.cholesky(Ky)

            for j in candidates:
                cur_ind = inds[j]
                marginal = self.compute_marginal_det(curX, newX[cur_ind], factor, kern) - newacf[j]

                if maxlogdet < marginal:
                    maxlogdet = marginal
                    jbest = j
            if jbest > maxjbest:
                maxjbest = jbest
            good_inds[next_ind] = inds[jbest]
            all_candidates = all_candidates[all_candidates != jbest]
            next_ind += 1
            rec.append(marginal)
        print('finished choosing newX, eplased time = ', time.time() - start)

        return newX[good_inds]

    def compute_marginal_det(self, X, xx, factor, kern):
        kXn = np.array(kern(xx, X))

        det = np.log(kern.xTxNorm - kXn.dot(scipy.linalg.cho_solve((factor, False), kXn.T)).sum())
        return det

    def get_best(self):
        cur = self.y.argmax(axis=0)
        self.bestx, self.besty = self.X[cur], self.y[cur]
        return self.bestx, self.besty, cur

    def print_step(self, newX, t):
        if self.options['isplot']:
            plot_ebo(self.tree, newX, t)
        _, besty, cur = self.get_best()
        print('t=', t, ', bestid=', cur, ', besty=', besty)

        print('final z=', self.z, ' final k=', self.k)


    def reload(self):
        fnm = self.options['save_file_name']
        if not os.path.isfile(fnm):
            return False
        self.X, self.y, self.z, self.k, self.timing = pickle.load(open(fnm))
        print('Successfully reloaded file.')


    # This will save the pool workers
    def pause(self):
        self.pool.delete_containers()

    # end() will release all the workers.
    def end(self):
        self.pool.end()

    def save(self):
        start = time.time()
        fnm = self.options['save_file_name']
        dirnm = os.path.dirname(fnm)
        if not os.path.exists(dirnm):
            os.makedirs(dirnm)
        pickle.dump([self.X, self.y, self.z, self.k, self.timing], open(fnm, 'wb'))
        print('saving file... ', time.time() - start, ' seconds')


def check_valid_options(options):
    all_params = ['x_range', 'dx', 'max_value', \
                  'T', 'B', 'dim_limit', 'isplot', 'z', 'k', 'alpha', 'beta', \
                  'opt_n', 'pid', 'datadir', 'gibbs_iter', 'useAzure', 'n_add', \
                  'gp_type', 'gp_sigma', 'n_bo', 'n_top', 'min_leaf_size', 'func_cheap', 'thresAzure', 'save_file_name']

    for a in all_params:
        assert a in options, a + ' is not defined in options.'
    assert options['x_range'].shape[1] == options['dx'], 'x_range and dx mismatched.'
    if 'X' in options:
        assert 'y' in options, 'y undefined.'
        assert options['X'].shape[0] == options['y'].shape[0], 'X, y size mismatched.'
        assert options['y'].shape[1] == 1, 'y should be n x 1 matrix.'
        assert options['X'].shape[1] == options['dx'], 'X should be n x dx matrix.'

    # check for gibbs
    beta, alpha, x_range, n_add = options['beta'], options['alpha'], options['x_range'], options['n_add']
    dim_limit = options['dim_limit']
    options['n_add'] = options['dx'] if n_add is None else n_add
    n_add = options['n_add']
    options['dim_limit'] = options['dx'] if dim_limit is None else dim_limit
    assert beta.dtype == float, 'Forgot to set beta to be float?'
    assert isinstance(alpha, float) or alpha.dtype == float, 'Forgot to set alpha to be float?'
    assert x_range.dtype == float, 'Forgot to set x_range to be float?'
    assert len(x_range) == 2 and len(x_range[0]) == len(x_range[1]), 'x_range not well defined'

    if isinstance(alpha, int) or isinstance(alpha, float):
        options['alpha'] = np.array([alpha * 1.0] * n_add)

    assert options['alpha'].shape[0] == n_add, 'alpha must be of size n_add'

    assert options['k'] is None or np.min(options['k']) >= 2, 'number of tiles must be at least 2'


def plot_ebo(tree, newX, t):
    # visualize EBO on 2D function
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ax, fig = tree.visualize()
    for x in newX:
        c = '#1f78b4'
        p = patches.Circle(
            x, 0.01,
            facecolor=c,
            linewidth=0
        )
        ax.add_patch(p)
    plt.show()
    fig.savefig('ebo_result_iteration_' + str(t) + '.eps', format='eps', dpi=1000)
