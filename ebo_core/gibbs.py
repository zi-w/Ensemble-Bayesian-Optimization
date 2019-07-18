import ebo_core.helper as helper
import numpy as np
import sklearn.random_projection as rp
from gp_tools.gp import SparseFeatureGP, DenseFeatureGP, DenseKernelGP, SparseKernelGP
from gp_tools.representation import TileCoding, IndexToBinarySparse, DenseKernel, DenseL1Kernel, SparseKernel, SparseRPTilecoding
from scipy.special import comb
from builtins import range


# remember to set random seed somewhere
class GibbsSampler(object):
    def __init__(self, X, y, options):
        self.options = options
        x_range, alpha, beta, z, k, dim_limit, n_add, sigma = self.get_params()

        # gp
        self.gp_type = self.options['gp_type']
        self.sigma = sigma
        self.gp_choices = {'l1': self.get_l1gp,
                           'sk': self.get_tilegp,
                           'sf': self.get_tilegp,
                           'dk': self.get_tilegp,
                           'df': self.get_tilegp}
        self.get_gp = self.gp_choices[self.gp_type]

        # data
        self.X = X
        self.y = y
        self.x_range = np.array(x_range)
        self.xdim = x_range.shape[1]

        # gp hyper parameters
        self.beta = beta

        # additive components
        self.n_add = n_add
        self.alpha = alpha
        self.dim_limit = dim_limit

        self.z = z
        if z is None:
            theta = np.random.dirichlet(self.alpha)
            self.z = helper.sample_multinomial(theta, self.xdim, self.dim_limit)

        self.k = k
        if k is None:
            # set the value for lambda
            lmd = np.random.gamma(self.beta[0], self.beta[1], self.xdim)

            self.k = np.maximum(np.random.poisson(lmd), 2)

        # todo:
        self.tilecap = None
        if X.shape[0] != 0:
            self.gp = self.get_gp()

    def get_params(self):
        all_params = ['x_range', 'alpha', 'beta', 'z', 'k', 'dim_limit', 'n_add', 'gp_sigma']
        return [self.options[t] for t in all_params]

    def get_l1gp(self):
        kern = DenseL1Kernel(self.z, self.k)
        gp = DenseKernelGP(self.X, self.y, sigma=self.sigma, kern=kern)
        gp.fit()
        return gp

    def get_tilegp(self):
        nlayers = self.options['nlayers']
        indices = []
        ntiles = []
        hashing = None
        all_cat = np.unique(self.z)
        if self.tilecap:
            hashing_mem = self.tilecap / len(all_cat) / nlayers
            hashing = [rp.UNH(hashing_mem) for _ in range(len(all_cat))]
        for a in all_cat:
            inds = helper.find(self.z == a)
            indices.append(inds)
            ntiles.append(self.k[inds])

        phi = TileCoding(input_indices=indices,
                         # ntiles = input dim x number of layers x tilings
                         ntiles=ntiles,
                         ntilings=[nlayers] * len(indices),
                         hashing=hashing,
                         state_range=self.x_range,
                         rnd_stream=np.random,
                         bias_term=False
                         )

        if self.gp_type == 'sk':
            # densekern > sparsekern \approx sparserp
            sparsekern = SparseKernel(phi, normalize=True)
            gp = SparseKernelGP(self.X, self.y, sigma=0.1, kern=sparsekern)
        elif self.gp_type == 'sf':
            # too slow
            sparsephi = IndexToBinarySparse(phi, normalize=True)
            gp = SparseFeatureGP(self.X, self.y, sigma=0.1, phi=sparsephi)

        elif self.gp_type == 'dk':
            densekern = DenseKernel(phi, normalize=True)
            gp = DenseKernelGP(self.X, self.y, sigma=self.sigma, kern=densekern)
        else:
            random_proj = rp.sparse_random_matrix(300, phi.size, random_state=np.random)
            densephi = SparseRPTilecoding(phi, random_proj=random_proj, normalize=True, output_dense=True)
            gp = DenseFeatureGP(self.X, self.y, sigma=self.sigma, phi=densephi)

        gp.fit()
        return gp

    # idea: can instead get log likelihood on different subset of data for gibbs
    def run(self, niter):
        for i in range(niter):
            # sample z w/ limit on size
            # random permute dimensions
            for d in np.random.permutation(range(self.xdim)):
                # sample z_d
                # initialize
                final_z = self.z.copy()
                zd_old = self.z[d]
                self.z[d] = -1;
                a_size = np.sum(self.z == zd_old)

                max_log_prob_perturbed = np.log(a_size + self.alpha[zd_old]) \
                                         - self.gp.nll + helper.gumbel()

                # find all possible category assignments

                # if z[d] is alone, the possible other category assignment is
                other_cat = np.unique(self.z)
                other_cat = other_cat[np.logical_and(other_cat != zd_old, other_cat != -1)]
                # otherwise, need to remove z[d] and add one additional category
                if a_size > 0 and other_cat.size + 1 < self.n_add:
                    for a in range(self.n_add):
                        if (a not in other_cat) and (a != zd_old):
                            other_cat = np.append(other_cat, [a])
                            break

                # start sampling
                for a in np.random.permutation(other_cat):
                    a_size = np.sum(self.z == a)
                    if a_size < self.dim_limit:
                        self.z[d] = a
                        gp = self.get_gp()
                        log_prob = np.log(a_size + self.alpha[a]) - gp.nll + helper.gumbel()
                        if log_prob > max_log_prob_perturbed:
                            max_log_prob_perturbed = log_prob
                            self.gp = gp
                            final_z = self.z.copy()
                self.z = final_z
                # end of sample z_d

                # sample k_d
                # initialize
                final_k = self.k.copy()
                kd_old = self.k[d]
                beta_post = lambda x: comb(self.beta[0] + x - 1., x) / ((1. / self.beta[1] + 1.) ** x)
                max_log_prob_perturbed = beta_post(kd_old) - self.gp.nll + helper.gumbel()

                # range of k_d is kd_old \pm 5
                other_k = np.arange(-5, 5) + kd_old
                other_k = other_k[np.logical_and(other_k >= 2, other_k != kd_old)]

                # start sampling
                for b in np.random.permutation(other_k):
                    self.k[d] = b
                    gp = self.get_gp()
                    log_prob = beta_post(b) - gp.nll + helper.gumbel()
                    if log_prob > max_log_prob_perturbed:
                        max_log_prob_perturbed = log_prob
                        self.gp = gp
                        final_k = self.k.copy()
                self.k = final_k
        return self.gp, self.z, self.k
