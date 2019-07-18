from __future__ import print_function
import numpy as np
import sklearn.random_projection as rp
from gp_tools.gp import SparseFeatureGP, DenseFeatureGP, DenseKernelGP, SparseKernelGP
from gp_tools.representation import TileCoding, IndexToBinarySparse, IndexToDense, DenseKernel, SparseKernel, SparseRPTilecoding
from builtins import range

class RandomSinFn:
    def __init__(self, weights, periods):
        self.w = weights
        self.p = periods

    def __call__(self, X):
        assert X.ndim == 2, 'input is expected to have data points row-wise'
        return np.sin(X.dot(self.p)).dot(self.w)


def generate_tilecoding_gps(X, y, sigma, tilecoding, include_sparse=True, include_dense=True):
    # create sparse format tilecoding projector
    sparsephi = IndexToBinarySparse(tilecoding, normalize=True)

    # create dense format tilecoding projector
    densephi = IndexToDense(tilecoding, normalize=True)

    # create dense kernel based off tilecoding
    densekern = DenseKernel(tilecoding, normalize=True)

    # create sparse kernel based off tilecoding
    sparsekern = SparseKernel(tilecoding, normalize=True)

    gps = []

    if include_dense:
        df_gp = DenseFeatureGP(X, y, sigma=sigma, phi=densephi)
        dk_gp = DenseKernelGP(X, y, sigma=sigma, kern=densekern)
        gps = gps + [('Dense Feature GP', df_gp),
                     ('Dense Kernel GP', dk_gp)]
    if include_sparse:
        sf_gp = SparseFeatureGP(X, y, sigma=sigma, phi=sparsephi)
        sk_gp = SparseKernelGP(X, y, sigma=sigma, kern=sparsekern)
        gps = gps + [('Sparse Feature GP', sf_gp),
                     ('Sparse Kernel GP', sk_gp)]
    return gps


def generate_rp_tilecoding_gps(X, y, sigma, tilecoding, random_proj, include_sparse=True, include_dense=True):
    # create dense format tilecoding projector
    densephi = SparseRPTilecoding(tilecoding, random_proj=random_proj, normalize=True, output_dense=True)

    # create dense kernel based off tilecoding
    densekern = DenseKernel(tilecoding, normalize=True)

    # create sparse kernel based off tilecoding
    sparsekern = SparseKernel(tilecoding, normalize=True)

    gps = []

    if include_dense:
        df_gp = DenseFeatureGP(X, y, sigma=sigma, phi=densephi)
        dk_gp = DenseKernelGP(X, y, sigma=sigma, kern=densekern)
        gps = gps + [('Dense Feature GP', df_gp),
                     ('Dense Kernel GP', dk_gp)]
    if include_sparse:
        sk_gp = SparseKernelGP(X, y, sigma=sigma, kern=sparsekern)
        gps = gps + [('Sparse Kernel GP', sk_gp)]
    return gps


def compare_nll(gps):
    nlls = np.array([gp.nll for n, gp in gps])
    diff_nlls = nlls[:, None] - nlls[None, :]

    nll_equal = np.isclose(diff_nlls, np.zeros_like(diff_nlls))
    all_equal = nll_equal.all()
    errors = 0
    if not all_equal:
        for i in range(diff_nlls.shape[0]):
            for j in range(i, diff_nlls.shape[1]):
                if not nll_equal[i, j]:
                    errors = errors + 1
                    print('ERROR: {0}, with nll {1}, differs beyond tolerance from {2}, with nll {3}'.format(gps[i][0], nlls[i], gps[j][0], nlls[j]))
    return errors


def compare_mean_var(test_x, gps):
    results = []
    for n, gp in gps:
        print("Evaluting {0}".format(n))
        results.append(gp.predict(test_x))

    # check variance estimates are positive
    errors = 0
    for (mu, var), (n, gp) in zip(results, gps):
        if (var < 0).any():
            errors = errors + 1
            print('ERROR: Negative variance estimate found in {0}'.format(n))

    all_mu = np.hstack([mu for mu, var in results])
    all_var = np.hstack([var for mu, var in results])

    diff_mu = np.abs(all_mu[:, None, :] - all_mu[:, :, None]).max(axis=0)
    mu_equals = np.isclose(diff_mu, np.zeros_like(diff_mu))
    all_mu_equal = mu_equals.all()
    if not all_mu_equal:
        for i in range(diff_mu.shape[0]):
            for j in range(i, diff_mu.shape[1]):
                if not mu_equals[i, j]:
                    errors = errors + 1
                    print('ERROR: {0} and {1} mean prediction differs, found a max absolute difference of {2}'.format(gps[i][0], gps[j][0], diff_mu[i, j]))

    diff_var = np.abs(all_var[:, None, :] - all_var[:, :, None]).max(axis=0)
    var_equals = np.isclose(diff_var, np.zeros_like(diff_var))
    all_var_equal = var_equals.all()
    if not all_var_equal:
        for i in range(diff_var.shape[0]):
            for j in range(i, diff_var.shape[1]):
                if not var_equals[i, j]:
                    errors = errors + 1
                    print('ERROR: {0} and {1} variance prediction differs, found a max absolute difference of {2}'.format(gps[i][0], gps[j][0], diff_var[i, j]))
    return errors


def fit_gps(test_name, gps):
    for n, gp in gps:
        try:
            print("Fitting {0} for test '{1}'".format(n, test_name))
            gp.fit()
        except:
            print("Exception raise while fitting {0} in test '{1}'".format(n, test_name))
            raise


def create_test_data1(rnd_stream):
    n = 5

    f = lambda x: np.sin(6 * x)
    X = rnd_stream.rand(n, 1)
    y = f(X) + rnd_stream.normal(0, 0.05, n)[:, None]

    test_x = np.linspace(0, 1., 100)[:, None]
    return X, y, f, test_x


def test1(rnd_stream):
    test_name = 'test1'

    print('##########################################################')
    print("Starting test '{0}'".format(test_name))

    t1_sigma = 0.1
    t1_ntiles = 5
    t1_ntilings = 300

    phi = TileCoding(input_indices=[[0]],
                     ntiles=[t1_ntiles],
                     ntilings=[t1_ntilings],
                     state_range=[[0], [1.]],
                     rnd_stream=np.random,
                     bias_term=False,
                     hashing=None
                     )
    X, y, f, test_x = create_test_data1(rnd_stream)
    gps = generate_tilecoding_gps(X, y, sigma=t1_sigma, tilecoding=phi)

    fit_gps(test_name, gps)

    errors = compare_nll(gps)
    errors = errors + compare_mean_var(test_x, gps)

    print("Ending test '{0}' with {1} errors".format(test_name, errors))
    return errors


def create_test_data2(rnd_stream):
    n = 20
    d = 3
    k = 4

    weights = rnd_stream.rand(k)
    periods = rnd_stream.rand(d, k) * 10

    vals = np.linspace(0, 1., 20)[:, None]
    vals = np.meshgrid(*[vals] * 3)
    test_x = np.hstack([v.reshape((-1, 1)) for v in vals])

    f = RandomSinFn(weights, periods)
    X = rnd_stream.rand(n, d)
    y = (f(X) + rnd_stream.normal(0, 0.05, n))[:, None]
    return X, y, f, test_x


def test2(rnd_stream):
    test_name = 'test2'

    print('##########################################################')
    print("Starting test '{0}'".format(test_name))

    t2_sigma = 0.1
    t2_ntiles = 5
    t2_ntilings = 200

    phi = TileCoding(input_indices=[[0], [1], [2]],
                     ntiles=[t2_ntiles] * 3,
                     ntilings=[t2_ntilings] * 3,
                     state_range=[[0] * 3, [1.] * 3],
                     rnd_stream=np.random,
                     bias_term=False,
                     hashing=None
                     )
    X, y, f, test_x = create_test_data2(rnd_stream)
    gps = generate_tilecoding_gps(X, y, sigma=t2_sigma, tilecoding=phi)

    fit_gps(test_name, gps)

    errors = compare_nll(gps)
    errors = errors + compare_mean_var(test_x, gps)

    print("Ending test '{0}' with {1} errors".format(test_name, errors))
    return errors


def create_test_data3(rnd_stream):
    n = 100
    d = 3
    k = 4

    weights = rnd_stream.rand(k)
    periods = rnd_stream.rand(d, k) * 10

    vals = np.linspace(0, 1., 20)[:, None]
    vals = np.meshgrid(*[vals] * 3)
    test_x = np.hstack([v.reshape((-1, 1)) for v in vals])

    f = RandomSinFn(weights, periods)
    X = rnd_stream.rand(n, d)
    y = (f(X) + rnd_stream.normal(0, 0.05, n))[:, None]
    return X, y, f, test_x


def test3(rnd_stream):
    test_name = 'test3'

    print('##########################################################')
    print("Starting test '{0}'".format(test_name))

    t3_sigma = 0.1
    t3_ntiles = 10
    t3_ntilings = 500

    phi = TileCoding(input_indices=[[0, 1, 2]],
                     ntiles=[t3_ntiles],
                     ntilings=[t3_ntilings],
                     state_range=[[0] * 3, [1.] * 3],
                     rnd_stream=np.random,
                     bias_term=False,
                     hashing=None
                     )

    X, y, f, test_x = create_test_data3(rnd_stream)
    random_proj = rp.sparse_random_matrix(100, phi.size)
    gps = generate_rp_tilecoding_gps(X, y,
                                     random_proj=random_proj,
                                     sigma=t3_sigma,
                                     tilecoding=phi,
                                     include_sparse=False)

    fit_gps(test_name, gps)

    errors = compare_nll(gps)
    errors = errors + compare_mean_var(test_x, gps)

    print("Ending test '{0}' with {1} errors".format(test_name, errors))
    return errors


def create_test_data4(rnd_stream):
    n = 300
    d = 3
    k = 4

    weights = rnd_stream.rand(k)
    periods = rnd_stream.rand(d, k) * 10

    vals = np.linspace(0, 1., 20)[:, None]
    vals = np.meshgrid(*[vals] * 3)
    test_x = np.hstack([v.reshape((-1, 1)) for v in vals])

    f = RandomSinFn(weights, periods)
    X = rnd_stream.rand(n, d)
    y = (f(X) + rnd_stream.normal(0, 0.05, n))[:, None]
    return X, y, f, test_x


def test4(rnd_stream):
    test_name = 'test4'

    print('##########################################################')
    print("Starting test '{0}'".format(test_name))

    t4_sigma = 0.1
    t4_ntiles = 10
    t4_ntilings = 500

    phi = TileCoding(input_indices=[[0, 1, 2]],
                     ntiles=[t4_ntiles],
                     ntilings=[t4_ntilings],
                     state_range=[[0] * 3, [1.] * 3],
                     rnd_stream=np.random,
                     bias_term=False,
                     hashing=None)

    X, y, f, test_x = create_test_data4(rnd_stream)
    gps = generate_tilecoding_gps(X, y, sigma=t4_sigma, tilecoding=phi, include_dense=False)

    fit_gps(test_name, gps)

    errors = compare_nll(gps)
    errors = errors + compare_mean_var(test_x, gps)

    print("Ending test '{0}' with {1} errors".format(test_name, errors))
    return errors


print('Starting GP solver tests')

# for reproducibility, create a seed and use the corresponding pseudorandom generator
seed = np.random.randint(np.iinfo(np.int32).max)
rnd_stream = np.random.RandomState(seed)


print('seed used was {0}'.format(seed))

# run test and count errors
errors = sum([test1(rnd_stream), test2(rnd_stream)])

print('##########################################################')

print('TEST COMPLETE')

if errors > 0:
    print('TOTAL ERROR COUNT: {0}'.format(errors))
else:
    print('No errors found')