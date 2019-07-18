import numpy as np
import scipy.linalg
import scipy.sparse


import sksparse.cholmod as spch

class SparseFeatureGP:
    def __init__(self, X, y, sigma, phi):
        self.X = X
        self.y = y
        self.sigma = sigma
        self.variance = self.sigma ** 2
        self.phi = phi
        self.factor = None

    def fit(self):
        n = self.X.shape[0]
        self.mphi = self.phi(self.X)
        dphi = self.mphi.shape[1]
        self.factor = spch.cholesky_AAt(self.mphi.T, beta=self.variance)

        # precompute some useful quantities
        z = self.mphi.T.dot(self.y)
        self.theta = self.factor(z)

        # compute nll
        ssqr = self.variance
        ll = 0.5 * (z.T.dot(self.theta) / ssqr - self.factor.logdet()
                    - self.y.T.dot(self.y) / ssqr - np.log(2 * np.pi) * n - np.log(ssqr) * (n - dphi))

        self._nll = -ll[0, 0]

    def predict(self, X):
        assert self.factor is not None, "The gp must be fit using fit() before predict is called."
        # predict mean
        phiX = self.phi(X)
        mu = phiX.dot(self.theta)

        # predict variance
        var = self.variance * ((phiX.multiply(self.factor(phiX.T).T)).sum(axis=1) + 1.)
        return mu, np.array(var)

    @property
    def nll(self):
        assert self.factor is not None, "The gp must be fit using fit() before querying the neg log likelihood."
        return self._nll


class DenseFeatureGP:
    def __init__(self, X, y, sigma, phi):
        self.X = X
        self.y = y
        self.sigma = sigma
        self.variance = self.sigma ** 2
        self.phi = phi
        self.factor = None

    def fit(self):
        n = self.X.shape[0]
        self.mphi = self.phi(self.X)
        dphi = self.mphi.shape[1]

        A = self.mphi.T.dot(self.mphi) + self.variance * np.eye(self.phi.size)
        self.factor = scipy.linalg.cholesky(A)

        # precompute some useful quantities
        z = self.mphi.T.dot(self.y)
        # print self.factor.shape, z.shape
        self.theta = scipy.linalg.cho_solve((self.factor, False), z)

        # compute nll
        logdet = 2 * np.sum(np.log(np.diag(self.factor)))
        ssqr = self.variance
        ll = 0.5 * (z.T.dot(self.theta) / ssqr - logdet - self.y.T.dot(self.y) / ssqr - np.log(2 * np.pi) * n - np.log(
            ssqr) * (n - dphi))

        self._nll = -ll[0, 0]

    def predict(self, X):
        assert self.factor is not None, "The gp must be fit using fit() before predict is called."
        # predict mean
        phiX = np.array(self.phi(X))
        mu = phiX.dot(self.theta)
        # predict variance
        var = self.variance * (1. + np.multiply(phiX,
                                                scipy.linalg.cho_solve((self.factor, False), phiX.T).T).sum(axis=1,
                                                                                                            keepdims=True))
        return mu, var

    @property
    def nll(self):
        assert self.factor is not None, "The gp must be fit using fit() before querying the neg log likelihood."
        return self._nll


class DenseKernelGP(object):
    def __init__(self, X, y, sigma, kern):
        assert X.shape[0] > 0
        self.X = X
        self.y = y
        self.sigma = sigma
        self.variance = sigma ** 2
        self.kern = kern
        self.factor = None
        self.Kinv = None

    def fit(self):
        # gram matrix
        Ky = self.kern(self.X) + self.variance * np.eye(self.X.shape[0])
        # compute K + sigma^2I inverse
        self.factor = scipy.linalg.cholesky(Ky)
        logdet = 2 * np.sum(np.log(np.diag(self.factor)))
        self.alpha = scipy.linalg.cho_solve((self.factor, False), self.y)

        # full nll
        self._nll = 0.5 * (logdet + np.sum(self.alpha * self.y) + self.y.shape[0] * np.log(2 * np.pi))

    def predict(self, X):
        assert self.factor is not None, "The gp must be fit using fit() before predict is called."
        # predict mean
        kXn = np.array(self.kern(X, self.X))
        mu = kXn.dot(self.alpha)
        # predict variance
        var = (self.kern.xTxNorm + self.variance
               - np.multiply(kXn, scipy.linalg.cho_solve((self.factor, False), kXn.T).T).sum(axis=1, keepdims=True))
        return mu, var

    @property
    def nll(self):
        assert self.factor is not None, "The gp must be fit using fit() before querying the neg log likelihood."
        return self._nll


class SparseKernelGP(object):
    def __init__(self, X, y, sigma, kern):
        self.X = X
        self.y = y
        self.sigma = sigma
        self.variance = sigma ** 2
        self.kern = kern
        self.factor = None
        self.Kinv = None

    def fit(self):
        # gram matrix
        Ky = self.kern(self.X)
        if scipy.sparse.isspmatrix_csr(Ky):
            Ky = Ky.T
        self.factor = spch.cholesky(Ky, beta=self.variance)

        # precompute some useful quantities
        self.alpha = self.factor(self.y)

        # partial nnl (ignore what kernel cannot change)
        self._nll = 0.5 * (self.factor.logdet() + np.sum(self.alpha * self.y) + self.y.shape[0] * np.log(2 * np.pi))

    def predict(self, X):
        assert self.factor is not None, "The gp must be fit using fit() before predict is called."
        # predict mean
        kxx = self.kern(X, self.X)
        mu = kxx.dot(self.alpha)

        # predict variance
        var = self.kern.xTxNorm + self.variance - kxx.multiply(self.factor(kxx.T).T).sum(axis=1)

        return mu, np.array(var)

    @property
    def nll(self):
        assert self.factor is not None, "The gp must be fit using fit() before querying the neg log likelihood."
        return self._nll
