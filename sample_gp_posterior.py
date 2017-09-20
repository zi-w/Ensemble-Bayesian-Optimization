import numpy as np
from gp import DenseKernelGP
import helper
import matplotlib.pyplot as plt
from representation import TileCoding, DenseKernel
import representation as rp

def get_kernel(z, k, x_range, nlayers=1000, hashing_mem=None):
  indices = []
  ntiles = []
  hashing = None
  all_cat = np.unique(z)
  if hashing_mem:
  	hashing = [rp.UNH(hashing_mem) for _ in xrange(len(all_cat))]
  for a in all_cat:
    inds = helper.find(z==a)
    indices.append(inds)
    ntiles.append(k[inds])
  phi = TileCoding( input_indices = indices,
  # ntiles = input dim x number of layers x tilings
  ntiles = ntiles,
  ntilings= [nlayers]*len(indices),
  hashing = hashing,
  state_range = x_range,
  rnd_stream = np.random,
  bias_term = False
  )
  #start = time.time()
  # sparsephi = IndexToBinarySparse(phi, normalize=True)
  # gp = SparseKernelGP(self.X, self.y, sigma = 0.1, phi = sparsephi)
  # gp = SparseFeatureGP(self.X, self.y, sigma = 0.1, phi = sparsephi) too slow
  densekern = DenseKernel(phi, normalize=True)
  return densekern
def sample_gp_posterior(dx, z, k, n, x_range):
	kern = get_kernel(z, k, x_range)
	X = np.random.uniform(x_range[0], x_range[1], (n, dx))
	kxx = kern(X) + 0.01*np.eye(X.shape[0])
	y = np.random.multivariate_normal(np.zeros(n), kxx).T
	gp = DenseKernelGP(X, y, sigma=0.1, kern=kern)
	gp.fit()
	f = lambda x: gp.predict(x)[0] if len(x.shape)>1 else gp.predict(x)[0][0]

	return f


def test():
  ## test 

  # get function
  dx = 10
  n = 100
  alpha = [1.0]*dx
  theta = np.random.dirichlet(alpha)
  z = helper.sample_multinomial(theta, dx, 2)
  print 'z=',z
  k = np.array([3]*dx)
  print 'k=',k
  x_range = np.matlib.repmat([[0],[1]], 1, dx)
  f = sample_gp_posterior(dx, z, k, n, x_range)

  # plot
  m = 1000
  tx = np.linspace(0,1., m)[:,None]
  fx = np.random.uniform(x_range[0], x_range[1])
  print fx
  rx = np.matlib.repmat(fx, m, 1)
  rx[:,0:1] = tx
  ty = f(rx)
  plt.plot(tx, ty)
  plt.show()