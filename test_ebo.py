import numpy as np
from ebo import ebo
from test_functions import plot_f, sampled_gp_func, sample_z
import time
import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)

##################### define test function ######################
dx = 2
z = sample_z(dx)
k = np.array([10]*dx)
x_range = np.matlib.repmat([[0.],[1.]], 1, dx)
sigma = 0.01
n = 100
f = sampled_gp_func(x_range, dx, z, k, n, sigma)
plot_f(f)
##################################################################

# run ebo
options = {'x_range':x_range, 
           'dx':x_range.shape[1],
           'max_value':f.fmax, 
           'T':10, # number of iterations
           'B':20, # number of candidates to be evaluated
           'dim_limit':3, 
           'isplot':1,
           'z':None, 'k':None,
           'alpha':1.,
           'beta':np.array([5.,2.]),
           'opt_n':1000, # points randomly sampled to start continuous optimization of acfun
           'pid':'test3',
           'datadir':'tmp_data3/',
           'gibbs_iter':10,
           'useAzure':False,
           'n_add':None, # this should always be None. it makes dim_limit complicated if not None.
           'nlayers': 100,
           'gp_type':'l1', # other choices are l1, sk, sf, dk, df
           'gp_sigma':0.1, # noise standard deviation
           'n_bo':1, # min number of points selected for each partition
           'n_bo_top_percent': 0.7, # percentage of top in bo selections
           'n_top':10, # how many points to look ahead when doing choose Xnew
           'min_leaf_size':20,
           'max_n_leaves':50,
           'func_cheap':True, # if func cheap, we do not use Azure to test functions
           'thresAzure':1, # if > thresAzure, we use Azure
           'save_file_name': 'plotdata/7.pk',
           'ziw':True
           }

e = ebo(f, options)
start = time.time()
e.run()

print 'elapsed time: ', time.time()-start