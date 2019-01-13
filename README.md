# Ensemble-Bayesian-Optimization
This is the code repository associated with the paper [_Batched Large-scale Bayesian Optimization in High-dimensional Spaces_](https://arxiv.org/pdf/1706.01445.pdf). We propose a new batch/distributed Bayesian optimization technique called **Ensemble Bayesian Optimization**, which unprecedentedly scales up Bayesian optimization both in terms of input dimensions and observation size. Please refer to the paper if you need more details on the algorithm.

## Requirements 
We tested our code with Python 2.7 on Ubuntu 14.04 LTS (64-bit).

See configs/start_commands for required packages.

## Implementations of Gaussian processes
We implemented 4 versions of Gaussian processes in gp_tools/gp.py, which can be used without the BO functionalities.

* DenseKernelGP: a GP which has a dense kernel matrix.
* SparseKernelGP: a GP which has a sparse kernel matrix.
* SparseFeatureGP: a GP whose kernel is defined by the inner product of two sparse feature vectors.
* DenseFeatureGP: a GP whose kernel is defined by the inner product of two dense feature vectors.

## Example
test_ebo.m gives an example of running EBO on a 2 dimensional function with visualizations. 

To run EBO on expensive functions using Microsoft Azure, set the account information in configuration.cfg and the desired pool information in ebo.cfg. Then in the options, set "useAzure" to be True and "func_cheap" to be False.

## Test functions
We provide 3 examples of black-box functions:

* test_functions/simple_functions.py: functions sampled from a GP.
* test_functions/push_function.py: a reward function for two robots pushing two objects. 
* test_functions/rover_function.py: a reward function for the trajectory of a 2D rover.

## Caveats on the hyperparameters of EBO
 From more extensive experiments we found that EBO is not be robust to the hyperparameters of the Mondrian trees including the size of each leaf (min_leaf_size), number of leaves (max_n_leaves), selections per partition (n_bo), etc. Principled ways of setting those parameters remain a future work. 
