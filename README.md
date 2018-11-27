# Ensemble-Bayesian-Optimization
This is the code repository associated with the paper [_Ensemble Bayesian Optimization_](https://arxiv.org/pdf/1706.01445.pdf). We propose a new batch/distributed Bayesian optimization technique called **Ensemble Bayesian Optimization**, which unprecedentedly scales up Bayesian optimization both in terms of input dimensions and observation size. Please refer to the paper if you need more details on the algorithm.

## Requirements 
See start_commands.

## Example
test_ebo.m gives an example of running EBO on a 2 dimensional function with visualizations. 

To run EBO on expensive functions using Microsoft Azure, set the account information in configuration.cfg and the desired pool information in ebo.cfg. Then in the options, set "useAzure" to be True and "func_cheap" to be False.


## Caveats on the hyperparameters of EBO
 From more extensive experiments we found that EBO is not be robust to the hyperparameters of the Mondrian trees including the size of each leaf (min_leaf_size), number of leaves (max_n_leaves), selections per partition (n_bo), etc. Principled ways of setting those parameters remain a future work. 