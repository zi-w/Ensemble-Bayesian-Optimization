This code base is under construction. It already has some basic functionals to repeat our experiments. However, we later found from more extensive experiments that our approach is not robust to a number of hyper-parameters of the Mondrian trees. I will not update this code base anymore unless someone really wants to make use of it.

# Ensemble-Bayesian-Optimization
This is the code repository associated with the paper [_Ensemble Bayesian Optimization_](https://arxiv.org/pdf/1706.01445.pdf). We propose a new batch/distributed Bayesian optimization technique called **Ensemble Bayesian Optimization**, which unprecedentedly scales up Bayesian optimization both in terms of input dimensions and observation size. Please refer to the paper if you need more details on the algorithm.

## Example
test_ebo.m gives an example of running EBO on a 2 dimensional function with visualizations. 
