#!/usr/bin/env python3

""" Demo for Renyi mutual information estimators.

Analytical vs estimated value is illustrated for normal random variables.

"""

from numpy.random import rand, multivariate_normal
from numpy import arange, zeros, dot, ones
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_i_renyi


def main():
    # parameters:
    alpha = 0.7  # parameter of Renyi mutual information, \ne 1
    dim = 2  # >=2; dimension of the distribution
    num_of_samples_v = arange(100, 10*1000+1, 500)

    cost_name = 'MIRenyi_DR'
    # cost_name = 'MIRenyi_HR'
    
    # initialization:
    distr = 'normal'  # distribution; fixed    
    ds = ones(dim, dtype='int')  # dimensions of the 'subspaces'
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True, alpha=alpha)  # cost object
    # vector of estimated mutual information values:
    i_hat_v = zeros(length)

    # distr, dim -> samples (y), distribution parameters (par), analytical 
    # value (i):
    if distr == 'normal':
        # mean (m), covariance matrix (c):
        m = rand(dim) 
        l = rand(dim, dim)
        c = dot(l, l.T)
        
        # generate samples (y~N(m,c)): 
        y = multivariate_normal(m, c, num_of_samples_max)
           
        par = {"cov": c} 
    else:
        raise Exception('Distribution=?')
    
    i = analytical_value_i_renyi(distr, alpha, par)
        
    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        i_hat_v[tk] = co.estimation(y[0:num_of_samples], ds)  # broadcast
        print("tk={0}/{1}".format(tk+1, length))
        
    # plot:    
    plt.plot(num_of_samples_v, i_hat_v, num_of_samples_v, ones(length)*i)
    plt.xlabel('Number of samples')
    plt.ylabel('Renyi mutual information')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
