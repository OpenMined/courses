#!/usr/bin/env python3

""" Demo for (Shannon) mutual information estimators.

Analytical vs estimated value is illustrated for normal random variables.

"""

from numpy.random import rand, multivariate_normal
from numpy import array, arange, zeros, dot, ones, sum
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_i_shannon


def main():
    # parameters:
    ds = array([4, 1])  # subspace dimensions: ds[0], ..., ds[M-1]
    num_of_samples_v = arange(1000, 20*1000+1, 1000)

    cost_name = 'MIShannon_DKL'  # d_m >= 1, M >= 2
    # cost_name = 'MIShannon_HS'  # d_m >= 1, M >= 2
   
    # initialization:
    distr = 'normal'  # distribution; fixed    
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True)  # cost object
    # vector of estimated mutual information values:
    i_hat_v = zeros(length)

    # distr, ds -> samples (y), distribution parameters (par), analytical 
    # value (i):
    if distr == 'normal':
        dim = sum(ds)  # dimension of the joint distribution
        
        # mean (m), covariance matrix (c):
        m = rand(dim) 
        l = rand(dim, dim)
        c = dot(l, l.T)
        
        # generate samples (y~N(m,c)): 
        y = multivariate_normal(m, c, num_of_samples_max)
        
        par = {"ds": ds, "cov": c} 
    else:
        raise Exception('Distribution=?')
    
    i = analytical_value_i_shannon(distr, par)
        
    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        i_hat_v[tk] = co.estimation(y[0:num_of_samples], ds)  # broadcast
        print("tk={0}/{1}".format(tk+1, length))
      
    # plot:    
    plt.plot(num_of_samples_v, i_hat_v, num_of_samples_v, ones(length)*i)
    plt.xlabel('Number of samples')
    plt.ylabel('Shannon mutual information')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
