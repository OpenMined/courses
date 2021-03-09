#!/usr/bin/env python3

""" Demo for Shannon entropy estimators.

Analytical vs estimated value is illustrated for normal and uniform
random variables.

"""    

from numpy.random import rand, multivariate_normal
from numpy import arange, zeros, dot, ones
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_h_shannon


def main():
    # parameters:
    distr = 'normal'  # distribution: 'uniform', 'normal'
    dim = 2  # dimension of the distribution
    num_of_samples_v = arange(1000, 50*1000+1, 1000)
    
    cost_name = 'BHShannon_KnnK'  # dim >= 1
    # cost_name = 'BHShannon_SpacingV'  # dim = 1
    # cost_name = 'BHShannon_MaxEnt1'  # dim = 1; approximation around N
    # cost_name = 'BHShannon_MaxEnt2'  # dim = 1; approximation around N
    # cost_name = 'MHShannon_DKLN'   # dim >= 1
    # cost_name = 'MHShannon_DKLU'  # dim >= 1
    
    # initialization:    
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True)  # cost object
    h_hat_v = zeros(length)  # vector of estimated entropy values
    
    # distr, dim -> samples (y), distribution parameters (par), analytical 
    # value (h):
    if distr == 'uniform':
        # U[a,b], (random) linear transformation applied to the data (l):
        a = -rand(1, dim)
        b = rand(1, dim)  # guaranteed that a<=b (coordinate-wise)
        l = rand(dim, dim)
        y = dot(rand(num_of_samples_max, dim)*(b-a) + a, l.T)  # lxU[a,b]
        
        par = {"a": a, "b": b, "l": l}
    elif distr == 'normal':
        # mean (m), covariance matrix (c):
        m = rand(dim)
        l = rand(dim, dim)
        c = dot(l, l.T)

        # generate samples (y~N(m,c)):
        y = multivariate_normal(m, c, num_of_samples_max)

        par = {"cov": c}
    else:
        raise Exception('Distribution=?')
                
    h = analytical_value_h_shannon(distr, par)
    
    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        h_hat_v[tk] = co.estimation(y[0:num_of_samples])  # broadcast
        print("tk={0}/{1}".format(tk+1, length))
        
    # plot:    
    plt.plot(num_of_samples_v, h_hat_v, num_of_samples_v, ones(length)*h)
    plt.xlabel('Number of samples')
    plt.ylabel('Shannon entropy')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
