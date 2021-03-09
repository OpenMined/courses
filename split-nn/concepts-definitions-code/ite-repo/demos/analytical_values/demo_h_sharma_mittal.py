#!/usr/bin/env python3

""" Demo for Sharma-Mittal entropy estimators.

Analytical vs estimated value is illustrated for normal random variables.

"""

from numpy.random import rand, multivariate_normal
from numpy import arange, zeros, dot, ones
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_h_sharma_mittal


def main():
    # parameters:
    dim = 1  # dimension of the distribution
    num_of_samples_v = arange(1000, 30*1000+1, 1000)
    alpha = 0.8  # parameter of the Sharma-Mittal entropy; alpha \ne 1
    beta = 0.6   # parameter of the Sharma-Mittal entropy; beta \ne 1
    cost_name = 'BHSharmaMittal_KnnK'  # dim >= 1
    
    # initialization:    
    distr = 'normal'  # fixed
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    # cost object:
    co = co_factory(cost_name, mult=True, alpha=alpha, beta=beta)
    h_hat_v = zeros(length)  # vector of estimated entropy values
    
    # distr, dim -> samples (y), distribution parameters (par), analytical 
    # value (h):
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
                
    h = analytical_value_h_sharma_mittal(distr, alpha, beta, par)
    
    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        h_hat_v[tk] = co.estimation(y[0:num_of_samples])  # broadcast
        print("tk={0}/{1}".format(tk+1, length))
        
    # plot:    
    plt.plot(num_of_samples_v, h_hat_v, num_of_samples_v, ones(length)*h)
    plt.xlabel('Number of samples')
    plt.ylabel('Sharma-Mittal entropy')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
