#!/usr/bin/env python3

""" Demo Jensen-Renyi divergence estimators.

Analytical vs estimated value is illustrated for spherical normal random
variables.

"""

from numpy.random import rand, multivariate_normal, randn
from numpy import arange, zeros, ones, array, eye
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_d_jensen_renyi


def main():
    # parameters:
    dim = 2  # dimension of the distribution
    w = array([1/3, 2/3])  # weight in the Jensen-Renyi divergence
    num_of_samples_v = arange(100, 12*1000+1, 500)
    cost_name = 'MDJR_HR'  # dim >= 1
    
    # initialization:
    alpha = 2  # parameter of the Jensen-Renyi divergence, \ne 1; fixed    
    distr = 'normal'  # fixed    
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True, alpha=alpha, w=w)  # cost object
    d_hat_v = zeros(length)  # vector of estimated divergence values

    # distr, dim -> samples (y1,y2), distribution parameters (par1,par2), 
    # analytical value (d):
    if distr == 'normal':
        # generate samples (y1,y2); y1~N(m1,s1^2xI), y2~N(m2,s2^2xI):
        m1, s1 = randn(dim), rand(1)
        m2, s2 = randn(dim), rand(1)
        y1 = multivariate_normal(m1, s1**2 * eye(dim), num_of_samples_max)
        y2 = multivariate_normal(m2, s2**2 * eye(dim), num_of_samples_max)
      
        par1 = {"mean": m1, "std": s1}
        par2 = {"mean": m2, "std": s2}
    else:
        raise Exception('Distribution=?')        
    
    d = analytical_value_d_jensen_renyi(distr, distr, w, par1, par2)
    
    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        d_hat_v[tk] = co.estimation(y1[0:num_of_samples],
                                    y2[0:num_of_samples])  # broadcast
        print("tk={0}/{1}".format(tk+1, length))
    
    # plot:    
    plt.plot(num_of_samples_v, d_hat_v, num_of_samples_v, ones(length)*d)
    plt.xlabel('Number of samples')
    plt.ylabel('Jensen-Renyi divergence')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
