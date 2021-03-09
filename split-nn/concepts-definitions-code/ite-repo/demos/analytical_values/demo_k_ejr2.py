#!/usr/bin/env python3

""" Demo for exponentiated Jensen-Renyi kernel-2 estimators.

Analytical vs estimated value is illustrated for spherical normal random
variables.

"""

from numpy import eye
from numpy.random import rand, multivariate_normal, randn
from scipy import arange, zeros, ones
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_k_ejr2


def main():
    # parameters:
    dim = 1  # dimension of the distribution
    num_of_samples_v = arange(100, 12*1000+1, 500)
    u = 1  # >0, parameter of the Jensen-Renyi kernel
    cost_name = 'MKExpJR2_DJR'  # dim >= 1

    # initialization:
    alpha = 2
    # fixed; parameter of the Jensen-Renyi kernel; for alpha = 2 we have
    # explicit formula for the Jensen-Renyi divergence, and hence for the
    # Jensen-Renyi kernel(-2).

    distr = 'normal'  # fixed
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True, alpha=alpha, u=u)  # cost object
    
    k_hat_v = zeros(length)  # vector of estimated kernel values

    # distr, dim -> samples (y1,y2), distribution parameters (par1,par2), 
    # analytical value (k):
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
        
    k = analytical_value_k_ejr2(distr, distr, u, par1, par2)
    
    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        k_hat_v[tk] = co.estimation(y1[0:num_of_samples],
                                    y2[0:num_of_samples])  # broadcast
        print("tk={0}/{1}".format(tk+1, length))
        
    # plot:    
    plt.plot(num_of_samples_v, k_hat_v, num_of_samples_v, ones(length)*k)
    plt.xlabel('Number of samples')
    plt.ylabel('Exponentiated Jensen-Renyi kernel-2')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
