#!/usr/bin/env python3

""" Demo for estimators of the expected kernel.

Analytical vs estimated value is illustrated for normal random variables.

"""

from numpy.random import rand, multivariate_normal
from scipy import arange, zeros, dot, ones
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_k_expected
from ite.cost.x_kernel import Kernel


def main():
    # parameters:
    dim = 1  # dimension of the distribution
    num_of_samples_v = arange(100, 5*1000+1, 100)  # number of samples
    cost_name = 'BKExpected'  # dim >= 1

    # initialization:
    distr = 'normal'  # fixed
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)

    # RBF kernel (sigma = std / bandwith parameter):
    kernel = Kernel({'name': 'RBF', 'sigma': 1})
    # polynomial kernel (quadratic / cubic; c = offset parameter = 1):
    # kernel = Kernel({'name': 'polynomial', 'exponent': 2, 'c': 1})
    # kernel = Kernel({'name': 'polynomial', 'exponent': 3, 'c': 1})

    co = co_factory(cost_name, mult=True, kernel = kernel) # cost object

    k_hat_v = zeros(length)  # vector of estimated kernel values

    # distr, dim -> samples (y1,y2), distribution parameters (par1,par2), 
    # analytical value (k):
    if distr == 'normal':
        # mean (m1,m2):
        m1 = rand(dim)
        m2 = rand(dim)
        
        # (random) linear transformation applied to the data (l1,l2) -> 
        # covariance matrix (c1,c2):
        l2 = rand(dim, dim)
        l1 = rand(dim, dim)
        c1 = dot(l1, l1.T)
        c2 = dot(l2, l2.T)

        # generate samples (y1~N(m1,c1), y2~N(m2,c2)):
        y1 = multivariate_normal(m1, c1, num_of_samples_max)
        y2 = multivariate_normal(m2, c2, num_of_samples_max)

        par1 = {"mean": m1, "cov": c1}
        par2 = {"mean": m2, "cov": c2}
    else:
        raise Exception('Distribution=?')        
        
    k = analytical_value_k_expected(distr, distr, co.kernel, par1, par2)
    
    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        k_hat_v[tk] = co.estimation(y1[0:num_of_samples],
                                    y2[0:num_of_samples])  # broadcast
        print("tk={0}/{1}".format(tk+1, length))
 
    # plot:    
    plt.plot(num_of_samples_v, k_hat_v, num_of_samples_v, ones(length)*k)
    plt.xlabel('Number of samples')
    plt.ylabel('Expected kernel')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
