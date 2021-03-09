#!/usr/bin/env python3

""" Demo for chi^2 divergence estimators.

Analytical vs estimated value is illustrated for uniform and spherical
normal random variables.

"""

from numpy.random import rand, multivariate_normal
from numpy import arange, zeros, ones, eye
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_d_chi_square


def main():
    # parameters:
    distr = 'normalI'  # 'uniform', 'normalI' (isotropic normal, Id cov.) 
    dim = 1  # dimension of the distribution
    num_of_samples_v = arange(1000, 50*1000+1, 1000)
    cost_name = 'BDChi2_KnnK'  # dim >= 1
    
    # initialization:
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True)  # cost object
    d_hat_v = zeros(length)  # vector of estimated divergence values

    # distr, dim -> samples (y1<<y2), distribution parameters (par1,par2), 
    # analytical value (d):
    if distr == 'uniform':
        b = 3 * rand(dim)
        a = b * rand(dim)
            
        y1 = rand(num_of_samples_max, dim) * a  # U[0,a]
        y2 = rand(num_of_samples_max, dim) * b
        # U[0,b], a<=b (coordinate-wise) => y1<<y2
        
        par1 = {"a": a}
        par2 = {"a": b}        
    elif distr == 'normalI':
        # mean (m1,m2):
        m1 = 2 * rand(dim)
        m2 = 2 * rand(dim)
        
        # generate samples (y1~N(m1,I), y2~N(m2,I)):
        y1 = multivariate_normal(m1, eye(dim), num_of_samples_max)
        y2 = multivariate_normal(m2, eye(dim), num_of_samples_max)
        
        par1 = {"mean": m1}
        par2 = {"mean": m2}
    else:
        raise Exception('Distribution=?')        
    
    d = analytical_value_d_chi_square(distr, distr, par1, par2)

    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        d_hat_v[tk] = co.estimation(y1[0:num_of_samples],
                                    y2[0:num_of_samples])  # broadcast
        print("tk={0}/{1}".format(tk+1, length))
    
    # plot:    
    plt.plot(num_of_samples_v, d_hat_v, num_of_samples_v, ones(length)*d)
    plt.xlabel('Number of samples')
    plt.ylabel('Chi square divergence')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
