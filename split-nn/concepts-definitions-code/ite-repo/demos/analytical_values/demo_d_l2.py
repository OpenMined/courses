#!/usr/bin/env python3

""" Demo for L2 divergence estimators.

Analytical vs estimated value is illustrated for uniform random variables.

"""

from numpy.random import rand
from numpy import arange, zeros, ones
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_d_l2


def main():
    # parameters:
    dim = 1  # dimension of the distribution
    num_of_samples_v = arange(1000, 20*1000+1, 1000)
    cost_name = 'BDL2_KnnK'  # dim >= 1
    
    # initialization:
    distr = 'uniform'  # fixed
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True)  # cost object
    d_hat_v = zeros(length)  # vector of estimated divergence values

    # distr, dim -> generate samples (y1>>y2), analytical value (d):
    if distr == 'uniform':
        a = 3 * rand(dim)
        b = a * rand(dim)
            
        y1 = rand(num_of_samples_max, dim) * a  # U[0,a]
        y2 = rand(num_of_samples_max, dim) * b
        # Note: U[0,b], b<=a (coordinate-wise)=> y2<<y1
        
        par1 = {"a": a}
        par2 = {"a": b}        
    else:
        raise Exception('Distribution=?')        
    
    d = analytical_value_d_l2(distr, distr, par1, par2)

    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        d_hat_v[tk] = co.estimation(y1[0:num_of_samples],
                                    y2[0:num_of_samples])  # broadcast
        print("tk={0}/{1}".format(tk+1, length))
    
    # plot:    
    plt.plot(num_of_samples_v, d_hat_v, num_of_samples_v, ones(length)*d)
    plt.xlabel('Number of samples')
    plt.ylabel('L2 divergence')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
