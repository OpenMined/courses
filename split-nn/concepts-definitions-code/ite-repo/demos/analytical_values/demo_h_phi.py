#!/usr/bin/env python3

""" Demo for Phi entropy estimators.

Analytical vs estimated value is illustrated for uniform random variables.

"""
    
from numpy.random import rand
from numpy import arange, zeros, ones
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_h_phi


def main():
    # parameters:
    c = 2  # >=1; c is also used in the analytical expression: 'h = ...'
    phi = lambda x: x**c  # phi (in the Phi-entropy)
    num_of_samples_v = arange(1000, 100*1000+1, 1000)
    cost_name = 'BHPhi_Spacing'  # dim = 1
    
    # initialization:    
    distr = 'uniform'  # fixed    
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True, phi=phi)  # cost object
    h_hat_v = zeros(length)  # vector of estimated entropy values
    
    # distr -> samples (y), distribution parameters (par), analytical 
    # value (h):
    if distr == 'uniform':  # U[a,b]:
        # a, b:
        a = rand(1)
        b = a + 4 * rand(1)  # guaranteed that a<=b
         
        # generate samples:
        y = (b-a)*rand(num_of_samples_max, 1) + a
        
        par = {"a": a, "b": b}
    else:
        raise Exception('Distribution=?')
        
    h = analytical_value_h_phi(distr, par, c)

    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        h_hat_v[tk] = co.estimation(y[0:num_of_samples])  # broadcast
        print("tk={0}/{1}".format(tk+1, length))
        
    # plot:
    plt.plot(num_of_samples_v, h_hat_v, num_of_samples_v, ones(length)*h)
    plt.xlabel('Number of samples')
    plt.ylabel('Phi entropy')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
