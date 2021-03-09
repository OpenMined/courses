#!/usr/bin/env python3

""" Demo for 'A(y^1,...,y^m)=0 if y^m-s are independent'.

In the demo, normal and uniform random variables are considered.

"""


from numpy import ones, arange, zeros
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from ite.cost.x_factory import co_factory


def main():
    # parameters:
    distr = 'uniform'  # possibilities: 'uniform', 'normal' 
    m = 2  # number of components
    num_of_samples_v = arange(1000, 30*1000+1, 1000)
    cost_name = 'BASpearman1'  # m >= 2
    # cost_name = 'BASpearman2'  # m >= 2
    # cost_name = 'BASpearman3'  # m >= 2
    # cost_name = 'BASpearman4'  # m >= 2
    # cost_name = 'BASpearmanCondLT'  # m >= 2
    # cost_name = 'BASpearmanCondUT'  # m >= 2
    # cost_name = 'BABlomqvist'  # m >= 2
    # cost_name = 'MASpearmanUT'  # m >= 2
    # cost_name = 'MASpearmanLT'  # m >= 2
            
    # initialization:
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True)  # cost object
    a_hat_v = zeros(length)  # vector to store the association values
    ds = ones(m, dtype='int')
    
    # distr -> samples (y); analytical value (a):
    if distr == 'uniform':
        y = rand(num_of_samples_max, m)
    elif distr == 'normal':
        y = randn(num_of_samples_max, m)
    else:
        raise Exception('Distribution=?')

    a = 0

    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        a_hat_v[tk] = co.estimation(y[0:num_of_samples], ds)  # broadcast
        print("tk={0}/{1}".format(tk+1, length))

    # plot:    
    plt.plot(num_of_samples_v, a_hat_v, num_of_samples_v, ones(length)*a)
    plt.xlabel('Number of samples')
    plt.ylabel('Association')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
