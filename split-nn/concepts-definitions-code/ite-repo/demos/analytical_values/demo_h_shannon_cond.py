#!/usr/bin/env python3

""" Demo for conditional Shannon entropy estimators.

Analytical vs estimated value is illustrated for normal random variables.

"""

from numpy import arange, zeros, dot, ones
from numpy.random import rand, multivariate_normal
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_cond_h_shannon


def main():
    # parameters:
    dim1 = 1  # dimension of y1
    dim2 = 2  # dimension of y2
    num_of_samples_v = arange(1000, 30 * 1000 + 1, 1000)
    cost_name = 'BcondHShannon_HShannon'  # dim1 >= 1, dim2 >= 1

    # initialization:
    distr = 'normal'  # fixed
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True)  # cost object
    # vector of estimated conditional entropy values:
    cond_h_hat_v = zeros(length)
    dim = dim1 + dim2

    # distr, dim1 -> samples (y), distribution parameters (par),
    # analytical value (cond_h):
    if distr == 'normal':
        # mean (m), covariance matrix (c):
        m, l = rand(dim), rand(dim, dim)
        c = dot(l, l.T)

        # generate samples (y~N(m,c)):
        y = multivariate_normal(m, c, num_of_samples_max)

        par = {"dim1": dim1, "cov": c}
    else:
        raise Exception('Distribution=?')

    cond_h = analytical_value_cond_h_shannon(distr, par)

    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        # broadcasting:
        cond_h_hat_v[tk] = co.estimation(y[:num_of_samples], dim1)
        print("tk={0}/{1}".format(tk + 1, length))

    # plot:
    plt.plot(num_of_samples_v, cond_h_hat_v,
             num_of_samples_v, ones(length) * cond_h)
    plt.xlabel('Number of samples')
    plt.ylabel('Conditional Shannon entropy')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
