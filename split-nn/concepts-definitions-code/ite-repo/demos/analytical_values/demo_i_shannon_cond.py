#!/usr/bin/env python3

""" Demo for conditional Shannon mutual information estimators.

Analytical vs estimated value is illustrated for normal random variables.

"""

from numpy import arange, zeros, dot, ones, array, sum
from numpy.random import rand, multivariate_normal
import matplotlib.pyplot as plt

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_cond_i_shannon


def main():
    # parameters:
    ds = array([1, 1, 1])  # dimensions of the subspaces, m=1,...,M+1, M>=2
    num_of_samples_v = arange(1000, 20 * 1000 + 1, 1000)
    cost_name = 'BcondIShannon_HShannon'  # dm >= 1, M >= 2

    # initialization:
    distr = 'normal'  # fixed
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True)  # cost object
    # vector of estimated conditional mutual information values:
    cond_i_hat_v = zeros(length)

    # distr, ds -> samples (y), distribution parameters (par), analytical
    # value (cond_i):
    if distr == 'normal':
        dim = sum(ds)
        # mean (m), covariance matrix (c):
        m, l = rand(dim), rand(dim, dim)
        c = dot(l, l.T)

        # generate samples (y~N(m,c)):
        y = multivariate_normal(m, c, num_of_samples_max)

        par = {"cov": c, "ds": ds}
    else:
        raise Exception('Distribution=?')

    cond_i = analytical_value_cond_i_shannon(distr, par)

    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        # broadcasting:
        cond_i_hat_v[tk] = co.estimation(y[:num_of_samples], ds)
        print("tk={0}/{1}".format(tk + 1, length))

    # plot:
    plt.plot(num_of_samples_v, cond_i_hat_v,
             num_of_samples_v, ones(length) * cond_i)
    plt.xlabel('Number of samples')
    plt.ylabel('Conditional Shannon mutual information')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
