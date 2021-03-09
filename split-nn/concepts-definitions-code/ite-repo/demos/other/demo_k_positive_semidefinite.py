#!/usr/bin/env python3

""" Demo for the positive semi-definiteness of the Gram matrix.

In the demo, normal and uniform random variables are considered.

"""

from scipy.linalg import eigh
from numpy import dot
from numpy.random import rand, randn, multivariate_normal
import matplotlib.pyplot as plt
from ite.cost.x_factory import co_factory


def main():
    # parameters:
    distr = 'uniform'  # possibilities: 'uniform', 'normal'
    dim = 2  # dimension of the distribution
    num_of_distributions = 5
    # each distribution is represented by num_of_samples samples:
    num_of_samples = 500

    # kernel used to evaluate the distributions:
    cost_name = 'BKExpected'
    # cost_name = 'BKProbProd_KnnK'
    # cost_name = 'MKExpJR1_HR'
    # cost_name = 'MKExpJR2_DJR'
    # cost_name = 'MKExpJS_DJS'
    # cost_name = 'MKExpJT1_HT'
    # cost_name = 'MKExpJT2_DJT'
    # cost_name = 'MKJS_DJS'
    # cost_name = 'MKJT_HT'

    # initialization:
    co = co_factory(cost_name, mult=True)
    ys = list()

    # generate samples from the distributions (ys):
    for n in range(num_of_distributions):
        # generate samples from the n^th distribution (y):
        if distr == 'uniform':
            a, b = -rand(dim), rand(dim)  # a,b
            # (random) linear transformation applied to the data (r x
            # U[a,b]):
            r = randn(dim, dim)
            y = dot(rand(num_of_samples, dim) * (b-a).T + a.T, r)
        elif distr == 'normal':
            m = rand(dim)  # mean
            # cov:
            l = rand(dim, dim)
            c = dot(l, l.T)  # cov

            # generate samples (yy~N(m,c)):
            y = multivariate_normal(m, c, num_of_samples)

        ys.append(y)

    # Gram matrix and its minimal eigenvalue:
    g = co.gram_matrix(ys)
    eigenvalues = eigh(g)[0]  # eigenvalues: reals, in increasing order
    min_eigenvalue = eigenvalues[0]

    # plot:
    plt.plot(range(1, num_of_distributions+1), eigenvalues)
    plt.xlabel('Index of the sorted eigenvalues: i')
    plt.ylabel('Eigenvalues of the (estimated) Gram matrix')
    plt.title("Minimal eigenvalue: " + str(round(min_eigenvalue, 2)))
    plt.show()


if __name__ == "__main__":
    main()
