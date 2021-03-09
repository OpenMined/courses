#!/usr/bin/env python3

""" Demo for incomplete Cholesky factorization based Gram matrix
approximation.

"""

from numpy import dot, zeros, power, log10
from numpy.random import randn
from numpy.linalg import norm
import matplotlib.pyplot as plt
from ite.cost.x_kernel import Kernel


def main():
    # parameters:
    num_of_samples = 1000
    dim = 2
    eta_v = power(10.0, range(-8, 0))  # 10^{-8}, 10^{-7}, ..., 10^{-1}

    # define a kernel:
    k = Kernel({'name': 'RBF', 'sigma': 1})
    # k = Kernel({'name': 'exponential', 'sigma': 1})
    # k = Kernel({'name': 'Cauchy', 'sigma': 1})
    # k = Kernel({'name': 'student', 'd': 1})
    # k = Kernel({'name': 'Matern3p2', 'l': 1})
    # k = Kernel({'name': 'Matern5p2', 'l': 1})
    # k = Kernel({'name': 'polynomial', 'exponent': 2, 'c': 1})
    # k = Kernel({'name': 'ratquadr', 'c': 1})
    # k = Kernel({'name': 'invmquadr', 'c': 1})

    # print(k)  # print the picked kernel

    # initialization:
    length = len(eta_v)
    error_v = zeros(length)  # vector of estimated entropy values

    # define a dataset:
    y = randn(num_of_samples, dim)

    # true Gram matrix:
    gram_matrix = k.gram_matrix1(y)

    for (tk, eta) in enumerate(eta_v):
        tol = eta * num_of_samples
        r = k.ichol(y, tol)
        gram_matrix_hat = dot(r, r.T)
        dim_red = r.shape[1]
        error_v[tk] = norm(gram_matrix - gram_matrix_hat) / \
                      norm(gram_matrix)
        print("tk={0}/{1}, log10(eta):{2}, dimension (reduced/original): "
              "{3}/{4}".format(tk + 1, length, log10(eta), dim_red,
                               num_of_samples))

    # plot:
    plt.plot(eta_v, error_v)
    plt.xlabel('Tolerance (eta)')
    plt.ylabel('Relative error in the incomplete Cholesky decomposition')
    plt.xscale('log')
    plt.show()


if __name__ == "__main__":
    main()
