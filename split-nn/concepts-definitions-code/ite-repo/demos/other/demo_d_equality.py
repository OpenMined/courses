#!/usr/bin/env python3

""" Demo for 'D(p,q)=0 if p=q'.

In the demo, normal and uniform random variables are considered.

"""

from numpy import dot, eye, zeros, arange, ones
from numpy.random import rand, randn, multivariate_normal
import matplotlib.pyplot as plt
from ite.cost.x_factory import co_factory


def main():
    # parameters:
    distr = 'normal'  # possibilities: 'normal', 'uniform', 'normalI'
    dim = 2  # dimension of the distribution
    num_of_samples_v = arange(1000, 30 * 1000 + 1, 1000)
    # for slower estimators:
    # num_of_samples_v = arange(500, 5 * 1000 + 1, 500)

    cost_name = 'BDKL_KnnK'            # dim >= 1
    # cost_name = 'BDEnergyDist'         # dim >= 1, slower
    # cost_name = 'BDBhattacharyya_KnnK' # dim >= 1
    # cost_name = 'BDBregman_KnnK'       # dim >= 1
    # cost_name = 'BDChi2_KnnK'          # dim >= 1
    # cost_name = 'BDHellinger_KnnK'     # dim >= 1
    # cost_name = 'BDKL_KnnKiTi'         # dim >= 1
    # cost_name = 'BDL2_KnnK'            # dim >= 1
    # cost_name = 'BDRenyi_KnnK'         # dim >= 1
    # cost_name = 'BDTsallis_KnnK'       # dim >= 1
    # cost_name = 'BDSharmaMittal_KnnK'  # dim >= 1
    # cost_name = 'BDSymBregman_KnnK'    # dim >= 1
    # cost_name = 'BDMMD_UStat'          # dim >= 1, slower
    # cost_name = 'BDMMD_UStat_IChol'    # dim >= 1, semi-slow
    # cost_name = 'BDMMD_VStat'          # dim >= 1, slower
    # cost_name = 'BDMMD_VStat_IChol'    # dim >= 1, semi-slow
    # cost_name = 'BDMMD_Online'         # dim >= 1
    # cost_name = 'MDBlockMMD'           # dim >= 1
    # cost_name = 'MDEnergyDist_DMMD'    # dim >= 1
    # cost_name = 'MDf_DChi2'            # dim >= 1
    # cost_name = 'MDJDist_DKL'          # dim >= 1
    # cost_name = 'MDJR_HR'              # dim >= 1
    # cost_name = 'MDJT_HT'              # dim >= 1
    # cost_name = 'MDJS_HS'              # dim >= 1
    # cost_name = 'MDK_DKL'              # dim >= 1
    # cost_name = 'MDL_DKL'              # dim >= 1
    # cost_name = 'MDSymBregman_DB'      # dim >= 1
    # cost_name = 'MDKL_HSCE'            # dim >= 1

    # initialization:
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True)  # cost object
    # vector to store the estimated divergence values:
    d_hat_v = zeros(length)

    # distr -> samples (y); analytical value (d):
    if distr == 'uniform':
        a = randn(dim, dim)  # (random) linear transformation
        y1 = dot(rand(num_of_samples_max, dim), a)
        y2 = dot(rand(num_of_samples_max, dim), a)
    elif distr == 'normal':
        m = rand(dim)  # mean
        l = rand(dim, dim)
        c = dot(l, l.T)  # cov
        # generate samples (y1~N(m,c), y2~N(m,c)):
        y1 = multivariate_normal(m, c, num_of_samples_max)
        y2 = multivariate_normal(m, c, num_of_samples_max)
    elif distr == 'normalI':
        m = 2 * rand(dim)
        c = eye(dim)
        # generate samples (y1~N(m,I), y2~N(m,I)):
        y1 = multivariate_normal(m, c, num_of_samples_max)
        y2 = multivariate_normal(m, c, num_of_samples_max)
    else:
        raise Exception('Distribution=?')

    d = 0

    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        # with broadcasting:
        d_hat_v[tk] = co.estimation(y1[0:num_of_samples],
                                    y2[0:num_of_samples])
        print("tk={0}/{1}".format(tk+1, length))

    # plot:
    plt.plot(num_of_samples_v, d_hat_v, num_of_samples_v, ones(length)*d)
    plt.xlabel('Number of samples')
    plt.ylabel('Divergence')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
