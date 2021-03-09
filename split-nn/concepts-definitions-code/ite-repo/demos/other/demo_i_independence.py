#!/usr/bin/env python3

""" Demo for 'I(y^1,...,y^m)=0 if y^m-s are independent'.

In the demo, uniform random variables are considered.

"""

from scipy.stats import special_ortho_group
from numpy import ones, arange, zeros, sum, cumsum, dot, hstack
from numpy.random import rand, choice
import matplotlib.pyplot as plt
from ite.cost.x_factory import co_factory


def main():
    # parameters:
    m = 2  # number of components
    dm = 1  # dimension of the components
    num_of_samples_v = arange(500, 5*1000+1, 500)

    cost_name = 'BIKCCA'           # m  = 2, dm >= 1
    # cost_name = 'BIKGV'            # m  = 2, dm >= 1
    # cost_name = 'BIHSIC_IChol'     # m >= 2, dm >= 1
    # cost_name = 'BIDistCov'        # m  = 2, dm >= 1
    # cost_name = 'BIDistCorr'       # m  = 2, dm >= 1
    # cost_name = 'BI3WayJoint'      # m  = 3, dm >= 1
    # cost_name = 'BI3WayLancaster'  # m  = 3, dm >= 1
    # cost_name = 'MIShannon_DKL'    # m >= 2, dm >= 1
    # cost_name = 'MIChi2_DChi2'     # m >= 2, dm >= 1
    # cost_name = 'MIL2_DL2'         # m >= 2, dm >= 1
    # cost_name = 'MIRenyi_DR'       # m >= 2, dm >= 1
    # cost_name = 'MITsallis_DT'     # m >= 2, dm >= 1
    # cost_name = 'MIMMD_CopulaDMMD' # m >= 2, dm  = 1
    # cost_name = 'MIRenyi_HR'       # m >= 2, dm  = 1
    # cost_name = 'MIShannon_HS'     # m >= 2, dm >= 1
    # cost_name = 'MIDistCov_HSIC'   # m  = 2, dm >= 1
    # cost_name = 'BIHoeffding'      # m >= 2, dm  = 1

    # initialization:
    distr = 'uniform'  # fixed
    ds = dm * ones(m, dtype='int')
    num_of_samples_max = num_of_samples_v[-1]
    length = len(num_of_samples_v)
    co = co_factory(cost_name, mult=True)  # cost object
    # vector to store the estimated mutual information values:
    i_hat_v = zeros(length)

    # distr -> samples (y); analytical value (i):
    if distr == 'uniform':
        y = rand(num_of_samples_max, sum(ds))
        # 0,d_1,d_1+d_2,...,d_1+...+d_{M-1}; starting indices of the
        # subspaces:
        cum_ds = cumsum(hstack((0, ds[:-1])))
        for i in range(len(ds)):
            # orthm : ds[i] x ds[i]-sized random orthogonal matrix
            if dm == 1:
                orthm = choice([-1, 1])
            else:
                orthm = special_ortho_group.rvs(ds[i])

            idx = range(cum_ds[i], cum_ds[i] + ds[i])
            y[:, idx] = dot(y[:, idx], orthm)
    else:
        raise Exception('Distribution=?')

    i = 0

    # estimation:
    for (tk, num_of_samples) in enumerate(num_of_samples_v):
        i_hat_v[tk] = co.estimation(y[0:num_of_samples], ds)  # broadcast
        print("tk={0}/{1}".format(tk+1, length))

    # plot:    
    plt.plot(num_of_samples_v, i_hat_v, num_of_samples_v, ones(length)*i)
    plt.xlabel('Number of samples')
    plt.ylabel('Mutual information')
    plt.legend(('estimation', 'analytical value'), loc='best')
    plt.title("Estimator: " + cost_name)
    plt.show()


if __name__ == "__main__":
    main()
