""" Meta conditional mutual information estimators. """

from numpy import cumsum, hstack

from ite.cost.x_initialization import InitX
from ite.cost.x_verification import VerCompSubspaceDims
from ite.cost.x_factory import co_factory


class BcondIShannon_HShannon(InitX, VerCompSubspaceDims):
    """ Estimate conditional mutual information from unconditional Shannon
    entropy.

    Partial initialization comes from 'InitX', verification is from
    'VerCompSubspaceDims' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').

    """

    def __init__(self, mult=True, h_shannon_co_name='BHShannon_KnnK',
                 h_shannon_co_pars=None):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        h_shannon_co_name : str, optional
                            You can change it to any Shannon entropy
                            estimator. (default is 'BHShannon_KnnK')
        h_shannon_co_pars : dictionary, optional
                            Parameters for the Shannon entropy estimator.
                            (default is None (=> {}); in this case the
                            default parameter values of the Shannon
                            entropy estimator are used)
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.BcondIShannon_HShannon()
        >>> co2 = ite.cost.BcondIShannon_HShannon(\
                              h_shannon_co_name='BHShannon_KnnK')
        >>> dict_ch = {'k': 2, 'eps': 0.2}
        >>> co3 = ite.cost.BcondIShannon_HShannon(\
                              h_shannon_co_name='BHShannon_KnnK', \
                              h_shannon_co_pars=dict_ch)
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)

        # initialize the Shannon entropy estimator:
        h_shannon_co_pars = h_shannon_co_pars or {}
        h_shannon_co_pars['mult'] = True  # guarantee this property
        self.h_shannon_co = co_factory(h_shannon_co_name,
                                       **h_shannon_co_pars)

    def estimation(self, y, ds):
        """ Estimate conditional Shannon mutual information.

        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector
             Dimensions of the individual subspaces in y; ds[i] = i^th
             subspace dimension. The last block is the conditioning
             variable.

        Returns
        -------
        cond_i : float
                 Estimated conditional mutual information.

        Examples
        --------
        cond_i = co.estimation(y,ds)

        """

        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        len_ds = len(ds)
        if len_ds <= 2:
            raise Exception('At least two non-conditioning subspaces are '
                            'needed!')

        # initialization:
        # 0,d_1,d_1+d_2,...,d_1+...+d_M; starting indices of the subspaces:
        cum_ds = cumsum(hstack((0, ds[:-1])))
        idx_condition = range(cum_ds[len_ds-1],
                              cum_ds[len_ds-1] + ds[len_ds-1])

        # h_joint:
        h_joint = self.h_shannon_co.estimation(y)

        # h_cross:
        h_cross = 0
        for m in range(len_ds-1):  # non-conditioning subspaces
            idx_m = range(cum_ds[m], cum_ds[m] + ds[m])
            h_cross += \
                self.h_shannon_co.estimation(y[:, hstack((idx_m,
                                                          idx_condition))])

        # h_condition:
        h_condition = self.h_shannon_co.estimation(y[:, idx_condition])

        cond_i = -h_joint + h_cross - (len_ds - 2) * h_condition

        return cond_i
