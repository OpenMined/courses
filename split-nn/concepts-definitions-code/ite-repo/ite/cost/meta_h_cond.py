""" Meta conditional entropy estimators. """

from ite.cost.x_initialization import InitX
from ite.cost.x_factory import co_factory


class BcondHShannon_HShannon(InitX):
    """ Conditional Shannon entropy estimator based on unconditional one.

    The estimation relies on the identity H(y^1|y^2) = H([y^1;y^2]) -
    H(y^2), where H is the Shannon differential entropy.

    Partial initialization comes from 'InitX' (see
    'ite.cost.x_initialization.py').

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

        --------
        >>> import ite
        >>> co1 = ite.cost.BcondHShannon_HShannon()
        >>> co2 = ite.cost.BcondHShannon_HShannon(\
                              h_shannon_co_name='BHShannon_KnnK')
        >>> dict_ch = {'k': 2, 'eps': 0.2}
        >>> co3 = ite.cost.BcondHShannon_HShannon(\
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

    def estimation(self, y, dim1):
        """ Estimate conditional Shannon entropy.

        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
              One row of y corresponds to one sample from [y1; y2].
        dim1: integer, >0
              Dimension of y1.

        Returns
        -------
        cond_h : float
                 Estimated conditional Shannon entropy.


        Examples
        --------
        cond_h = co.estimation(y,dim1)

        """

        # Shannon entropy of y2:
        h2 = self.h_shannon_co.estimation(y[:, dim1:])

        # Shannon entropy of [y1;y2]:
        h12 = self.h_shannon_co.estimation(y)

        cond_h = h12 - h2
        return cond_h
