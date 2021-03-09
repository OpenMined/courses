""" Initialization classes for estimators.

For entropy / mutual information / divergence / cross quantity /
association / distribution kernel estimators.

These initialization classes are not called directly, but they are used by
inheritance. For example one typically derives a k-nearest neighbor based
estimation method from InitKnnK. InitKnnK sets (default values) for (i)
the kNN computation technique called (kNN_method), (ii) the number of
neighbors (k) and (iii) the accuracy required in kNN computation (eps).

Note: InitKnnK (and all other classes here, except for InitBagGram) are
subclasses of InitX, which makes them printable. InitBagGram is used with
classes derived from InitX.

"""

from numpy import zeros, mod, array
# from numpy import zeros, floor, mod
from ite.cost.x_kernel import Kernel


class InitX(object):
    """ Base class of all estimators giving string representation and mult.

    """

    def __init__(self, mult=True):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)

        """

        self.mult = mult

    def __str__(self):
        """ String representation of the estimator.

        Application: print(cost_object)

        Examples
        --------
        >>> import ite
        >>> co = ite.cost.x_initialization.InitX()
        >>> print(co)
        InitX -> {'mult': True}

        """

        return ''.join((self.__class__.__name__, ' -> ',
                        str(self.__dict__)))


class InitKnnK(InitX):
    """ Initialization class for estimators based on kNNs.

    k-nearest neighbors: S = {k}.

    Partial initialization comes from 'InitX'.

    """

    def __init__(self, mult=True, knn_method='cKDTree', k=3, eps=0):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        knn_method : str, optional
                     kNN computation method; 'cKDTree' or 'KDTree'.
        k : int, >= 1, optional
            k-nearest neighbors (default is 3).
        eps : float, >= 0, optional
              The k^th returned value is guaranteed to be no further than
              (1+eps) times the distance to the real kNN (default is 0).

        Examples
        --------
        >>> import ite
        >>> co = ite.cost.x_initialization.InitKnnK()

        """

        # initialize with 'InitX':
        super().__init__(mult=mult)

        # other attributes:
        self.knn_method, self.k, self.eps = knn_method, k, eps


class InitKnnKiTi(InitX):
    """ Initialization class for estimators based on k-nearest neighbors.

    k here depends on the number of samples: S = {ki(Ti)}.

    Partial initialization comes from 'InitX'.

    """

    def __init__(self, mult=True, knn_method='cKDTree', eps=0):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        knn_method : str, optional
                     kNN computation method; 'cKDTree' or 'KDTree'.
        eps : float, >= 0, optional
              The k^th returned value is guaranteed to be no further than
              (1+eps) times the distance to the real kNN (default is 0).

        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.x_initialization.InitKnnKiTi()
        >>> co2 = ite.cost.x_initialization.InitKnnKiTi(eps=0.1)

        """

        # initialize with 'InitX':
        super().__init__(mult=mult)

        # other attributes:
        self.knn_method, self.eps = knn_method, eps


class InitAlpha(InitX):
    """ Initialization class for estimators using an alpha \ne 1 parameter.

    Partial initialization comes from 'InitX'.

    """

    def __init__(self, mult=True, alpha=0.99):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        alpha : float, alpha \ne 1, optional
                (default is 0.99)

        Examples
        --------
        >>> import ite
        >>> co = ite.cost.x_initialization.InitAlpha()

        """

        # initialize with 'InitX':
        super().__init__(mult=mult)

        # alpha:
        if alpha == 1:
            raise Exception('Alpha can not be 1 for this estimator!')

        self.alpha = alpha


class InitUAlpha(InitAlpha):
    """ Initialization for estimators with an u>0 & alpha \ne 1 parameter.

    Partial initialization comes from 'InitAlpha'.

    """

    def __init__(self, mult=True, u=1.0, alpha=0.99):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        u    : float, 0 < u, optional
               (default is 1.0)
        alpha : float, alpha \ne 1, optional
                (default is 0.99)

        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.x_initialization.InitUAlpha()
        >>> co2 = ite.cost.x_initialization.InitUAlpha(alpha=0.7)
        >>> co3 = ite.cost.x_initialization.InitUAlpha(u=1.2)
        >>> co4 = ite.cost.x_initialization.InitUAlpha(u=1.2, alpha=0.7)


        """

        # initialize with 'InitAlpha' (it also checks the validity of
        # alpha):
        super().__init__(mult=mult, alpha=alpha)

        # u verification:
        if u <= 0:
            raise Exception('u has to positive for this estimator!')
        self.u = u


class InitKnnKAlpha(InitAlpha):
    """ Initialization for estimators based on kNNs and an alpha \ne 1.

    k-nearest neighbors: S = {k}.

    Partial initialization comes from 'InitAlpha'.

    """

    def __init__(self, mult=True, knn_method='cKDTree', k=3, eps=0,
                 alpha=0.99):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        knn_method : str, optional
                     kNN computation method; 'cKDTree' or 'KDTree'.
        k : int, >= 1, optional
            k-nearest neighbors (default is 3).
        eps : float, >= 0
              The k^th returned value is guaranteed to be no further than
              (1+eps) times the distance to the real kNN (default is 0).
        alpha : float, alpha \ne 1, optional
                (default is 0.99)

        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.x_initialization.InitKnnKAlpha()
        >>> co2 = ite.cost.x_initialization.InitKnnKAlpha(k=2)
        >>> co3 = ite.cost.x_initialization.InitKnnKAlpha(alpha=0.9)
        >>> co4 = ite.cost.x_initialization.InitKnnKAlpha(k=2, alpha=0.9)

        """

        # initialize with 'InitAlpha' (it also checks the validity of
        # alpha):
        super().__init__(mult=mult, alpha=alpha)

        # kNN attributes:
        self.knn_method, self.k, self.eps = knn_method, k, eps


class InitKnnKAlphaBeta(InitKnnKAlpha):
    """ Initialization for estimators based on kNNs; alpha & beta \ne 1.

    k-nearest neighbors: S = {k}.

    Partial initialization comes from 'InitKnnKAlpha'.

    """

    def __init__(self, mult=True, knn_method='cKDTree', k=3, eps=0,
                 alpha=0.9, beta=0.99):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        knn_method : str, optional
                     kNN computation method; 'cKDTree' or 'KDTree'.
        k : int, >= 1, optional
            k-nearest neighbors
            (default is 3).
        eps : float, >= 0
              The k^th returned value is guaranteed to be no further than
              (1+eps) times the distance to the real kNN (default is 0).
        alpha : float, alpha \ne 1, optional
                (default is 0.9)
        beta : float, beta \ne 1, optional
               (default is 0.99)

        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.x_initialization.InitKnnKAlphaBeta()
        >>> co2 = ite.cost.x_initialization.InitKnnKAlphaBeta(k=2)
        >>> co3 = ite.cost.x_initialization.InitKnnKAlphaBeta(alpha=0.8)
        >>> co4 = ite.cost.x_initialization.InitKnnKAlphaBeta(beta=0.7)
        >>> co5 = ite.cost.x_initialization.InitKnnKAlphaBeta(eps=0.1)

        >>> co6 = ite.cost.x_initialization.InitKnnKAlphaBeta(k=2,\
                                                              alpha=0.8)
        >>> co7 = ite.cost.x_initialization.InitKnnKAlphaBeta(k=2,\
                                                              beta=0.7)
        >>> co8 = ite.cost.x_initialization.InitKnnKAlphaBeta(k=2,\
                                                              eps=0.1)
        >>> co9 = ite.cost.x_initialization.InitKnnKAlphaBeta(alpha=0.8,\
                                                              beta=0.7)
        >>> co10 = ite.cost.x_initialization.InitKnnKAlphaBeta(alpha=0.8,\
                                                               eps=0.1)
        >>> co11 = ite.cost.x_initialization.InitKnnKAlphaBeta(beta=0.7,\
                                                               eps=0.1)
        >>> co12 = ite.cost.x_initialization.InitKnnKAlphaBeta(alpha=0.8,\
                                                               beta=0.7,\
                                                               eps=0.2)
        >>> co13 = ite.cost.x_initialization.InitKnnKAlphaBeta(k=2,\
                                                               beta=0.7,\
                                                               eps=0.2)
        >>> co14 = ite.cost.x_initialization.InitKnnKAlphaBeta(k=2,\
                                                               alpha=0.8,\
                                                               eps=0.2)
        >>> co15 = ite.cost.x_initialization.InitKnnKAlphaBeta(k=2,\
                                                               alpha=0.8,\
                                                               beta=0.7)
        >>> co16 = ite.cost.x_initialization.InitKnnKAlphaBeta(k=2,\
                                                               alpha=0.8,\
                                                               beta=0.7,\
                                                               eps=0.2)

        """

        # initialize with 'InitKnnKAlpha' (it also checks the validity of
        # alpha):
        super().__init__(mult=mult, knn_method=knn_method, k=k, eps=eps,
                         alpha=alpha)

        # b eta verification:
        if beta == 1:
            raise Exception('Beta can not be 1 for this estimator!')

        self.beta = beta


class InitKnnSAlpha(InitAlpha):
    """ Initialization for methods based on generalized kNNs & alpha \ne 1.

    k-nearest neighbors: S \subseteq {1,...,k}.

    Partial initialization comes from 'InitAlpha'.

    """

    def __init__(self, mult=True, knn_method='cKDTree', k=None, eps=0,
                 alpha=0.99):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        knn_method : str, optional
                     kNN computation method; 'cKDTree' or 'KDTree'.
        k : int, >= 1, optional
            k-nearest neighbors.  In case of 'None' a default
             array([1,2,4]) is taken.
        eps : float, >= 0
              The k^th returned value is guaranteed to be no further than
              (1+eps) times the distance to the real kNN (default is 0).
        alpha : float, alpha \ne 1, optional
                (default is 0.99)

        Examples
        --------
        >>> from numpy import array
        >>> import ite
        >>> co1 = ite.cost.x_initialization.InitKnnSAlpha()
        >>> co2a = ite.cost.x_initialization.InitKnnSAlpha(k=2)
        >>> co2b =\
                 ite.cost.x_initialization.InitKnnSAlpha(k=array([1,2,5]))
        >>> co3 = ite.cost.x_initialization.InitKnnSAlpha(alpha=0.8)
        >>> co4 = ite.cost.x_initialization.InitKnnSAlpha(eps=0.1)

        >>> co5a = ite.cost.x_initialization.InitKnnSAlpha(k=2, alpha=0.8)
        >>> co5b =\
                 ite.cost.x_initialization.InitKnnSAlpha(k=array([1,2,5]),\
                                                         alpha=0.8)
        >>> co6a = ite.cost.x_initialization.InitKnnSAlpha(k=2, eps=0.1)
        >>> co6b =\
                 ite.cost.x_initialization.InitKnnSAlpha(k=array([1,2,5]),\
                                                         eps=0.1)
        >>> co7 = ite.cost.x_initialization.InitKnnSAlpha(alpha=0.8,\
                                                          eps=0.1)
        >>> co8 = ite.cost.x_initialization.InitKnnSAlpha(k=2, alpha=0.8,\
                                                          eps=0.2)
        >>> co9 =\
                ite.cost.x_initialization.InitKnnSAlpha(k=array([1,2,5]),\
                                                        alpha=0.8,\
                                                        eps=0.2)

        """

        # initialize with 'InitAlpha' (it also checks the validity of
        # alpha):
        super().__init__(mult=mult, alpha=alpha)

        # kNN attribute:
        if k is None:
            k = array([1, 2, 4])

        self.knn_method, self.k, self.eps = knn_method, k, eps

        # alpha:
        if alpha == 1:
            raise Exception('Alpha can not be 1 for this estimator!')

        self.alpha = alpha


class InitKernel(InitX):
    """ Initialization class for kernel based estimators.

    Partial initialization comes from 'InitX'.

    """

    def __init__(self, mult=True, kernel=Kernel()):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        kernel : Kernel, optional
                 For examples, see 'ite.cost.x_kernel.Kernel'


        """

        # initialize with 'InitX':
        super().__init__(mult=mult)

        # other attributes:
        self.kernel = kernel


class InitBagGram(object):
    """ Initialization class for kernels on distributions.

    The class provides Gram matrix computation capability.

    """

    def gram_matrix(self, ys):
        """ Gram matrix computation on a collection of bags.

        Examples
        --------
        See 'ite/demos/other/demo_k_positive_semidefinite.py'.

        """

        num_of_distributions = len(ys)
        g = zeros((num_of_distributions, num_of_distributions))
        print('G computation: started.')

        for k1 in range(num_of_distributions):       # k1^th distribution
            if mod(k1, 10) == 0:
                print('k1=' + str(k1+1) + '/' + str(num_of_distributions) +
                      ': started.')

            for k2 in range(k1, num_of_distributions):  # k2^th distr.
                # K(y[k1], y[k2]):

                # version-1 (we care about independence for k1 == k2;
                # import floor):
                # if k1 == k2:
                #   num_of_samples_half = int(floor(ys[k1].shape[0] / 2))
                #   g[k1, k1] = \
                #       self.estimation(ys[k1][:num_of_samples_half],
                #                       ys[k1][num_of_samples_half:])
                # else:
                #   g[k1, k2] = self.estimation(ys[k1], ys[k2])
                #   g[k2, k1] = g[k1, k2]  # assumption: symmetry

                # version-2 (we do not care):
                g[k1, k2] = self.estimation(ys[k1], ys[k2])
                # Note: '.estimation()' is implemented in the kernel
                # classes
                g[k2, k1] = g[k1, k2]  # assumption: symmetry

        return g


class InitEtaKernel(InitKernel):
    """ Initialization for kernel based methods with an eta > 0 parameter.

    Eta is a tolerance parameter; it is used to control the approximation
    quality of incomplete Cholesky decomposition based approximation.

    Partial initialization comes from 'InitKernel'.

    """

    def __init__(self, mult=True, kernel=Kernel(), eta=1e-2):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        kernel : Kernel, optional
                 For examples, see 'ite.cost.x_kernel.Kernel'
        eta : float, >0, optional
              It is used to control the quality of the incomplete Cholesky
              decomposition based Gram matrix approximation. Smaller 'eta'
              means larger-sized Gram factor and better approximation.
              (default is 1e-2)
        """

        # initialize with 'InitKernel':
        super().__init__(mult=mult, kernel=kernel)

        # other attributes:
        self.eta = eta
