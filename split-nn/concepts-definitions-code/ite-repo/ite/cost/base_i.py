""" Base mutual information estimators. """

from numpy import sum, sqrt, isnan, exp, mean, eye, ones, dot, cumsum, \
                  hstack, newaxis, maximum, prod, abs, arange, log
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from scipy.special import factorial
from scipy.linalg import det
from scipy.sparse.linalg import eigsh

from ite.cost.x_initialization import InitX, InitEtaKernel
from ite.cost.x_verification import VerCompSubspaceDims, \
                                    VerSubspaceNumberIsK,\
                                    VerOneDSubspaces
from ite.shared import compute_dcov_dcorr_statistics, median_heuristic,\
                       copula_transformation, compute_matrix_r_kcca_kgv
from ite.cost.x_kernel import Kernel


class BIDistCov(InitX, VerCompSubspaceDims, VerSubspaceNumberIsK):
    """ Distance covariance estimator using pairwise distances.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerCompSubspaceDims' and 'VerSubspaceNumber' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True, alpha=1):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        alpha : float, optional
                Parameter of the distance covariance: 0 < alpha < 2
                (default is 1).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.BIDistCov()
        >>> co2 = ite.cost.BIDistCov(alpha = 1.2)
        
        """
        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # other attribute:
        if alpha <= 0 or alpha >= 2:
            raise Exception('0 < alpha < 2 is needed for this estimator!')

        self.alpha = alpha
        
    def estimation(self, y, ds):
        """ Estimate distance covariance.
        
        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector
             Dimensions of the individual subspaces in y; ds[i] = i^th
             subspace dimension. len(ds) = 2.
    
        Returns
        -------
        i : float
            Estimated distance covariance.
            
        References
        ----------
        Gabor J. Szekely and Maria L. Rizzo. Brownian distance covariance. 
        The Annals of Applied Statistics, 3:1236-1265, 2009.
        
        Gabor J. Szekely, Maria L. Rizzo, and Nail K. Bakirov. Measuring
        and testing dependence by correlation of distances. The Annals of
        Statistics, 35:2769-2794, 2007.
        
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    
        
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        self.verification_subspace_number_is_k(ds, 2)
       
        num_of_samples = y.shape[0]  # number of samples
        a = compute_dcov_dcorr_statistics(y[:, :ds[0]], self.alpha)
        b = compute_dcov_dcorr_statistics(y[:, ds[0]:], self.alpha)
        i = sqrt(sum(a*b)) / num_of_samples
        
        return i


class BIDistCorr(InitX, VerCompSubspaceDims, VerSubspaceNumberIsK):
    """ Distance correlation estimator using pairwise distances.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerCompSubspaceDims' and 'VerSubspaceNumber' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True, alpha=1):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        alpha : float, optional
                 Parameter of the distance covariance: 0 < alpha < 2
                 (default is 1).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.BIDistCorr()
        >>> co2 = ite.cost.BIDistCorr(alpha = 1.2)
        
        """
        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # other attribute:
        if alpha <= 0 or alpha >= 2:
            raise Exception('0 < alpha < 2 is needed for this estimator!')

        self.alpha = alpha
        
    def estimation(self, y, ds):
        """ Estimate distance correlation.
        
        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector
             Dimensions of the individual subspaces in y; ds[i] = i^th
             subspace dimension. len(ds) = 2.
    
        Returns
        -------
        i : float
            Estimated distance correlation.
            
        References
        ----------
        Gabor J. Szekely and Maria L. Rizzo. Brownian distance covariance. 
        The Annals of Applied Statistics, 3:1236-1265, 2009.
        
        Gabor J. Szekely, Maria L. Rizzo, and Nail K. Bakirov. Measuring
        and testing dependence by correlation of distances. The Annals of
        Statistics, 35:2769-2794, 2007.
        
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    
        
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        self.verification_subspace_number_is_k(ds, 2)
        
        a = compute_dcov_dcorr_statistics(y[:, :ds[0]], self.alpha)
        b = compute_dcov_dcorr_statistics(y[:, ds[0]:], self.alpha)

        n = sum(a*b)  # numerator
        d1 = sum(a**2)  # denumerator-1 (without sqrt)
        d2 = sum(b**2)  # denumerator-2 (without sqrt)

        if (d1 * d2) == 0:  # >=1 of the random variables is constant
            i = 0
        else:
            i = n / sqrt(d1 * d2)  # <A,B> / sqrt(<A,A><B,B>)
            i = sqrt(i)
       
        return i


class BI3WayJoint(InitX, VerCompSubspaceDims, VerSubspaceNumberIsK):
    """ Joint dependency from the mean embedding of the 'joint minus the
    product of the marginals'.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerCompSubspaceDims' and 'VerSubspaceNumber' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True, sigma1=0.1, sigma2=0.1, sigma3=0.1):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        sigma1 : float, optional
                 Std in the RBF kernel on the first subspace (default is
                 sigma1 = 0.1). sigma1 = nan means 'use median heuristic'.
        sigma2 : float, optional
                 Std in the RBF kernel on the second subspace (default is 
                 sigma2 = 0.1). sigma2 = nan means 'use median heuristic'.
        sigma3 : float, optional
                 Std in the RBF kernel on the third subspace (default is
                 sigma3 = 0.1). sigma3 = nan means 'use median heuristic'.
                     
        Examples
        --------
        >>> from numpy import nan
        >>> import ite
        >>> co1 = ite.cost.BI3WayJoint()
        >>> co2 = ite.cost.BI3WayJoint(sigma1=0.1,sigma2=0.1,sigma3=0.1)
        >>> co3 = ite.cost.BI3WayJoint(sigma1=nan,sigma2=nan,sigma3=nan)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # other attributes:
        self.sigma1, self.sigma2, self.sigma3 = sigma1, sigma2, sigma3
        
    def estimation(self, y, ds):
        """ Estimate joint dependency.
        
        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector
             Dimensions of the individual subspaces in y; ds[i] = i^th
             subspace dimension. len(ds) = 3.
    
        Returns
        -------
        i : float
            Estimated joint dependency.
            
        References
        ----------
        Dino Sejdinovic, Arthur Gretton, and Wicher Bergsma. A kernel test
        for three-variable interactions. In Advances in Neural Information
        Processing Systems (NIPS), pages 1124-1132, 2013. (Lancaster 
        three-variable interaction based dependency index).
        
        Henry Oliver Lancaster. The Chi-squared Distribution. John Wiley
        and Sons Inc, 1969. (Lancaster interaction)
        
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    
        
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        self.verification_subspace_number_is_k(ds, 3)
        
        # Gram matrices (k1,k2,k3):
        sigma1, sigma2, sigma3 = self.sigma1, self.sigma2, self.sigma3
        # k1 (set co.sigma1 using median heuristic, if needed):
        if isnan(sigma1):
            sigma1 = median_heuristic(y[:, 0:ds[0]])
            
        k1 = squareform(pdist(y[:, 0:ds[0]]))
        k1 = exp(-k1**2 / (2 * sigma1**2))

        # k2 (set co.sigma2 using median heuristic, if needed):
        if isnan(sigma2):
            sigma2 = median_heuristic(y[:, ds[0]:ds[0]+ds[1]])
            
        k2 = squareform(pdist(y[:, ds[0]:ds[0]+ds[1]]))
        k2 = exp(-k2**2 / (2 * sigma2**2))

        # k3 (set co.sigma3 using median heuristic, if needed):
        if isnan(sigma3):
            sigma3 = median_heuristic(y[:, ds[0]+ds[1]:])
            
        k3 = squareform(pdist(y[:, ds[0]+ds[1]:], 'euclidean'))
        k3 = exp(-k3**2 / (2 * sigma3**2))

        prod_of_ks = k1 * k2 * k3  # Hadamard product
        term1 = mean(prod_of_ks)
        term2 = -2 * mean(mean(k1, axis=1) * mean(k2, axis=1) *
                          mean(k3, axis=1))
        term3 = mean(k1) * mean(k2) * mean(k3)
        i = term1 + term2 + term3
            
        return i


class BI3WayLancaster(InitX, VerCompSubspaceDims, VerSubspaceNumberIsK):
    """ Estimate the Lancaster three-variable interaction measure. 
    
    Partial initialization comes from 'InitX', verification is from 
    'VerCompSubspaceDims' and 'VerSubspaceNumber' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True, sigma1=0.1, sigma2=0.1, sigma3=0.1):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        sigma1 : float, optional
                 Std in the RBF kernel on the first subspace (default is
                 sigma1 = 0.1). sigma1 = nan means 'use median heuristic'.
        sigma2 : float, optional
                 Std in the RBF kernel on the second subspace (default is 
                 sigma2 = 0.1). sigma2 = nan means 'use median heuristic'.
        sigma3 : float, optional
                 Std in the RBF kernel on the third subspace (default is
                 sigma3 = 0.1). sigma3 = nan means 'use median heuristic'.
                     
        Examples
        --------
        >>> from numpy import nan
        >>> import ite
        >>> co1 = ite.cost.BI3WayLancaster()
        >>> co2 = ite.cost.BI3WayLancaster(sigma1=0.1, sigma2=0.1,\
                                           sigma3=0.1)
        >>> co3 = ite.cost.BI3WayLancaster(sigma1=nan, sigma2=nan,\
                                           sigma3=nan)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # other attributes:
        self.sigma1, self.sigma2, self.sigma3 = sigma1, sigma2, sigma3
        
    def estimation(self, y, ds):
        """ Estimate Lancaster three-variable interaction measure.
        
        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector
             Dimensions of the individual subspaces in y; ds[i] = i^th
             subspace dimension. len(ds) = 3.
    
        Returns
        -------
        i : float
            Estimated Lancaster three-variable interaction measure.
            
        References
        ----------
        Dino Sejdinovic, Arthur Gretton, and Wicher Bergsma. A kernel test
        for three-variable interactions. In Advances in Neural Information
        Processing Systems (NIPS), pages 1124-1132, 2013. (Lancaster 
        three-variable interaction based dependency index).
        
        Henry Oliver Lancaster. The Chi-squared Distribution. John Wiley
        and Sons Inc, 1969. (Lancaster interaction)
        
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    
        
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        self.verification_subspace_number_is_k(ds, 3)
        
        num_of_samples = y.shape[0]  # number of samples
        
        # Gram matrices (k1,k2,k3):
        sigma1, sigma2, sigma3 = self.sigma1, self.sigma2, self.sigma3
        # k1 (set co.sigma1 using median heuristic, if needed):
        if isnan(sigma1):
            sigma1 = median_heuristic(y[:, 0:ds[0]])
            
        k1 = squareform(pdist(y[:, 0:ds[0]]))
        k1 = exp(-k1**2 / (2 * sigma1**2))

        # k2 (set co.sigma2 using median heuristic, if needed):
        if isnan(sigma2):
            sigma2 = median_heuristic(y[:, ds[0]:ds[0]+ds[1]])
            
        k2 = squareform(pdist(y[:, ds[0]:ds[0]+ds[1]]))
        k2 = exp(-k2**2 / (2 * sigma2**2))

        # k3 set co.sigma3 using median heuristic, if needed():
        if isnan(sigma3):
            sigma3 = median_heuristic(y[:, ds[0]+ds[1]:])
            
        k3 = squareform(pdist(y[:, ds[0]+ds[1]:]))
        k3 = exp(-k3**2 / (2 * sigma3**2))

        # centering of k1, k2, k3:
        h = eye(num_of_samples) -\
            ones((num_of_samples, num_of_samples)) / num_of_samples
        k1 = dot(dot(h, k1), h)
        k2 = dot(dot(h, k2), h)
        k3 = dot(dot(h, k3), h)
        i = mean(k1 * k2 * k3)
           
        return i


class BIHSIC_IChol(InitEtaKernel, VerCompSubspaceDims):
    """ Estimate HSIC using incomplete Cholesky decomposition.

    HSIC refers to Hilbert-Schmidt Independence Criterion.

    Partial initialization comes from 'InitEtaKernel', verification is
    from 'VerCompSubspaceDims' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').

    Notes
    -----
    The current implementation uses the same kernel an all the subspaces:
    k = k_1 = ... = k_M, where y = [y^1;...;y^M].

    Examples
    --------
    >>> from ite.cost.x_kernel import Kernel
    >>> import ite
    >>> co1 = ite.cost.BIHSIC_IChol()
    >>> co2 = ite.cost.BIHSIC_IChol(eta=1e-3)
    >>> k = Kernel({'name': 'RBF','sigma': 1})
    >>> co3 = ite.cost.BIHSIC_IChol(kernel=k, eta=1e-3)

    """

    def estimation(self, y, ds):
        """ Estimate HSIC.

        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector
             Dimensions of the individual subspaces in y; ds[i] = i^th
             subspace dimension.

        Returns
        -------
        i : float
            Estimated value of HSIC.

        References
        ----------
        Arthur Gretton, Olivier Bousquet, Alexander Smola and Bernhard
        Scholkopf. Measuring Statistical Dependence with Hilbert-Schmidt
        Norms. International Conference on Algorithmic Learnng Theory
        (ALT), 63-78, 2005.

        Alain Berlinet and Christine Thomas-Agnan. Reproducing Kernel
        Hilbert Spaces in Probability and Statistics. Kluwer, 2004. (mean
        embedding)

        Examples
        --------
        i = co.estimation(y,ds)

        """

        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)

        # initialization:
        num_of_samples = y.shape[0]  # number of samples
        num_of_subspaces = len(ds)

        # Step-1 (g1, g2, ...):
        # 0,d_1,d_1+d_2,...,d_1+...+d_{M-1}; starting indices of the
        # subspaces:
        cum_ds = cumsum(hstack((0, ds[:-1])))
        gs = list()
        for m in range(num_of_subspaces):
            idx = range(cum_ds[m], cum_ds[m] + ds[m])
            g = self.kernel.ichol(y[:, idx], num_of_samples * self.eta)
            g = g - mean(g, axis=0)  # center the Gram matrix: dot(g,g.T)
            gs.append(g)

        # Step-2 (g1, g2, ... -> i):
        i = 0
        for i1 in range(num_of_subspaces-1):       # i1 = 0:M-2
            for i2 in range(i1+1, num_of_subspaces):  # i2 = i1+1:M-1
                i += norm(dot(gs[i2].T, gs[i1]))**2  # norm = Frob. norm

        i /= num_of_samples**2

        return i


class BIHoeffding(InitX, VerOneDSubspaces, VerCompSubspaceDims):
    """ Estimate the multivariate version of Hoeffding's Phi.

       Partial initialization comes from 'InitX', verification is from
       'VerCompSubspaceDims' and 'VerSubspaceNumber' (see
       'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').

       """

    def __init__(self, mult=True, small_sample_adjustment=True):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        small_sample_adjustment: boolean, optional
                                 Whether we want small-sample adjustment.

        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.BIHoeffding()
        >>> co2 = ite.cost.BIHoeffding(small_sample_adjustment=False)

        """

        # initialize with 'InitX':
        super().__init__(mult=mult)

        # other attributes:
        self.small_sample_adjustment = small_sample_adjustment

    def estimation(self, y, ds):
        """ Estimate multivariate version of Hoeffding's Phi.

        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector
             Dimensions of the individual subspaces in y; ds[i] = i^th
             subspace dimension = 1 for this estimator.

        Returns
        -------
        i : float
            Estimated value of the multivariate version of Hoeffding's Phi.

        References
        ----------
        Sandra Gaiser, Martin Ruppert, Friedrich Schmid. A multivariate
        version of Hoeffding's Phi-Square. Journal of Multivariate
        Analysis. 101: pages 2571-2586, 2010.

        Examples
        --------
        i = co.estimation(y,ds)

        """

        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        self.verification_one_dimensional_subspaces(ds)

        num_of_samples, dim = y.shape
        u = copula_transformation(y)

        # term1:
        m = 1 - maximum(u[:, 0][:, newaxis], u[:, 0])
        for i in range(1, dim):
            m *= 1 - maximum(u[:, i][:, newaxis], u[:, i])

        term1 = mean(m)

        # term2:
        if self.small_sample_adjustment:
            term2 = \
                - mean(prod(1 - u**2 - (1 - u) / num_of_samples,
                            axis=1)) / \
                (2**(dim - 1))
        else:
            term2 = - mean(prod(1 - u**2, axis=1)) / (2 ** (dim - 1))

        # term3:
        if self.small_sample_adjustment:
            term3 = \
                ((num_of_samples - 1) * (2 * num_of_samples-1) /
                 (3 * 2 * num_of_samples**2))**dim
        else:
            term3 = 1 / 3**dim

        i = term1 + term2 + term3

        if self.mult:
            if self.small_sample_adjustment:
                t1 = \
                    sum((1 - arange(1,
                                    num_of_samples) / num_of_samples)**dim
                        * (2*arange(1, num_of_samples) - 1)) \
                    / num_of_samples**2
                t2 = \
                    -2 * mean(((num_of_samples * (num_of_samples - 1) -
                                arange(1, num_of_samples+1) *
                                arange(num_of_samples)) /
                               (2 * num_of_samples ** 2))**dim)
                t3 = term3
                inv_hd = t1 + t2 + t3  # 1 / h(d, n)
            else:
                inv_hd = \
                    2 / ((dim + 1) * (dim + 2)) - factorial(dim) / \
                    (2 ** dim * prod(arange(dim + 1) + 1 / 2)) + \
                    1 / 3 ** dim  # 1 / h(d)s

            i /= inv_hd

        i = sqrt(abs(i))

        return i


class BIKGV(InitEtaKernel, VerCompSubspaceDims):
    """ Estimate kernel generalized variance (KGV).

     Partial initialization comes from 'InitEtaKernel', verification is
     from 'VerCompSubspaceDims' (see 'ite.cost.x_initialization.py',
     'ite.cost.x_verification.py').

    """

    def __init__(self, mult=True, kernel=Kernel(), eta=1e-2, kappa=0.01):
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
              means larger sized Gram factor and better approximation.
              (default is 1e-2)
        kappa: float, >0
               Regularization parameter.

        Examples
        --------
        >>> import ite
        >>> from ite.cost.x_kernel import Kernel
        >>> co1 = ite.cost.BIKGV()
        >>> co2 = ite.cost.BIKGV(eta=1e-4)
        >>> co3 = ite.cost.BIKGV(eta=1e-4, kappa=0.02)
        >>> k =  Kernel({'name': 'RBF', 'sigma': 0.3})
        >>> co4 = ite.cost.BIKGV(eta=1e-4, kernel=k)

        """

        # initialize with 'InitEtaKernel':
        super().__init__(mult=mult, kernel=kernel, eta=eta)

        # other attributes:
        self.kappa = kappa

    def estimation(self, y, ds):
        """ Estimate KGV.

        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector
             Dimensions of the individual subspaces in y; ds[i] = i^th
             subspace dimension.

        Returns
        -------
        i : float
            Estimated value of KGV.

        References
        ----------
        Francis Bach, Michael I. Jordan. Kernel Independent Component
        Analysis. Journal of Machine Learning Research, 3: 1-48, 2002.

        Francis Bach, Michael I. Jordan. Learning graphical models with
        Mercer kernels. International Conference on Neural Information
        Processing Systems (NIPS), pages 1033-1040, 2002.

        Examples
        --------
        i = co.estimation(y,ds)

        """

        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)

        num_of_samples = y.shape[0]
        tol = num_of_samples * self.eta

        r = compute_matrix_r_kcca_kgv(y, ds, self.kernel, tol, self.kappa)
        i = -log(det(r)) / 2

        return i


class BIKCCA(InitEtaKernel, VerCompSubspaceDims):
    """ Kernel canonical correlation analysis (KCCA) based estimator.

     Partial initialization comes from 'InitEtaKernel', verification is
     from 'VerCompSubspaceDims' (see 'ite.cost.x_initialization.py',
     'ite.cost.x_verification.py').

    """

    def __init__(self, mult=True, kernel=Kernel(), eta=1e-2, kappa=0.01):
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
              means larger sized Gram factor and better approximation.
              (default is 1e-2)
        kappa: float, >0
               Regularization parameter.

        Examples
        --------
        >>> import ite
        >>> from ite.cost.x_kernel import Kernel
        >>> co1 = ite.cost.BIKCCA()
        >>> co2 = ite.cost.BIKCCA(eta=1e-4)
        >>> co3 = ite.cost.BIKCCA(eta=1e-4, kappa=0.02)
        >>> k =  Kernel({'name': 'RBF', 'sigma': 0.3})
        >>> co4 = ite.cost.BIKCCA(eta=1e-4, kernel=k)

        """

        # initialize with 'InitEtaKernel':
        super().__init__(mult=mult, kernel=kernel, eta=eta)

        # other attributes:
        self.kappa = kappa

    def estimation(self, y, ds):
        """ Estimate KCCA.

        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector
             Dimensions of the individual subspaces in y; ds[i] = i^th
             subspace dimension.

        Returns
        -------
        i : float
            Estimated value of KCCA.

        References
        ----------
        Francis Bach, Michael I. Jordan. Learning graphical models with
        Mercer kernels. International Conference on Neural Information
        Processing Systems (NIPS), pages 1033-1040, 2002.

        Examples
        --------
        i = co.estimation(y,ds)

        """

        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)

        num_of_samples = y.shape[0]
        tol = num_of_samples * self.eta

        r = compute_matrix_r_kcca_kgv(y, ds, self.kernel, tol, self.kappa)
        eig_min = eigsh(r, k=1, which='SM')[0][0]
        i = -log(eig_min) / 2

        return i
