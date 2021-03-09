""" Base divergence estimators. """

from numpy import mean, log, absolute, sqrt, floor, sum, arange, vstack, \
                  dot, abs
from scipy.spatial.distance import cdist, pdist 

from ite.cost.x_initialization import InitX, InitKnnK, InitKnnKiTi, \
                                      InitKnnKAlpha, InitKnnKAlphaBeta, \
                                      InitKernel, InitEtaKernel
from ite.cost.x_verification import VerEqualDSubspaces, \
                                    VerEqualSampleNumbers, \
                                    VerEvenSampleNumbers

from ite.shared import knn_distances, estimate_d_temp2, estimate_i_alpha,\
                       estimate_d_temp3, volume_of_the_unit_ball,\
                       estimate_d_temp1


class BDKL_KnnK(InitKnnK, VerEqualDSubspaces):
    """ Kullback-Leibler divergence estimator using the kNN method (S={k}).
    
    Initialization is inherited from 'InitKnnK', verification comes from
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BDKL_KnnK()
    >>> co2 = ite.cost.BDKL_KnnK(knn_method='cKDTree', k=5, eps=0.1)
    >>> co3 = ite.cost.BDKL_KnnK(k=4)

    """
    
    def estimation(self, y1, y2):
        """ Estimate KL divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated KL divergence.
            
        References
        ----------
        Fernando Perez-Cruz. Estimation of Information Theoretic Measures
        for Continuous Random Variables. Advances in Neural Information
        Processing Systems (NIPS), pp. 1257-1264, 2008.
        
        Nikolai Leonenko, Luc Pronzato, and Vippal Savani. A class of
        Renyi information estimators for multidimensional densities.
        Annals of Statistics, 36(5):2153-2182, 2008.
        
        Quing Wang, Sanjeev R. Kulkarni, and Sergio Verdu. Divergence 
        estimation for multidimensional densities via k-nearest-neighbor 
        distances. IEEE Transactions on Information Theory, 55:2392-2405,
        2009.
        
        Examples
        --------
        d = co.estimation(y1,y2)

        """
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        # sizes:
        num_of_samples1, dim = y1.shape
        num_of_samples2 = y2.shape[0]
        
        # computation:
        distances_y1y1 = knn_distances(y1, y1, True, self.knn_method,
                                       self.k, self.eps, 2)[0]
        distances_y2y1 = knn_distances(y2, y1, False, self.knn_method,
                                       self.k, self.eps, 2)[0]
        d = dim * mean(log(distances_y2y1[:, -1] /
                           distances_y1y1[:, -1])) + \
            log(num_of_samples2/(num_of_samples1-1))
          
        return d


class BDEnergyDist(InitX, VerEqualDSubspaces):
    """ Energy distance estimator using pairwise distances of the samples.
    
    Initialization is inherited from 'InitX', verification comes from
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    Examples
    --------
    >>> import ite
    >>> co = ite.cost.BDEnergyDist()

    """
    
    def estimation(self, y1, y2):
        """ Estimate energy distance.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated energy distance.
            
        References
        ----------
        Gabor J. Szekely and Maria L. Rizzo. A new test for multivariate 
        normality. Journal of Multivariate Analysis, 93:58-80, 2005.
        (metric space of negative type)
        
        Gabor J. Szekely and Maria L. Rizzo. Testing for equal
        distributions in high dimension. InterStat, 5, 2004. (R^d)
        
        Ludwig Baringhaus and C. Franz. On a new multivariate 
        two-sample test. Journal of Multivariate Analysis, 88, 190-206, 
        2004. (R^d)
        
        Lev Klebanov. N-Distances and Their Applications. Charles 
        University, Prague, 2005. (N-distance)
        
        A. A. Zinger and A. V. Kakosyan and L. B. Klebanov. A 
        characterization of distributions by mean values of statistics 
        and certain probabilistic metrics. Journal of Soviet 
        Mathematics, 1992 (N-distance, general case).

        Examples
        --------
        d = co.estimation(y1, y2)

        """
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        # Euclidean distances:
        num_of_samples1, num_of_samples2 = y1.shape[0], y2.shape[0]
        mean_dist_y1y1 = 2 * sum(pdist(y1)) / num_of_samples1**2
        mean_dist_y2y2 = 2 * sum(pdist(y2)) / num_of_samples2**2
        mean_dist_y1y2 = mean(cdist(y1, y2))
        
        d = 2 * mean_dist_y1y2 - mean_dist_y1y1 - mean_dist_y2y2
        
        return d


class BDBhattacharyya_KnnK(InitKnnK, VerEqualDSubspaces):
    """ Bhattacharyya distance estimator using the kNN method (S={k}).

    Partial initialization comes from 'InitKnnK', verification is
    inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    """
    
    def __init__(self, mult=True, knn_method='cKDTree', k=3, eps=0,
                 pxdx=True):
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
        pxdx : boolean, optional
               If pxdx == True, then we rewrite the Bhattacharyya distance
               as \int p^{1/2}(x)q^{1/2}(x)dx = \int p^{-1/2}(x)q^{1/2}(x)
               p(x)dx. [p(x)dx] Else, the Bhattacharyya distance is
               rewritten as \int p^{1/2}(x)q^{1/2}(x)dx =
               \int q^{-1/2}(x)p^{1/2}(x) q(x)dx. [q(x)dx]
             
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.BDBhattacharyya_KnnK()
        >>> co2 = ite.cost.BDBhattacharyya_KnnK(k=4)
        
        """
        
        # initialize with 'InitKnnK':
        super().__init__(mult=mult, knn_method=knn_method, k=k, eps=eps)

        # other attributes (pxdx,_a,_b):
        self.pxdx, self._a, self._b = pxdx, -1/2, 1/2
        
    def estimation(self, y1, y2):
        """ Estimate Bhattacharyya distance.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated Bhattacharyya distance.
            
        References
        ----------            
        Barnabas Poczos and Liang Xiong and Dougal Sutherland and Jeff 
        Schneider. Support Distribution Machines. Technical Report, 2012. 
        "http://arxiv.org/abs/1202.0302" (estimation of d_temp2)
            
        Examples
        --------
        d = co.estimation(y1,y2)

        """
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        if self.pxdx:
            d_ab = estimate_d_temp2(y1, y2, self)
        else:
            d_ab = estimate_d_temp2(y2, y1, self)
        # absolute() to avoid possible 'log(negative)' values due to the
        # finite number of samples:
        d = -log(absolute(d_ab))

        return d


class BDBregman_KnnK(InitKnnKAlpha, VerEqualDSubspaces):
    """ Bregman distance estimator using the kNN method (S={k}).

    Initialization comes from 'InitKnnKAlpha', verification is inherited
    from 'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BDBregman_KnnK()
    >>> co2 = ite.cost.BDBregman_KnnK(alpha=0.9, k=5, eps=0.1)
    
    """

    def estimation(self, y1, y2):
        """ Estimate Bregman distance.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated Bregman distance.
            
        References
        ----------            
        Nikolai Leonenko, Luc Pronzato, and Vippal Savani. A class of
        Renyi information estimators for multidimensional densities.
        Annals of Statistics, 36(5):2153-2182, 2008.
        
        Imre Csiszar. Generalized projections for non-negative functions.
        Acta Mathematica Hungarica, 68:161-185, 1995.
        
        Lev M. Bregman. The relaxation method of finding the common points
        of convex sets and its application to the solution of problems in
        convex programming. USSR Computational Mathematics and
        Mathematical Physics, 7:200-217, 1967.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        i_alpha_y1 = estimate_i_alpha(y1, self)
        i_alpha_y2 = estimate_i_alpha(y2, self)
        d_temp3 = estimate_d_temp3(y1, y2, self)
    
        d = i_alpha_y2 + i_alpha_y1 / (self.alpha - 1) -\
            self.alpha / (self.alpha - 1) * d_temp3
        
        return d


class BDChi2_KnnK(InitKnnK, VerEqualDSubspaces):
    """ Chi-square distance estimator using the kNN method (S={k}).

    Partial initialization comes from 'InitKnnK', verification is
    inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    """
    
    def __init__(self, mult=True, knn_method='cKDTree', k=3, eps=0,
                 pxdx=True):
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
        pxdx : boolean, optional
               If pxdx == True, then we rewrite the Pearson chi-square 
               divergence as \int p^2(x)q^{-1}(x)dx - 1 = 
               \int p^1(x)q^{-1}(x) p(x)dx - 1. [p(x)dx]
               Else, the Pearson chi-square divergence is rewritten as
               \int p^2(x)q^{-1}(x)dx - 1= \int q^{-2}(x)p^2(x) q(x)dx -1.
               [q(x)dx]
             
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.BDChi2_KnnK()
        >>> co2 = ite.cost.BDChi2_KnnK(k=4)
        
        """
        
        # initialize with 'InitKnnK':
        super().__init__(mult=mult, knn_method=knn_method, k=k, eps=eps)

        # other attributes (pxdx,_a,_b):
        self.pxdx = pxdx
        if pxdx:
            self._a, self._b = 1, -1
        else:
            self._a, self._b = -2, 2

    def estimation(self, y1, y2):
        """ Estimate Pearson chi-square divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated Pearson chi-square divergence.
            
        References
        ----------   
        Barnabas Poczos, Liang Xiong, Dougal Sutherland, and Jeff
        Schneider. Support distribution machines. Technical Report,
        Carnegie Mellon University, 2012. http://arxiv.org/abs/1202.0302.
        (estimation of d_temp2)
        
        Karl Pearson. On the criterion that a given system of deviations
        from the probable in the case of correlated system of variables is
        such that it can be reasonable supposed to have arisen from random
        sampling. Philosophical Magazine Series, 50:157-172, 1900.

        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        if self.pxdx:
            d = estimate_d_temp2(y1, y2, self) - 1
        else:
            d = estimate_d_temp2(y2, y1, self) - 1

        return d


class BDHellinger_KnnK(InitKnnK, VerEqualDSubspaces):
    """ Hellinger distance estimator using the kNN method (S={k}).

    Partial initialization comes from 'InitKnnK', verification is
    inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    """
    
    def __init__(self, mult=True, knn_method='cKDTree', k=3, eps=0,
                 pxdx=True):
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
        pxdx : boolean, optional
               If pxdx == True, then we rewrite the Pearson chi-square 
               divergence as \int p^{1/2}(x)q^{1/2}(x)dx = 
               \int p^{-1/2}(x)q^{1/2}(x) p(x)dx. [p(x)dx]
               Else, the Pearson chi-square divergence is rewritten as
               \int p^{1/2}(x)q^{1/2}(x)dx =
               \int q^{-1/2}(x)p^{1/2}(x) q(x)dx. [q(x)dx]
             
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.BDHellinger_KnnK()
        >>> co2 = ite.cost.BDHellinger_KnnK(k=4)
        
        """
        
        # initialize with 'InitKnnK':
        super().__init__(mult=mult, knn_method=knn_method, k=k, eps=eps)

        # other attributes (pxdx,_a,_b):
        self.pxdx, self._a, self._b = pxdx, -1/2, 1/2

    def estimation(self, y1, y2):
        """ Estimate Hellinger distance.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated Hellinger distance.
            
        References
        ----------   
        Barnabas Poczos, Liang Xiong, Dougal Sutherland, and Jeff
        Schneider. Support distribution machines. Technical Report,
        Carnegie Mellon University, 2012. http://arxiv.org/abs/1202.0302.
        (estimation of d_temp2)
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        # D_ab (Bhattacharyya coefficient):
        if self.pxdx:
            d_ab = estimate_d_temp2(y1, y2, self)
        else:
            d_ab = estimate_d_temp2(y2, y1, self)
        # absolute() to avoid possible 'sqrt(negative)' values due to the
        # finite number of samples:
        d = sqrt(absolute(1 - d_ab))

        return d


class BDKL_KnnKiTi(InitKnnKiTi, VerEqualDSubspaces):
    """ Kullback-Leibler divergence estimator using the kNN method.

     In the kNN method: S_1={k_1}, S_2={k_2}; ki-s depend on the number of
     samples.
    
    Initialization is inherited from 'InitKnnKiTi', verification comes
    from 'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BDKL_KnnKiTi()
    >>> co2 = ite.cost.BDKL_KnnKiTi(knn_method='cKDTree', eps=0.1)
    
    """
    
    def estimation(self, y1, y2):
        """ Estimate KL divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated KL divergence.
            
        References
        ----------
        Quing Wang, Sanjeev R. Kulkarni, and Sergio Verdu. Divergence 
        estimation for multidimensional densities via k-nearest-neighbor 
        distances. IEEE Transactions on Information Theory, 55:2392-2405,
        2009.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        # sizes:
        num_of_samples1, dim = y1.shape
        num_of_samples2 = y2.shape[0]
        
        # ki-s depend on the number of samples:
        k1 = int(floor(sqrt(num_of_samples1)))
        k2 = int(floor(sqrt(num_of_samples2)))
    
        # computation:
        dist_k1_y1y1 = knn_distances(y1, y1, True, self.knn_method, k1,
                                     self.eps, 2)[0]
        dist_k2_y2y1 = knn_distances(y2, y1, False, self.knn_method, k2,
                                     self.eps, 2)[0]

        d = dim * mean(log(dist_k2_y2y1[:, -1] / dist_k1_y1y1[:, -1])) +\
            log(k1 / k2 * num_of_samples2 / (num_of_samples1 - 1))
        
        return d  


class BDL2_KnnK(InitKnnK, VerEqualDSubspaces):
    """ L2 divergence estimator using the kNN method (S={k}).
    
    Initialization is inherited from 'InitKnnK', verification comes from
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BDL2_KnnK()
    >>> co2 = ite.cost.BDL2_KnnK(knn_method='cKDTree', k=5, eps=0.1)
    >>> co3 = ite.cost.BDL2_KnnK(k=4)
    
    """
    
    def estimation(self, y1, y2):
        """ Estimate L2 divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated L2 divergence.
            
        References
        ----------
        Barnabas Poczos, Zoltan Szabo, Jeff Schneider. Nonparametric 
        divergence estimators for Independent Subspace Analysis. European 
        Signal Processing Conference (EUSIPCO), pages 1849-1853, 2011.
        
        Barnabas Poczos, Liang Xiong, Jeff Schneider. Nonparametric 
        Divergence: Estimation with Applications to Machine Learning on 
        Distributions. Uncertainty in Artificial Intelligence (UAI), 2011.

        Barnabas Poczos and Jeff Schneider. On the Estimation of
        alpha-Divergences. International Conference on Artificial
        Intelligence and Statistics (AISTATS), pages 609-617, 2011.

        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """

        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        # sizes:
        num_of_samples1, dim = y1.shape
        num_of_samples2 = y2.shape[0]
        
        c = volume_of_the_unit_ball(dim)
        dist_k_y1y1 = knn_distances(y1, y1, True, self.knn_method, self.k,
                                    self.eps, 2)[0][:, -1]
        dist_k_y2y1 = knn_distances(y2, y1, False, self.knn_method, self.k,
                                    self.eps, 2)[0][:, -1]

        term1 = \
            mean(dist_k_y1y1**(-dim)) * (self.k - 1) /\
            ((num_of_samples1 - 1) * c)
        term2 = \
            mean(dist_k_y2y1**(-dim)) * 2 * (self.k - 1) /\
            (num_of_samples2 * c)
        term3 = \
            mean((dist_k_y1y1**dim) / (dist_k_y2y1**(2 * dim))) *\
            (num_of_samples1 - 1) * (self.k - 2) * (self.k - 1) /\
            (num_of_samples2**2 * c * self.k)
        l2 = term1 - term2 + term3
        # absolute() to avoid possible 'sqrt(negative)' values due to the
        # finite number of samples:
        d = sqrt(absolute(l2))
         
        return d        


class BDRenyi_KnnK(InitKnnKAlpha, VerEqualDSubspaces):
    """ Renyi divergence estimator using the kNN method (S={k}).

    Initialization comes from 'InitKnnKAlpha', verification is inherited
    from 'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    The Renyi divergence (D_{R,alpha}) equals to the Kullback-Leibler 
    divergence (D) in limit: D_{R,alpha} -> D, as alpha -> 1.
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BDRenyi_KnnK()
    >>> co2 = ite.cost.BDRenyi_KnnK(alpha=0.9, k=5, eps=0.1)
    
    """

    def estimation(self, y1, y2):
        """ Estimate Renyi divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated Renyi divergence.
            
        References
        ----------            
        Barnabas Poczos, Zoltan Szabo, Jeff Schneider. Nonparametric 
        divergence estimators for Independent Subspace Analysis. European 
        Signal Processing Conference (EUSIPCO), pages 1849-1853, 2011.
        
        Barnabas Poczos, Jeff Schneider. On the Estimation of 
        alpha-Divergences. International conference on Artificial
        Intelligence and Statistics (AISTATS), pages 609-617, 2011.
        
        Barnabas Poczos, Liang Xiong, Jeff Schneider. Nonparametric 
        Divergence: Estimation with Applications to Machine Learning on 
        Distributions. Uncertainty in Artificial Intelligence (UAI), 2011.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        d_temp1 = estimate_d_temp1(y1, y2, self)
        d = log(d_temp1) / (self.alpha - 1)
        return d


class BDTsallis_KnnK(InitKnnKAlpha, VerEqualDSubspaces):
    """ Tsallis divergence estimator using the kNN method (S={k}).

    Initialization comes from 'InitKnnKAlpha', verification is inherited
    from 'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    The Tsallis divergence (D_{T,alpha}) equals to the Kullback-Leibler 
    divergence (D) in limit: D_{T,alpha} -> D, as alpha -> 1.
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BDTsallis_KnnK()
    >>> co2 = ite.cost.BDTsallis_KnnK(alpha=0.9, k=5, eps=0.1)
    
    """

    def estimation(self, y1, y2):
        """ Estimate Tsallis divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated Tsallis divergence.
            
        References
        ----------        
        Barnabas Poczos, Zoltan Szabo, Jeff Schneider. Nonparametric 
        divergence estimators for Independent Subspace Analysis. European 
        Signal Processing Conference (EUSIPCO), pages 1849-1853, 2011.
        
        Barnabas Poczos, Jeff Schneider. On the Estimation of 
        alpha-Divergences. International conference on Artificial
        Intelligence and Statistics (AISTATS), pages 609-617, 2011.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        d_temp1 = estimate_d_temp1(y1, y2, self)
        d = (d_temp1 - 1) / (self.alpha - 1)
        
        return d


class BDSharmaMittal_KnnK(InitKnnKAlphaBeta, VerEqualDSubspaces):
    """ Sharma-Mittal divergence estimator using the kNN method (S={k}).

    Initialization comes from 'InitKnnKAlphaBeta', verification is
    inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    The Sharma-Mittal divergence (D_{SM,alpha,beta}) equals to the
    1)Tsallis divergence (D_{T,alpha}): D_{SM,alpha,beta} = D_{T,alpha},
    if alpha = beta.
    2)Kullback-Leibler divergence (D): D_{SM,alpha,beta} -> D, as
    (alpha,beta) -> (1,1).
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BDSharmaMittal_KnnK()
    >>> co2 = ite.cost.BDSharmaMittal_KnnK(alpha=0.9, beta=0.7, k=5,\
                                           eps=0.1)
    
    """

    def estimation(self, y1, y2):
        """ Estimate Sharma-Mittal divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated Sharma-Mittal divergence.
            
        References
        ----------     
        Barnabas Poczos, Zoltan Szabo, Jeff Schneider. Nonparametric 
        divergence estimators for Independent Subspace Analysis. European 
        Signal Processing Conference (EUSIPCO), pages 1849-1853, 2011.
        
        Barnabas Poczos, Jeff Schneider. On the Estimation of 
        alpha-Divergences. International conference on Artificial
        Intelligence and Statistics (AISTATS), pages 609-617, 2011.
        
        Marco Massi. A step beyond Tsallis and Renyi entropies. Physics 
        Letters A, 338:217-224, 2005. (Sharma-Mittal divergence definition)
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        d_temp1 = estimate_d_temp1(y1, y2, self)
        
        d = (d_temp1**((1 - self.beta) / (1 - self.alpha)) - 1) /\
            (self.beta - 1)
       
        return d


class BDSymBregman_KnnK(InitKnnKAlpha, VerEqualDSubspaces):
    """ Symmetric Bregman distance estimator using the kNN method (S={k}).

    Initialization comes from 'InitKnnKAlpha', verification is inherited
    from 'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BDSymBregman_KnnK()
    >>> co2 = ite.cost.BDSymBregman_KnnK(alpha=0.9, k=5, eps=0.1)
    
    """

    def estimation(self, y1, y2):
        """ Estimate symmetric Bregman distance.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated symmetric Bregman distance.
            
        References
        ----------        
        Nikolai Leonenko, Luc Pronzato, and Vippal Savani. A class of
        Renyi information estimators for multidimensional densities.
        Annals of Statistics, 36(5):2153-2182, 2008.
        
        Imre Csiszar. Generalized projections for non-negative functions.
        Acta Mathematica Hungarica, 68:161-185, 1995.
        
        Lev M. Bregman. The relaxation method of finding the common points
        of convex sets and its application to the solution of problems in
        convex programming. USSR Computational Mathematics and
        Mathematical Physics, 7:200-217, 1967.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        i_alpha_y1 = estimate_i_alpha(y1, self)
        i_alpha_y2 = estimate_i_alpha(y2, self)
        
        d_temp3_y1y2 = estimate_d_temp3(y1, y2, self)
        d_temp3_y2y1 = estimate_d_temp3(y2, y1, self)
        
        d = (i_alpha_y1 + i_alpha_y2 - d_temp3_y1y2 - d_temp3_y2y1) /\
            (self.alpha - 1)
             
        return d


class BDMMD_UStat(InitKernel, VerEqualDSubspaces):
    """ MMD (maximum mean discrepancy) estimator applying U-statistic.

    Initialization comes from 'InitKernel', verification is inherited from
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    Examples
    --------
    >>> import ite
    >>> from ite.cost.x_kernel import Kernel
    >>> co1 = ite.cost.BDMMD_UStat()
    >>> k2 = Kernel({'name': 'RBF','sigma': 1})
    >>> co2 = ite.cost.BDMMD_UStat(kernel=k2)
    >>> k3 = Kernel({'name': 'exponential','sigma': 1})
    >>> co3 = ite.cost.BDMMD_UStat(kernel=k3)
    >>> k4 = Kernel({'name': 'Cauchy','sigma': 1})
    >>> co4 = ite.cost.BDMMD_UStat(kernel=k4)
    >>> k5 = Kernel({'name': 'student','d': 1})
    >>> co5 = ite.cost.BDMMD_UStat(kernel=k5)
    >>> k6 = Kernel({'name': 'Matern3p2','l': 1})
    >>> co6 = ite.cost.BDMMD_UStat(kernel=k6)
    >>> k7 = Kernel({'name': 'Matern5p2','l': 1})
    >>> co7 = ite.cost.BDMMD_UStat(kernel=k7)
    >>> k8 = Kernel({'name': 'polynomial','exponent':2,'c': 1})
    >>> co8 = ite.cost.BDMMD_UStat(kernel=k8)
    >>> k9 = Kernel({'name': 'ratquadr','c': 1})
    >>> co9 = ite.cost.BDMMD_UStat(kernel=k9)
    >>> k10 = Kernel({'name': 'invmquadr','c': 1})
    >>> co10 = ite.cost.BDMMD_UStat(kernel=k10)
    
    """
    
    def estimation(self, y1, y2):
        """ Estimate MMD.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated value of MMD.
            
        References
        ----------
        Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard 
        Scholkopf and Alexander Smola. A Kernel Two-Sample Test. Journal
        of Machine Learning Research 13 (2012) 723-773.
        
        Alain Berlinet and Christine Thomas-Agnan. Reproducing Kernel
        Hilbert Spaces in Probability and Statistics. Kluwer, 2004. (mean
        embedding)

        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """

        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        num_of_samples1, num_of_samples2 = y1.shape[0], y2.shape[0]
        
        kernel = self.kernel
        ky1y1 = kernel.gram_matrix1(y1)
        ky2y2 = kernel.gram_matrix1(y2)
        ky1y2 = kernel.gram_matrix2(y1, y2)
        
        # make the diagonal zero in ky1y1 and ky2y2:
        ky1y1[arange(num_of_samples1), arange(num_of_samples1)] = 0
        ky2y2[arange(num_of_samples2), arange(num_of_samples2)] = 0

        term1 = sum(ky1y1) / (num_of_samples1 * (num_of_samples1-1))
        term2 = sum(ky2y2) / (num_of_samples2 * (num_of_samples2-1))
        term3 = -2 * sum(ky1y2) / (num_of_samples1 * num_of_samples2)

        # absolute(): to avoid 'sqrt(negative)' values:
        d = sqrt(absolute(term1 + term2 + term3))

        return d


class BDMMD_VStat(InitKernel, VerEqualDSubspaces):
    """ MMD (maximum mean discrepancy) estimator applying V-statistic.

    Initialization comes from 'InitKernel', verification is inherited from
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').

    Examples
    --------
    >>> import ite
    >>> from ite.cost.x_kernel import Kernel
    >>> co1 = ite.cost.BDMMD_VStat()
    >>> k2 = Kernel({'name': 'RBF','sigma': 1})
    >>> co2 = ite.cost.BDMMD_VStat(kernel=k2)
    >>> k3 = Kernel({'name': 'exponential','sigma': 1})
    >>> co3 = ite.cost.BDMMD_VStat(kernel=k3)
    >>> k4 = Kernel({'name': 'Cauchy','sigma': 1})
    >>> co4 = ite.cost.BDMMD_VStat(kernel=k4)
    >>> k5 = Kernel({'name': 'student','d': 1})
    >>> co5 = ite.cost.BDMMD_VStat(kernel=k5)
    >>> k6 = Kernel({'name': 'Matern3p2','l': 1})
    >>> co6 = ite.cost.BDMMD_VStat(kernel=k6)
    >>> k7 = Kernel({'name': 'Matern5p2','l': 1})
    >>> co7 = ite.cost.BDMMD_VStat(kernel=k7)
    >>> k8 = Kernel({'name': 'polynomial','exponent':2,'c': 1})
    >>> co8 = ite.cost.BDMMD_VStat(kernel=k8)
    >>> k9 = Kernel({'name': 'ratquadr','c': 1})
    >>> co9 = ite.cost.BDMMD_VStat(kernel=k9)
    >>> k10 = Kernel({'name': 'invmquadr','c': 1})
    >>> co10 = ite.cost.BDMMD_VStat(kernel=k10)

    """

    def estimation(self, y1, y2):
        """ Estimate MMD.

        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.

        Returns
        -------
        d : float
            Estimated value of MMD.

        References
        ----------
        Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard
        Scholkopf and Alexander Smola. A Kernel Two-Sample Test. Journal
        of Machine Learning Research 13 (2012) 723-773.

        Alain Berlinet and Christine Thomas-Agnan. Reproducing Kernel
        Hilbert Spaces in Probability and Statistics. Kluwer, 2004. (mean
        embedding)

        Examples
        --------
        d = co.estimation(y1,y2)

        """

        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        num_of_samples1, num_of_samples2 = y1.shape[0], y2.shape[0]

        kernel = self.kernel
        ky1y1 = kernel.gram_matrix1(y1)
        ky2y2 = kernel.gram_matrix1(y2)
        ky1y2 = kernel.gram_matrix2(y1, y2)

        term1 = sum(ky1y1) / (num_of_samples1**2)
        term2 = sum(ky2y2) / (num_of_samples2**2)
        term3 = -2 * sum(ky1y2) / (num_of_samples1 * num_of_samples2)

        # absolute(): to avoid 'sqrt(negative)' values:
        d = sqrt(absolute(term1 + term2 + term3))

        return d


class BDMMD_Online(InitKernel, VerEqualDSubspaces, VerEqualSampleNumbers,
                   VerEvenSampleNumbers):
    """ Online MMD (maximum mean discrepancy) estimator.

    Initialization comes from 'InitKernel', verification is inherited from
    'VerEqualDSubspaces', 'VerEqualSampleNumbers', 'VerEvenSampleNumbers' 
    (see 'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').

    Examples
    --------
    >>> import ite
    >>> from ite.cost.x_kernel import Kernel
    >>> co1 = ite.cost.BDMMD_Online()
    >>> k2 = Kernel({'name': 'RBF','sigma': 1})
    >>> co2 = ite.cost.BDMMD_Online(kernel=k2)
    >>> k3 = Kernel({'name': 'exponential','sigma': 1})
    >>> co3 = ite.cost.BDMMD_Online(kernel=k3)
    >>> k4 = Kernel({'name': 'Cauchy','sigma': 1})
    >>> co4 = ite.cost.BDMMD_Online(kernel=k4)
    >>> k5 = Kernel({'name': 'student','d': 1})
    >>> co5 = ite.cost.BDMMD_Online(kernel=k5)
    >>> k6 = Kernel({'name': 'Matern3p2','l': 1})
    >>> co6 = ite.cost.BDMMD_Online(kernel=k6)
    >>> k7 = Kernel({'name': 'Matern5p2','l': 1})
    >>> co7 = ite.cost.BDMMD_Online(kernel=k7)
    >>> k8 = Kernel({'name': 'polynomial', 'exponent': 2, 'c': 1})
    >>> co8 = ite.cost.BDMMD_Online(kernel=k8)
    >>> k9 = Kernel({'name': 'ratquadr','c': 1})
    >>> co9 = ite.cost.BDMMD_Online(kernel=k9)
    >>> k10 = Kernel({'name': 'invmquadr','c': 1})
    >>> co10 = ite.cost.BDMMD_Online(kernel=k10)
    
    """
    
    def estimation(self, y1, y2):
        """ Estimate MMD.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
             Assumption: number of samples1 = number of samples2 = even.
             
        Returns
        -------
        d : float
            Estimated value of MMD.
            
        References
        ----------
        Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard 
        Scholkopf and Alexander Smola. A Kernel Two-Sample Test. Journal
        of Machine Learning Research 13 (2012) 723-773.
        
        Alain Berlinet and Christine Thomas-Agnan. Reproducing Kernel
        Hilbert Spaces in Probability and Statistics. Kluwer, 2004. (mean
        embedding)

        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """

        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        self.verification_equal_sample_numbers(y1, y2)
        self.verification_even_sample_numbers(y1)
        # the order of 'verification_equal_sample_numbers' and 
        # 'verification_even_sample_numbers' is important here
        
        num_of_samples = y1.shape[0]  # = y2.shape[0]
    
        # y1i,y1j,y2i,y2j:
        y1i = y1[0:num_of_samples:2, :]
        y1j = y1[1:num_of_samples:2, :]
        y2i = y2[0:num_of_samples:2, :]
        y2j = y2[1:num_of_samples:2, :]
        
        kernel = self.kernel
    
        d = (kernel.sum(y1i, y1j) + kernel.sum(y2i, y2j) -
             kernel.sum(y1i, y2j) -
             kernel.sum(y1j, y2i)) / (num_of_samples / 2)
        
        return d


class BDMMD_UStat_IChol(InitEtaKernel, VerEqualDSubspaces):
    """ MMD estimator with U-statistic & incomplete Cholesky decomposition.

    MMD refers to maximum mean discrepancy.

    Initialization comes from 'InitKernel', verification is inherited from
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').

    Examples
    --------
    >>> import ite
    >>> from ite.cost.x_kernel import Kernel
    >>> eta = 1e-2
    >>> co1 = ite.cost.BDMMD_UStat_IChol()
    >>> co1b = ite.cost.BDMMD_UStat_IChol(eta=eta)
    >>> k2 = Kernel({'name': 'RBF','sigma': 1})
    >>> co2 = ite.cost.BDMMD_UStat_IChol(kernel=k2)
    >>> co2b = ite.cost.BDMMD_UStat_IChol(kernel=k2,eta=eta)
    >>> k3 = Kernel({'name': 'exponential','sigma': 1})
    >>> co3 = ite.cost.BDMMD_UStat_IChol(kernel=k3)
    >>> co3b = ite.cost.BDMMD_UStat_IChol(kernel=k3,eta=eta)
    >>> k4 = Kernel({'name': 'Cauchy','sigma': 1})
    >>> co4 = ite.cost.BDMMD_UStat_IChol(kernel=k4)
    >>> co4b = ite.cost.BDMMD_UStat_IChol(kernel=k4,eta=eta)
    >>> k5 = Kernel({'name': 'student','d': 1})
    >>> co5 = ite.cost.BDMMD_UStat_IChol(kernel=k5)
    >>> k6 = Kernel({'name': 'Matern3p2','l': 1})
    >>> co6 = ite.cost.BDMMD_UStat_IChol(kernel=k6)
    >>> k7 = Kernel({'name': 'Matern5p2','l': 1})
    >>> co7 = ite.cost.BDMMD_UStat_IChol(kernel=k7)
    >>> k8 = Kernel({'name': 'polynomial','exponent':2,'c': 1})
    >>> co8 = ite.cost.BDMMD_UStat_IChol(kernel=k8)
    >>> k9 = Kernel({'name': 'ratquadr','c': 1})
    >>> co9 = ite.cost.BDMMD_UStat_IChol(kernel=k9)
    >>> k10 = Kernel({'name': 'invmquadr','c': 1})
    >>> co10 = ite.cost.BDMMD_UStat_IChol(kernel=k10)

    """

    def estimation(self, y1, y2):
        """ Estimate MMD.

        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.

        Returns
        -------
        d : float
            Estimated value of MMD.

        References
        ----------
        Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard
        Scholkopf and Alexander Smola. A Kernel Two-Sample Test. Journal
        of Machine Learning Research 13 (2012) 723-773.

        Alain Berlinet and Christine Thomas-Agnan. Reproducing Kernel
        Hilbert Spaces in Probability and Statistics. Kluwer, 2004. (mean
        embedding)

        Examples
        --------
        d = co.estimation(y1,y2)

        """

        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        # sample numbers:
        num_of_samples1, num_of_samples2 = y1.shape[0], y2.shape[0]
        num_of_samples = num_of_samples1 + num_of_samples2  # total

        # low-rank approximation of the joint Gram matrix:
        kernel = self.kernel
        tolerance = self.eta * num_of_samples
        l = kernel.ichol(vstack((y1, y2)), tolerance)
        l1 = l[0:num_of_samples1]  # broadcast
        l2 = l[num_of_samples1:]   # broadcast
        e1l1 = sum(l1, axis=0)  # row vector
        e2l2 = sum(l2, axis=0)  # row vector

        term1 = \
            (dot(e1l1, e1l1) - sum(l1**2)) / \
            (num_of_samples1 * (num_of_samples1 - 1))
        term2 = \
            (dot(e2l2, e2l2) - sum(l2**2)) / \
            (num_of_samples2 * (num_of_samples2 - 1))
        term3 = -2 * dot(e1l1, e2l2) / (num_of_samples1 * num_of_samples2)

        # abs(): to avoid 'sqrt(negative)' values
        d = sqrt(abs(term1 + term2 + term3))

        return d


class BDMMD_VStat_IChol(InitEtaKernel, VerEqualDSubspaces):
    """ MMD estimator with V-statistic & incomplete Cholesky decomposition.

    MMD refers to maximum mean discrepancy.

    Initialization comes from 'InitKernel', verification is inherited from
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').

    Examples
    --------
    >>> import ite
    >>> from ite.cost.x_kernel import Kernel
    >>> co1 = ite.cost.BDMMD_VStat_IChol()
    >>> k2 = Kernel({'name': 'RBF','sigma': 1})
    >>> co2 = ite.cost.BDMMD_VStat_IChol(kernel=k2)
    >>> k3 = Kernel({'name': 'exponential','sigma': 1})
    >>> co3 = ite.cost.BDMMD_VStat_IChol(kernel=k3)
    >>> k4 = Kernel({'name': 'Cauchy','sigma': 1})
    >>> co4 = ite.cost.BDMMD_VStat_IChol(kernel=k4)
    >>> k5 = Kernel({'name': 'student','d': 1})
    >>> co5 = ite.cost.BDMMD_VStat_IChol(kernel=k5)
    >>> k6 = Kernel({'name': 'Matern3p2','l': 1})
    >>> co6 = ite.cost.BDMMD_VStat_IChol(kernel=k6)
    >>> k7 = Kernel({'name': 'Matern5p2','l': 1})
    >>> co7 = ite.cost.BDMMD_VStat_IChol(kernel=k7)
    >>> k8 = Kernel({'name': 'polynomial','exponent':2,'c': 1})
    >>> co8 = ite.cost.BDMMD_VStat_IChol(kernel=k8)
    >>> k9 = Kernel({'name': 'ratquadr','c': 1})
    >>> co9 = ite.cost.BDMMD_VStat_IChol(kernel=k9)
    >>> k10 = Kernel({'name': 'invmquadr','c': 1})
    >>> co10 = ite.cost.BDMMD_VStat_IChol(kernel=k10)

    """

    def estimation(self, y1, y2):
        """ Estimate MMD.

        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.

        Returns
        -------
        d : float
            Estimated value of MMD.

        References
        ----------
        Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard
        Scholkopf and Alexander Smola. A Kernel Two-Sample Test. Journal
        of Machine Learning Research 13 (2012) 723-773.

        Alain Berlinet and Christine Thomas-Agnan. Reproducing Kernel
        Hilbert Spaces in Probability and Statistics. Kluwer, 2004. (mean
        embedding)

        Examples
        --------
        d = co.estimation(y1,y2)

        """

        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        num_of_samples1, num_of_samples2 = y1.shape[0], y2.shape[0]
        num_of_samples = num_of_samples1 + num_of_samples2  # total

        # low-rank approximation of the joint Gram matrix:
        kernel = self.kernel
        tolerance = self.eta * num_of_samples
        l = kernel.ichol(vstack((y1, y2)), tolerance)
        # broadcasts; result:row vector:
        e1l1 = sum(l[:num_of_samples1], axis=0)
        e2l2 = sum(l[num_of_samples1:], axis=0)

        term1 = dot(e1l1, e1l1) / num_of_samples1**2
        term2 = dot(e2l2, e2l2) / num_of_samples2**2
        term3 = -2 * dot(e1l1, e2l2) / (num_of_samples1 * num_of_samples2)

        # abs(): to avoid 'sqrt(negative)' values
        d = sqrt(abs(term1 + term2 + term3))

        return d
