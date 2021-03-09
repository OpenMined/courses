""" Base entropy estimators on distributions. """

from scipy.special import psi, gamma
# from scipy.special import psi, gammaln
from numpy import floor, sqrt, concatenate, ones, sort, mean, log, absolute,\
                  exp, pi, sum, max

from ite.cost.x_initialization import InitKnnK, InitX, InitKnnKAlpha, \
                                      InitKnnKAlphaBeta, InitKnnSAlpha
from ite.cost.x_verification import VerOneDSignal
from ite.shared import volume_of_the_unit_ball, knn_distances, \
                       estimate_i_alpha, replace_infs_with_max


class BHShannon_KnnK(InitKnnK):
    """ Shannon differential entropy estimator using kNNs (S = {k}).

    Initialization is inherited from 'InitKnnK' (see
    'ite.cost.x_initialization.py').
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BHShannon_KnnK()
    >>> co2 = ite.cost.BHShannon_KnnK(knn_method='cKDTree', k=3, eps=0.1)
    >>> co3 = ite.cost.BHShannon_KnnK(k=5)
                                
    """

    def estimation(self, y):
        """ Estimate Shannon entropy.
        
        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
            One row of y corresponds to one sample.
    
        Returns
        -------
        h : float
            Estimated Shannon entropy.
            
        References
        ----------
        M. N. Goria, Nikolai N. Leonenko, V. V. Mergel, and P. L. Novi 
        Inverardi. A new class of random vector entropy estimators and its 
        applications in testing statistical hypotheses. Journal of 
        Nonparametric Statistics, 17: 277-297, 2005. (S={k})
        
        Harshinder Singh, Neeraj Misra, Vladimir Hnizdo, Adam Fedorowicz
        and Eugene Demchuk. Nearest neighbor estimates of entropy.
        American Journal of Mathematical and Management Sciences, 23,
        301-321, 2003. (S={k})
        
        L. F. Kozachenko and Nikolai N. Leonenko. A statistical estimate
        for the entropy of a random vector. Problems of Information
        Transmission, 23:9-16, 1987. (S={1})
        
        Examples
        --------
        h = co.estimation(y)

        """
        
        num_of_samples, dim = y.shape
        distances_yy = knn_distances(y, y, True, self.knn_method, self.k,
                                     self.eps, 2)[0]
        v = volume_of_the_unit_ball(dim)
        h = log(num_of_samples - 1) - psi(self.k) + log(v) + \
            dim * sum(log(distances_yy[:, self.k-1])) / num_of_samples

        return h


class BHShannon_SpacingV(InitX, VerOneDSignal):
    """ Shannon entropy estimator using Vasicek's spacing method.

    Initialization is inherited from 'InitX', verification comes from
    'VerOneDSignal' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').

    Examples
    --------
    >>> import ite
    >>> co = ite.cost.BHShannon_SpacingV()

    """

    def estimation(self, y):
        """ Estimate Shannon entropy.
        
        Parameters
        ----------
        y : (number of samples, 1)-ndarray (column vector)
            One coordinate of y corresponds to one sample.
    
        Returns
        -------
        h : float
            Estimated Shannon entropy.
            
        References
        ----------
        Oldrich Vasicek. A test for normality based on sample entropy.
        Journal of the Royal Statistical Society, Series B, 38(1):54-59,
        1976.
        
        Examples
        --------
        h = co.estimation(y)

        """

        # verification:
        self.verification_one_d_signal(y)
        
        # estimation:
        num_of_samples = y.shape[0]  # y : Tx1
        m = int(floor(sqrt(num_of_samples)))
        y = sort(y, axis=0)
        y = concatenate((y[0] * ones((m, 1)), y, y[-1] * ones((m, 1))))
        diffy = y[2*m:] - y[:num_of_samples]
        h = mean(log(num_of_samples / (2*m) * diffy))
        
        return h        


class BHRenyi_KnnK(InitKnnKAlpha):
    """ Renyi entropy estimator using the kNN method (S={k}). 
    
    Initialization comes from 'InitKnnKAlpha' (see
    'ite.cost.x_initialization.py').
    
    Notes
    -----
    The Renyi entropy (H_{R,alpha}) equals to the Shannon differential (H) 
    entropy in limit: H_{R,alpha} -> H, as alpha -> 1.
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BHRenyi_KnnK()
    >>> co2 = ite.cost.BHRenyi_KnnK(knn_method='cKDTree', k=4, eps=0.01, \
                                   alpha=0.9)
    >>> co3 = ite.cost.BHRenyi_KnnK(k=5, alpha=0.9)

    """
    
    def estimation(self, y):
        """ Estimate Renyi entropy.
        
        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
            One row of y corresponds to one sample.
    
        Returns
        -------
        h : float
            Estimated Renyi entropy.
            
        References
        ----------
        Nikolai Leonenko, Luc Pronzato, and Vippal Savani. A class of
        Renyi information estimators for multidimensional densities.
        Annals of Statistics, 36(5):2153-2182, 2008.
        
        Joseph E. Yukich. Probability Theory of Classical Euclidean 
        Optimization Problems, Lecture Notes in Mathematics, 1998, vol.
        1675.
        
        Examples
        --------
        h = co.estimation(y)

        """
        
        i_alpha = estimate_i_alpha(y, self)
        h = log(i_alpha) / (1 - self.alpha)
        
        return h


class BHTsallis_KnnK(InitKnnKAlpha):
    """ Tsallis entropy estimator using the kNN method (S={k}). 
    
    Initialization comes from 'InitKnnKAlpha' (see
    'ite.cost.x_initialization.py').
    
    Notes
    -----
    The Tsallis entropy (H_{T,alpha}) equals to the Shannon differential
    (H) entropy in limit: H_{T,alpha} -> H, as alpha -> 1.
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BHTsallis_KnnK()
    >>> co2 = ite.cost.BHTsallis_KnnK(knn_method='cKDTree', k=4,\
                                      eps=0.01, alpha=0.9)
    >>> co3 = ite.cost.BHTsallis_KnnK(k=5, alpha=0.9)
              
    """
    
    def estimation(self, y):
        """ Estimate Tsallis entropy.
        
        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
            One row of y corresponds to one sample.
    
        Returns
        -------
        h : float
            Estimated Tsallis entropy.
            
        References
        ----------
        Nikolai Leonenko, Luc Pronzato, and Vippal Savani. A class of
        Renyi information estimators for multidimensional densities.
        Annals of Statistics, 36(5):2153-2182, 2008.
        
        Examples
        --------
        h = co.estimation(y)  
        
        """
        
        i_alpha = estimate_i_alpha(y, self)
        h = (1 - i_alpha) / (self.alpha - 1)
        
        return h

        
class BHSharmaMittal_KnnK(InitKnnKAlphaBeta):
    """ Sharma-Mittal entropy estimator using the kNN method (S={k}). 
    
    Initialization comes from 'InitKnnKAlphaBeta' (see
    'ite.cost.x_initialization.py').
    
    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BHSharmaMittal_KnnK()
    >>> co2 = ite.cost.BHSharmaMittal_KnnK(knn_method='cKDTree', k=4,\
                                           eps=0.01, alpha=0.9, beta=0.9)
    >>> co3 = ite.cost.BHSharmaMittal_KnnK(k=5, alpha=0.9, beta=0.9)
    
    Notes
    -----
    The Sharma-Mittal entropy (H_{SM,alpha,beta}) equals to the 
    1)Renyi entropy (H_{R,alpha}): H_{SM,alpha,beta} -> H_{R,alpha}, as 
    beta -> 1.
    2)Tsallis entropy (H_{T,alpha}): H_{SM,alpha,beta} = H_{T,alpha}, if 
    alpha = beta.
    3)Shannon entropy (H): H_{SM,alpha,beta} -> H, as (alpha,beta) ->
    (1,1).
              
    """
    
    def estimation(self, y):
        """ Estimate Sharma-Mittal entropy.
        
        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
            One row of y corresponds to one sample.
    
        Returns
        -------
        h : float
            Estimated Sharma-Mittal entropy.
            
        References
        ----------
        Nikolai Leonenko, Luc Pronzato, and Vippal Savani. A class of
        Renyi information estimators for multidimensional densities.
        Annals of Statistics, 36(5):2153-2182, 2008. (i_alpha estimation)
        
        Joseph E. Yukich. Probability Theory of Classical Euclidean 
        Optimization Problems, Lecture Notes in Mathematics, 1998, vol.
        1675. (i_alpha estimation)
        
        Ethem Akturk, Baris Bagci, and Ramazan Sever. Is Sharma-Mittal
        entropy really a step beyond Tsallis and Renyi entropies?
        Technical report, 2007. http://arxiv.org/abs/cond-mat/0703277.
        (Sharma-Mittal entropy)
        
        Bhudev D. Sharma and Dharam P. Mittal. New nonadditive measures of 
        inaccuracy. Journal of Mathematical Sciences, 10:122-133, 1975. 
        (Sharma-Mittal entropy)
        
        Examples
        --------
        h = co.estimation(y)  
        
        """
        
        i_alpha = estimate_i_alpha(y, self)
        h = (i_alpha**((1-self.beta) / (1-self.alpha)) - 1) / (1 -
                                                               self.beta)
       
        return h


class BHShannon_MaxEnt1(InitX, VerOneDSignal):
    """ Maximum entropy distribution based Shannon entropy estimator.

    The used Gi functions are G1(x) = x exp(-x^2/2) and G2(x) = abs(x).
    
    Initialization is inherited from 'InitX', verification comes from
    'VerOneDSignal' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    Examples
    --------
    >>> import ite
    >>> co = ite.cost.BHShannon_MaxEnt1()

    """

    def estimation(self, y):
        """ Estimate Shannon entropy.
        
        Parameters
        ----------
        y : (number of samples, 1)-ndarray (column vector)
            One coordinate of y corresponds to one sample.
    
        Returns
        -------
        h : float
            Estimated Shannon entropy.
            
        References
        ----------
        Aapo Hyvarinen. New approximations of differential entropy for 
        independent component analysis and projection pursuit. In Advances
        in Neural Information Processing Systems (NIPS), pages 273-279,
        1997. (entropy approximation based on the maximum entropy
        distribution)
        
        Thomas M. Cover and Joy A. Thomas. Elements of Information Theory.
        John Wiley and Sons, New York, USA, 1991. (maximum entropy
        distribution)
        
        Examples
        --------
        h = co.estimation(y)  
            
        """

        # verification:
        self.verification_one_d_signal(y)
        
        # estimation:
        num_of_samples = y.shape[0] 
        
        # normalize 'y' to have zero mean and unit std:
        # step-1 [E=0, this step does not change the Shannon entropy of
        # the variable]:
        y = y - mean(y)

        # step-2 [std(Y) = 1]:
        s = sqrt(sum(y**2) / (num_of_samples - 1))
        # print(s)
        y /= s

        # we will take this scaling into account via the entropy
        # transformation rule [ H(wz) = H(z) + log(|w|) ] at the end:
        h_whiten = log(s)

        # h1, h2 -> h:
        h1 = (1 + log(2 * pi)) / 2  # =H[N(0,1)]
        # H2:
        k1 = 36 / (8 * sqrt(3) - 9)
        k2a = 1 / (2 - 6 / pi)
        h2 = \
            k1 * mean(y * exp(-y**2 / 2))**2 +\
            k2a * (mean(absolute(y)) - sqrt(2 / pi))**2
        h = h1 - h2
        
        # take into account the 'std=1' pre-processing:
        h += h_whiten
        
        return h        


class BHShannon_MaxEnt2(InitX, VerOneDSignal):
    """ Maximum entropy distribution based Shannon entropy estimator.

    The used Gi functions are G1(x) = x exp(-x^2/2) and G2(x) =
    exp(-x^2/2).
    
    Initialization is inherited from 'InitX', verification comes from
    'VerOneDSignal' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    Examples
    --------
    >>> import ite
    >>> co = ite.cost.BHShannon_MaxEnt2()

    """

    def estimation(self, y):
        """ Estimate Shannon entropy.
        
        Parameters
        ----------
        y : (number of samples, 1)-ndarray (column vector)
            One coordinate of y corresponds to one sample.
    
        Returns
        -------
        h : float
            Estimated Shannon entropy.
            
        References
        ----------
        Aapo Hyvarinen. New approximations of differential entropy for 
        independent component analysis and projection pursuit. In Advances
        in Neural Information Processing Systems (NIPS), pages 273-279,
        1997. (entropy approximation based on the maximum entropy
        distribution)
        
        Thomas M. Cover and Joy A. Thomas. Elements of Information Theory.
        John Wiley and Sons, New York, USA, 1991. (maximum entropy
        distribution)
        
        Examples
        --------
        h = co.estimation(y)  
            
        """

        # verification:
        self.verification_one_d_signal(y)
        
        # estimation:
        num_of_samples = y.shape[0] 
        
        # normalize 'y' to have zero mean and unit std:
        # step-1 [E=0, this step does not change the Shannon entropy of
        # the variable]:

        y = y - mean(y)

        # step-2 [std(y) = 1]:
        s = sqrt(sum(y**2) / (num_of_samples - 1))
        y /= s

        # we will take this scaling into account via the entropy
        # transformation rule [ H(wz) = H(z) + log(|w|) ] at the end:
        h_whiten = log(s)

        # h1, h2 -> h:
        h1 = (1 + log(2 * pi)) / 2  # =H[N(0,1)]
        # h2:
        k1 = 36 / (8 * sqrt(3) - 9)
        k2b = 24 / (16 * sqrt(3) - 27)
        h2 = \
            k1 * mean(y * exp(-y**2 / 2))**2 + \
            k2b * (mean(exp(-y**2 / 2)) - sqrt(1/2))**2

        h = h1 - h2
        
        # take into account the 'std=1' pre-processing:
        h += h_whiten
        
        return h        


class BHPhi_Spacing(InitX, VerOneDSignal):
    """ Phi entropy estimator using the spacing method.
    
    Partial initialization is inherited from 'InitX', verification comes
    from 'VerOneDSignal' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
    
    """
    
    def __init__(self, mult=True, w=lambda x: 1, phi=lambda x: x**2):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        w : function, optional 
            This weight function is used in the Phi entropy (default 
            is w=lambda x: 1, i.e., x-> 1).
        phi : function, optional
              This is the Phi function in the Phi entropy (default is 
              phi=lambda x: x**2, i.e. x->x**2)

        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.BHPhi_Spacing()
        >>> co2 = ite.cost.BHPhi_Spacing(phi=lambda x: x**2)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # other attributes:
        self.w = w
        self.phi = phi

    def estimation(self, y):
        """ Estimate Phi entropy.
        
        Parameters
        ----------
        y : (number of samples, 1)-ndarray (column vector)
            One coordinate of y corresponds to one sample.
    
        Returns
        -------
        h : float
            Estimated Phi entropy.
            
        References
        ----------
        Bert van Es. Estimating Functionals Related to a Density by a
        Class of Statistics Based on Spacings. Scandinavian Journal of
        Statistics, 19:61-72, 1992.
        
        Examples
        --------
        h = co.estimation(y)  
            
        """

        # verification:
        self.verification_one_d_signal(y)

        num_of_samples = y.shape[0]  # y : Tx1
        # m / num_of_samples -> 0, m / log(num_of_samples) -> infty a.s.,
        # m, num_of_samples -> infty:
        m = int(floor(sqrt(num_of_samples)))

        y = sort(y, axis=0)
        y1 = y[0:num_of_samples-m]  # y_{(0)},...,y_{(T-m-1)}
        y2 = y[m:]  # y_{m},...,y_{T-1}
        h = mean(self.phi((m / (num_of_samples + 1)) / (y2 - y1)) *
                 (self.w(y1) + self.w(y2))) / 2
       
        return h


class BHRenyi_KnnS(InitKnnSAlpha):
    """ Renyi entropy estimator using the generalized kNN method.

    In this case the kNN parameter is a set: S \subseteq {1,...,k}).
    Initialization comes from 'InitKnnSAlpha' (see
    'ite.cost.x_initialization.py').

    Notes
    -----
    The Renyi entropy (H_{R,alpha}) equals to the Shannon differential (H)
    entropy in limit: H_{R,alpha} -> H, as alpha -> 1.

    Examples
    --------
    >>> from numpy import array
    >>> import ite
    >>> co1 = ite.cost.BHRenyi_KnnS()
    >>> co2 = ite.cost.BHRenyi_KnnS(knn_method='cKDTree', k=4, eps=0.01, \
                                   alpha=0.9)
    >>> co3 = ite.cost.BHRenyi_KnnS(k=array([1,2,6]), eps=0.01, alpha=0.9)

    >>> co4 = ite.cost.BHRenyi_KnnS(k=5, alpha=0.9)

    """

    def estimation(self, y):
        """ Estimate Renyi entropy.

        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
            One row of y corresponds to one sample.

        Returns
        -------
        h : float
            Estimated Renyi entropy.

        References
        ----------
        David Pal, Barnabas Poczos, Csaba Szepesvari. Estimation of Renyi
        Entropy and Mutual Information Based on Generalized
        Nearest-Neighbor Graphs. Advances in Neural Information Processing
        Systems (NIPS), pages 1849-1857, 2010. (general S)

        Barnabas Poczos, Andras Lorincz. Independent Subspace Analysis
        Using k-Nearest Neighborhood Estimates. International Conference on
        Artificial Neural Networks (ICANN), pages 163-168, 2005. (S =
        {1,...,k})

        Examples
        --------
        h = co.estimation(y)

        """

        num_of_samples, dim = y.shape

        # compute length (L):
        distances_yy = knn_distances(y, y, True, self.knn_method,
                                     max(self.k), self.eps, 2)[0]
        gam = dim * (1 - self.alpha)
        # S = self.k:
        l = sum(replace_infs_with_max(distances_yy[:, self.k-1]**gam))
        # Note: if 'distances_yy[:, self.k-1]**gam' contains inf elements
        # (this may accidentally happen in small dimensions in case of
        #  large sample numbers, e.g., for d=1, T=10000), then the inf-s
        # are replaced with the maximal, non-inf element.

        # compute const = const(S):

        # Solution-1 (normal k):
        const = sum(gamma(self.k + 1 - self.alpha) / gamma(self.k))

        # Solution-2 (if k is 'extreme large', say self.k=180 [=>
        #            gamma(self.k)=inf], then use this alternative form of
        #            'const', after importing gammaln). Note: we used the
        #            'gamma(a) / gamma(b) = exp(gammaln(a) - gammaln(b))'
        #            identity.
        # const = sum(exp(gammaln(self.k + 1 - self.alpha) -
        #                 gammaln(self.k)))

        vol = volume_of_the_unit_ball(dim)
        const *= ((num_of_samples - 1) / num_of_samples * vol) ** \
                 (self.alpha - 1)

        h = log(l / (const * num_of_samples**self.alpha)) / (1 -
                                                             self.alpha)

        return h
