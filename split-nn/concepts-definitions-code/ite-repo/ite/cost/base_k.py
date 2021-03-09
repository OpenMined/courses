""" Base kernel estimators on distributions. """

from ite.cost.x_initialization import InitKernel, InitKnnK, InitBagGram
from ite.cost.x_verification import VerEqualDSubspaces
from ite.shared import estimate_d_temp2
from numpy import mean

# scipy.spatial.distance.cdist is slightly slow; you can obtain some
# speed-up in case of larger dimensions by using
# ite.shared.cdist_large_dim:
# from ite.shared import cdist_large_dim


class BKProbProd_KnnK(InitKnnK, InitBagGram, VerEqualDSubspaces):
    """ Probability product kernel estimator using the kNN method (S={k}).

    Partial initialization comes from 'InitKnnK' and 'InitBagGram',
    verification is inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    """
    
    def __init__(self, mult=True, knn_method='cKDTree', k=3, eps=0, rho=2,
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
        rho: float, >0, optional
             Parameter of the probability product kernel (default is 2).
             Specially, for rho=1/2, one gets the Bhattacharyya kernel
             (also known as the Bhattacharyya coefficient, Hellinger
             affinity).
        pxdx : boolean, optional
               If pxdx == True, then we rewrite the probability product
               kernel as \int p^{rho}(x)q^{rho}(x)dx =
               \int p^{rho-1}(x)q^{rho}(x) p(x)dx. [p(x)dx]
               Else, the probability product kernel is rewritten as
               \int p^{rho}(x)q^{rho}(x)dx= \int q^{rho-1}(x)p^{rho}(x)
               q(x)dx. [q(x)dx]
             
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.BKProbProd_KnnK()
        >>> co2 = ite.cost.BKProbProd_KnnK(rho=0.5)
        >>> co3 = ite.cost.BKProbProd_KnnK(k=4, pxdx=False, rho=1.4)
        
        """
        
        # initialize with 'InitKnnK':
        super().__init__(mult=mult, knn_method=knn_method, k=k, eps=eps)

        # other attributes:
        self.rho, self.pxdx, self._a, self._b = rho, pxdx, rho-1, rho
        
    def estimation(self, y1, y2):
        """ Estimate probability product kernel.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        k : float
            Estimated probability product kernel.
            
        References
        ----------            
        Barnabas Poczos and Liang Xiong and Dougal Sutherland and 
        Jeff Schneider. Support Distribution Machines. Technical Report,
        2012. "http://arxiv.org/abs/1202.0302" (k-nearest neighbor based
        estimation of d_temp2)
        
        Tony Jebara, Risi Kondor, and Andrew Howard. Probability product 
        kernels. Journal of Machine Learning Research, 5:819-844, 2004. 
        (probability product kernels --specifically--> Bhattacharyya
        kernel)
    
        Anil K. Bhattacharyya. On a measure of divergence between two 
        statistical populations defined by their probability distributions. 
        Bulletin of the Calcutta Mathematical Society, 35:99-109, 1943. 
        (Bhattacharyya kernel)
            
        Examples
        --------
        k = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        if self.pxdx:
            k = estimate_d_temp2(y1, y2, self)
        else:
            k = estimate_d_temp2(y2, y1, self)
        
        return k


class BKExpected(InitKernel, InitBagGram, VerEqualDSubspaces):
    """ Estimator for the expected kernel.
    
    Initialization comes from 'InitKernel' and 'InitBagGram', verification
    is inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    Examples
    --------
    >>> import ite
    >>> from ite.cost.x_kernel import Kernel
    >>> co1 = ite.cost.BKExpected()
    >>> k2 = Kernel({'name': 'RBF','sigma': 1})
    >>> co2 = ite.cost.BKExpected(kernel=k2)
    >>> k3 = Kernel({'name': 'exponential','sigma': 1})
    >>> co3 = ite.cost.BKExpected(kernel=k3)
    >>> k4 = Kernel({'name': 'Cauchy','sigma': 1})
    >>> co4 = ite.cost.BKExpected(kernel=k4)
    >>> k5 = Kernel({'name': 'student','d': 1})
    >>> co5 = ite.cost.BKExpected(kernel=k5)
    >>> k6 = Kernel({'name': 'Matern3p2','l': 1})
    >>> co6 = ite.cost.BKExpected(kernel=k6)
    >>> k7 = Kernel({'name': 'Matern5p2','l': 1})
    >>> co7 = ite.cost.BKExpected(kernel=k7)
    >>> k8 = Kernel({'name': 'polynomial','exponent': 2,'c': 1})
    >>> co8 = ite.cost.BKExpected(kernel=k8)
    >>> k9 = Kernel({'name': 'ratquadr','c': 1})
    >>> co9 = ite.cost.BKExpected(kernel=k9)
    >>> k10 = Kernel({'name': 'invmquadr','c': 1})
    >>> co10 = ite.cost.BKExpected(kernel=k10)
    
    """

    def estimation(self, y1, y2):
        """ Estimate the value of the expected kernel.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        k : float
            Estimated value of the expected kernel.
            
        References
        ----------            
        Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, 
        Bernhard Scholkopf, and Alexander Smola. A kernel two-sample test. 
        Journal of Machine Learning Research, 13:723-773, 2012.
        
        Krikamol Muandet, Kenji Fukumizu, Francesco Dinuzzo, and Bernhard 
        Scholkopf. Learning from distributions via support measure
        machines. In Advances in Neural Information Processing Systems
        (NIPS), pages 10-18, 2011.
        
        Alain Berlinet and Christine Thomas-Agnan. Reproducing Kernel
        Hilbert Spaces in Probability and Statistics. Kluwer, 2004. (mean
        embedding)
        
        Thomas Gartner, Peter A. Flach, Adam Kowalczyk, and Alexander
        Smola. Multi-instance kernels. In International Conference on
        Machine Learning (ICML), pages 179-186, 2002.
        (multi-instance/set/ensemble kernel)
        
        David Haussler. Convolution kernels on discrete structures.
        Technical report, Department of Computer Science, University of
        California at Santa Cruz, 1999. (convolution kernel -spec-> set
        kernel)

            
        Examples
        --------
        k = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        kernel = self.kernel
        ky1y2 = kernel.gram_matrix2(y1, y2)
   
        k = mean(ky1y2)
        
        return k
