""" Base cross-quantity estimators. """

from scipy.special import psi
from numpy import mean, log

from ite.cost.x_initialization import InitKnnK
from ite.cost.x_verification import VerEqualDSubspaces
from ite.shared import volume_of_the_unit_ball, knn_distances


class BCCE_KnnK(InitKnnK, VerEqualDSubspaces):
    """ Cross-entropy estimator using the kNN method (S={k})
    
    Initialization is inherited from 'InitKnnK', verification comes from
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').

    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.BCCE_KnnK()
    >>> co2 = ite.cost.BCCE_KnnK(knn_method='cKDTree', k=4, eps=0.1)
    >>> co3 = ite.cost.BCCE_KnnK(k=4)
    
    """
    
    def estimation(self, y1, y2):
        """ Estimate cross-entropy.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        c : float
            Estimated cross-entropy.
            
        References
        ----------
        Nikolai Leonenko, Luc Pronzato, and Vippal Savani. A class of
        Renyi information estimators for multidimensional densities.
        Annals of Statistics, 36(5):2153-2182, 2008.
        
        Examples
        --------
        c = co.estimation(y1,y2)  

        """
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        num_of_samples2, dim = y2.shape  # number of samples, dimension
        
        # computation:
        v = volume_of_the_unit_ball(dim)
        distances_y2y1 = knn_distances(y2, y1, False, self.knn_method,
                                       self.k, self.eps, 2)[0]
        c = log(v) + log(num_of_samples2) - psi(self.k) + \
            dim * mean(log(distances_y2y1[:, -1]))
            
        return c
