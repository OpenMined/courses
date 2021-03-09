""" Meta entropy estimators. """

from numpy import mean, cov, log, pi, exp, array, min, max, prod
from numpy.random import multivariate_normal, rand
from scipy.linalg import det

from ite.cost.x_initialization import InitX, InitAlpha
from ite.cost.x_factory import co_factory


class MHShannon_DKLN(InitX):
    """ Shannon entropy estimator using a Gaussian auxiliary variable.

    The estimtion relies on H(Y) = H(G) - D(Y,G), where G is Gaussian
    [N(E(Y),cov(Y)] and D is the Kullback-Leibler divergence.
    
    Partial initialization comes from 'InitX' (see
    'ite.cost.x_initialization.py').
    
    """
    
    def __init__(self, mult=True, kl_co_name='BDKL_KnnK', kl_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        kl_co_name : str, optional 
                     You can change it to any Kullback-Leibler divergence 
                     estimator. (default is 'BDKL_KnnK')
        kl_co_pars : dictionary, optional
                     Parameters for the KL divergence estimator. (default
                     is None (=> {}); in this case the default parameter
                     values of the KL divergence estimator are used)

        --------
        >>> import ite
        >>> co1 = ite.cost.MHShannon_DKLN()
        >>> co2 = ite.cost.MHShannon_DKLN(kl_co_name='BDKL_KnnK')

        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 4, 'eps': 0.2}
        >>> co3 = ite.cost.MHShannon_DKLN(kl_co_name='BDKL_KnnK', \
                                          kl_co_pars=dict_ch)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the KL divergence estimator:
        kl_co_pars = kl_co_pars or {}
        kl_co_pars['mult'] = True  # guarantee this property
        self.kl_co = co_factory(kl_co_name, **kl_co_pars) 
        
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
        Quing Wang, Sanjeev R. Kulkarni, and Sergio Verdu. Universal
        estimation of information measures for analog sources. Foundations
        And Trends In Communications And Information Theory, 5:265-353,
        2009.
        
        Examples
        --------
        h = co.estimation(y,ds)

        """
        
        num_of_samples, dim = y.shape  # number of samples, dimension
        
        # estimate the mean and the covariance of y:
        m = mean(y, axis=0)
        c = cov(y, rowvar=False)  # 'rowvar=False': 1 row = 1 observation
        
        # entropy of N(m,c):
        if dim == 1: 
            det_c = c  # det(): 'expected square matrix' exception
            # multivariate_normal(): 'cov must be 2 dimensional and square'
            # exception:
            c = array([[c]])

        else:
            det_c = det(c)
            
        h_normal = 1/2 * log((2*pi*exp(1))**dim * det_c)

        # generate samples from N(m,c):
        y_normal = multivariate_normal(m, c, num_of_samples)
    
        h = h_normal - self.kl_co.estimation(y, y_normal)

        return h

 
class MHShannon_DKLU(InitX):
    """ Shannon entropy estimator using a uniform auxiliary variable.


    The estimation relies on H(y) = -D(y',u) + log(\prod_i(b_i-a_i)),
    where y\in U[a,b] = \times_{i=1}^d U[a_i,b_i], D is the
    Kullback-Leibler divergence, y' = linearly transformed version of y to
    [0,1]^d, and U is the uniform distribution on [0,1]^d.
    
    Partial initialization comes from 'InitX' (see
    'ite.cost.x_initialization.py').
    
    """
    
    def __init__(self, mult=True, kl_co_name='BDKL_KnnK', kl_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        kl_co_name : str, optional 
                     You can change it to any Kullback-Leibler divergence 
                     estimator. (default is 'BDKL_KnnK')
        kl_co_pars : dictionary, optional
                     Parameters for the KL divergence estimator. (default
                     is None (=> {}); in this case the default parameter
                     values of the KL divergence estimator are used)

        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MHShannon_DKLU()
        >>> co2 = ite.cost.MHShannon_DKLU(kl_co_name='BDKL_KnnK')

        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 5, 'eps': 0.3}
        >>> co3 = ite.cost.MHShannon_DKLU(kl_co_name='BDKL_KnnK', \
                                          kl_co_pars=dict_ch)

        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the KL divergence estimator:
        kl_co_pars = kl_co_pars or {}
        kl_co_pars['mult'] = mult  # guarantee this property
        self.kl_co = co_factory(kl_co_name, **kl_co_pars) 
        
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
            
        Examples
        --------
        h = co.estimation(y,ds)

        """ 
        
        # estimate the support (a,b) of y, transform y to [0,1]^d:
        a, b = min(y, axis=0), max(y, axis=0)
        y = y/(b-a) + a/(a-b)

        # generate samples from U[0,1]^d:
        u = rand(*y.shape)  # '*': seq unpacking
    
        h = - self.kl_co.estimation(y, u) + log(prod(b-a))
    
        return h


class MHTsallis_HR(InitAlpha):
    """ Tsallis entropy estimator from Renyi entropy.

    The estimation relies on H_{T,alpha} = (e^{H_{R,alpha}(1-alpha)} - 1) /
    (1-alpha), where H_{T,alpha} and H_{R,alpha} denotes the Tsallis and
    the Renyi entropy, respectively.
    
    Partial initialization comes from 'InitAlpha' see
    'ite.cost.x_initialization.py').

    Notes
    -----
    The Tsallis entropy (H_{T,alpha}) equals to the Shannon differential
    (H) entropy in limit: H_{T,alpha} -> H, as alpha -> 1.
    
    """
    
    def __init__(self, mult=True, alpha=0.99, renyi_co_name='BHRenyi_KnnK',
                 renyi_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        alpha : float, alpha \ne 1, optional
                alpha in the Tsallis entropy. (default is 0.99)
        renyi_co_name : str, optional 
                     You can change it to any Renyi entropy estimator.
                     (default is 'BHRenyi_KnnK')
        renyi_co_pars : dictionary, optional
                     Parameters for the Renyi entropy estimator. (default
                     is None (=> {}); in this case the default parameter
                     values of the Renyi entropy estimator are used)

        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MHTsallis_HR()
        >>> co2 = ite.cost.MHTsallis_HR(renyi_co_name='BHRenyi_KnnK')
        >>> co3 = ite.cost.MHTsallis_HR(alpha=0.9, \
                                       renyi_co_name='BHRenyi_KnnK')

        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 5, 'eps': 0.1}
        >>> co4 = ite.cost.MHTsallis_HR(alpha=0.9, \
                                       renyi_co_name='BHRenyi_KnnK', \
                                       renyi_co_pars=dict_ch)

        """

        # initialize with 'InitX':
        super().__init__(mult=mult, alpha=alpha)
        
        # initialize the Renyi entropy estimator:
        renyi_co_pars = renyi_co_pars or {}
        renyi_co_pars['mult'] = mult    # guarantee this property
        renyi_co_pars['alpha'] = alpha  # -||-
        self.renyi_co = co_factory(renyi_co_name, **renyi_co_pars)

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
            
        Examples
        --------
        h = co.estimation(y,ds)

        """

        # Renyi entropy:
        h = self.renyi_co.estimation(y)
        
        # transform Renyi entropy to Tsallis entropy:
        h = (exp(h * (1 - self.alpha)) - 1) / (1 - self.alpha)
    
        return h
