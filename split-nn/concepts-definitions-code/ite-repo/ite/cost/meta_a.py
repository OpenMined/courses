""" Meta association measure estimators. """

from numpy import sqrt, floor, ones

from ite.cost.x_initialization import InitX
from ite.cost.x_verification import VerOneDSubspaces, VerCompSubspaceDims
from ite.cost.x_factory import co_factory


class MASpearmanLT(InitX, VerOneDSubspaces, VerCompSubspaceDims):
    """ Estimate lower tail dependence based on conditional Spearman's rho.

    Partial initialization comes from 'InitX'; verification capabilities
    are inherited from 'VerOneDSubspaces' and 'VerCompSubspaceDims' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').

    """

    def __init__(self, mult=True,
                 spearman_cond_lt_co_name='BASpearmanCondLT',
                 spearman_cond_lt_co_pars=None):
        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        spearman_cond_lt_co_name : str, optional
                     You can change it to any conditional Spearman's rho
                     (of lower tail) estimator. (default is
                     'BASpearmanCondLT')
        spearman_cond_lt_co_pars : dictionary, optional
                     Parameters for the conditional Spearman's rho
                     estimator. (default is None (=> {}); in this case the
                     default parameter values of the conditional
                     Spearman's rho estimator are used)

        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MASpearmanLT()
        >>> co2 = ite.cost.MASpearmanLT(spearman_cond_lt_co_name=\
                                        'BASpearmanCondLT')

        """

        # initialize with 'InitX':
        spearman_cond_lt_co_pars = spearman_cond_lt_co_pars or {}
        super().__init__(mult=mult)

        # initialize the conditional Spearman's rho estimator:
        spearman_cond_lt_co_pars['mult'] = mult  # guarantee this property
        self.spearman_cond_lt_co = co_factory(spearman_cond_lt_co_name,
                                              **spearman_cond_lt_co_pars)

    def estimation(self, y, ds=None):
        """ Estimate lower tail dependence.

        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector, vector of ones
             ds[i] = 1 (for all i): the i^th subspace is one-dimensional.
             If ds is not given (ds=None), the vector of ones [ds =
             ones(y.shape[1],dtype='int')] is emulated inside the function.

        Returns
        -------
        a : float
            Estimated lower tail dependence.

        References
        ----------
        Friedrich Schmid and Rafael Schmidt. Multivariate conditional
        versions of Spearman's rho and related measures of tail
        dependence. Journal of Multivariate Analysis, 98:1123-1140, 2007.

        C. Spearman. The proof and measurement of association between two
        things. The American Journal of Psychology, 15:72-101, 1904.

        Examples
        --------
        a = co.estimation(y,ds)

        """

        if ds is None:  # emulate 'ds = vector of ones'
            ds = ones(y.shape[1], dtype='int')

        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        self.verification_one_dimensional_subspaces(ds)

        # p:
        num_of_samples = y.shape[0]
        k = int(floor(sqrt(num_of_samples)))
        self.spearman_cond_lt_co.p = k / num_of_samples  # set p

        a = self.spearman_cond_lt_co.estimation(y, ds)

        return a


class MASpearmanUT(InitX, VerOneDSubspaces, VerCompSubspaceDims):
    """ Estimate upper tail dependence based on conditional Spearman's rho.
    
    Partial initialization comes from 'InitX'; verification capabilities
    are inherited from 'VerOneDSubspaces' and 'VerCompSubspaceDims' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True,
                 spearman_cond_ut_co_name='BASpearmanCondUT',
                 spearman_cond_ut_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        spearman_cond_ut_co_name : str, optional 
                     You can change it to any conditional Spearman's rho 
                     (of upper tail) estimator. (default is
                     'BASpearmanCondUT')
        spearman_cond_ut_co_pars : dictionary, optional
                     Parameters for the conditional Spearman's rho
                     estimator. (default is None (=> {}); in this case the
                     default parameter values of the conditional
                     Spearman's rho estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MASpearmanUT()
        >>> co2 = ite.cost.MASpearmanUT(spearman_cond_ut_co_name=\
                                        'BASpearmanCondUT')


        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the conditional Spearman's rho estimator:
        spearman_cond_ut_co_pars = spearman_cond_ut_co_pars or {}
        spearman_cond_ut_co_pars['mult'] = mult  # guarantee this property
        self.spearman_cond_ut_co = co_factory(spearman_cond_ut_co_name,
                                              **spearman_cond_ut_co_pars) 

    def estimation(self, y, ds=None):
        """ Estimate upper tail dependence.
        
        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector, vector of ones
             ds[i] = 1 (for all i): the i^th subspace is one-dimensional.
             If ds is not given (ds=None), the vector of ones [ds =
             ones(y.shape[1],dtype='int')] is emulated inside the function.
    
        Returns
        -------
        a : float
            Estimated upper tail dependence.

        References
        ----------
        Friedrich Schmid and Rafael Schmidt. Multivariate conditional
        versions of Spearman's rho and related measures of tail
        dependence. Journal of Multivariate Analysis, 98:1123-1140, 2007.
        
        C. Spearman. The proof and measurement of association between two 
        things. The American Journal of Psychology, 15:72-101, 1904.
            
        Examples
        --------
        a = co.estimation(y,ds)  
            
        """    
        
        if ds is None:  # emulate 'ds = vector of ones'
            ds = ones(y.shape[1], dtype='int')

        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        self.verification_one_dimensional_subspaces(ds)        

        # p:    
        num_of_samples = y.shape[0]
        k = int(floor(sqrt(num_of_samples)))
        self.spearman_cond_ut_co.p = k / num_of_samples  # set p
        
        a = self.spearman_cond_ut_co.estimation(y, ds)
         
        return a
