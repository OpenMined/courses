""" Meta mutual information estimators. """

from numpy.random import rand
from numpy import ones

from ite.cost.x_initialization import InitX, InitAlpha
from ite.cost.x_verification import VerCompSubspaceDims, VerOneDSubspaces,\
                                    VerSubspaceNumberIsK
from ite.cost.x_factory import co_factory
from ite.shared import joint_and_product_of_the_marginals_split,\
                       copula_transformation


class MIShannon_DKL(InitX, VerCompSubspaceDims):
    """ Shannon mutual information estimator based on KL divergence.
    
    The estimation is based on the relation I(y^1,...,y^M) = 
    D(f_y,\prod_{m=1}^M f_{y^m}), where I is the Shannon mutual
    information, D is the Kullback-Leibler divergence.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerCompSubspaceDims' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
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
                     estimator (default is 'BDKL_KnnK').
        kl_co_pars : dictionary, optional
                     Parameters for the KL divergence estimator (default
                     is None (=> {}); in this case the default parameter
                     values of the KL divergence estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MIShannon_DKL()
        >>> co2 = ite.cost.MIShannon_DKL(kl_co_name='BDKL_KnnK')
        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 4, 'eps': 0.1}
        >>> co3 = ite.cost.MIShannon_DKL(kl_co_name='BDKL_KnnK',\
                                         kl_co_pars=dict_ch)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the KL divergence estimator:
        kl_co_pars = kl_co_pars or {}
        kl_co_pars['mult'] = mult  # guarantee this property
        self.kl_co = co_factory(kl_co_name, **kl_co_pars) 
        
    def estimation(self, y, ds):
        """ Estimate Shannon mutual information.
        
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
            Estimated Shannon mutual information.
            
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    
        
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        
        y1, y2 = joint_and_product_of_the_marginals_split(y, ds)
        i = self.kl_co.estimation(y1, y2)
        
        return i


class MIChi2_DChi2(InitX, VerCompSubspaceDims):
    """ Chi-square mutual information estimator based on chi^2 distance.
    
    The estimation is based on the relation I(y^1,...,y^M) = 
    D(f_y,\prod_{m=1}^M f_{y^m}), where I is the chi-square mutual 
    information, D is the chi^2 distance.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerCompSubspaceDims' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True, chi2_co_name='BDChi2_KnnK',
                 chi2_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        chi2_co_name : str, optional 
                       You can change it to any Pearson chi-square
                       divergence estimator (default is 'BDChi2_KnnK').
        chi2_co_pars : dictionary, optional
                      Parameters for the Pearson chi-square divergence 
                      estimator (default is None (=> {}); in this case the
                      default parameter values of the Pearson chi-square
                      divergence estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MIChi2_DChi2()
        >>> co2 = ite.cost.MIChi2_DChi2(chi2_co_name='BDChi2_KnnK')
        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 4, 'eps': 0.1}
        >>> co3 = ite.cost.MIChi2_DChi2(chi2_co_name='BDChi2_KnnK', \
                                        chi2_co_pars=dict_ch)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the chi-square divergence estimator:
        chi2_co_pars = chi2_co_pars or {}
        chi2_co_pars['mult'] = mult  # guarantee this property
        self.chi2_co = co_factory(chi2_co_name, **chi2_co_pars) 
        
    def estimation(self, y, ds):
        """ Estimate chi-square mutual information.
        
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
            Estimated chi-square mutual information.
            
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    
        
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        
        y1, y2 = joint_and_product_of_the_marginals_split(y, ds)
        i = self.chi2_co.estimation(y1, y2)
        
        return i


class MIL2_DL2(InitX, VerCompSubspaceDims):
    """ L2 mutual information estimator based on L2 divergence. 
    
    The estimation is based on the relation I(y^1,...,y^M) = 
    D(f_y,\prod_{m=1}^M f_{y^m}), where I is the L2 mutual 
    information, D is the L2 divergence.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerCompSubspaceDims' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True, l2_co_name='BDL2_KnnK', l2_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        l2_co_name : str, optional 
                     You can change it to any L2 divergence estimator
                     (default is 'BDL2_KnnK').
        l2_co_pars : dictionary, optional
                     Parameters for the L2 divergence estimator (default
                     is None (=> {}); in this case the default parameter
                     values of the L2 divergence estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MIL2_DL2()
        >>> co2 = ite.cost.MIL2_DL2(l2_co_name='BDL2_KnnK')
        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 2, 'eps': 0.1}
        >>> co3 = ite.cost.MIL2_DL2(l2_co_name='BDL2_KnnK',\
                                    l2_co_pars=dict_ch)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the L2 divergence estimator:
        l2_co_pars = l2_co_pars or {}
        l2_co_pars['mult'] = mult  # guarantee this property
        self.l2_co = co_factory(l2_co_name, **l2_co_pars)
        
    def estimation(self, y, ds):
        """ Estimate L2 mutual information.
        
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
            Estimated L2 mutual information.
            
        References
        ----------
        Barnabas Poczos, Zoltan Szabo, Jeff Schneider: Nonparametric 
        divergence estimators for Independent Subspace Analysis. European 
        Signal Processing Conference (EUSIPCO), pages 1849-1853, 2011.
        
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    
        
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        
        y1, y2 = joint_and_product_of_the_marginals_split(y, ds)
        i = self.l2_co.estimation(y1, y2)
        
        return i       


class MIRenyi_DR(InitAlpha, VerCompSubspaceDims):
    """ Renyi mutual information estimator based on Renyi divergence. 
    
    The estimation is based on the relation I(y^1,...,y^M) = 
    D(f_y,\prod_{m=1}^M f_{y^m}), where I is the Renyi mutual 
    information, D is the Renyi divergence.
    
    Partial initialization comes from 'InitAlpha', verification is from 
    'VerCompSubspaceDims' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True, alpha=0.99, renyi_co_name='BDRenyi_KnnK',
                 renyi_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        alpha : float, optional
                Parameter of the Renyi mutual information (default is
                0.99).
        renyi_co_name : str, optional 
                        You can change it to any Renyi divergence
                        estimator (default is 'BDRenyi_KnnK').
        renyi_co_pars : dictionary, optional
                        Parameters for the Renyi divergence estimator 
                        (default is None (=> {}); in this case the default
                        parameter values of the Renyi divergence estimator
                        are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MIRenyi_DR()
        >>> co2 = ite.cost.MIRenyi_DR(renyi_co_name='BDRenyi_KnnK')
        >>> co3 = ite.cost.MIRenyi_DR(renyi_co_name='BDRenyi_KnnK',\
                                      alpha=0.4)
        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 2, 'eps': 0.1}
        >>> co4 = ite.cost.MIRenyi_DR(mult=True,alpha=0.9,\
                                      renyi_co_name='BDRenyi_KnnK',\
                                      renyi_co_pars=dict_ch)
        
        """

        # initialize with 'InitAlpha':
        super().__init__(mult=mult, alpha=alpha)
        
        # initialize the Renyi divergence estimator:
        renyi_co_pars = renyi_co_pars or {}
        renyi_co_pars['mult'] = mult    # guarantee this property
        renyi_co_pars['alpha'] = alpha  # -||-
        self.renyi_co = co_factory(renyi_co_name, **renyi_co_pars)
        
    def estimation(self, y, ds):
        """ Estimate Renyi mutual information.
        
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
            Estimated Renyi mutual information.
            
        References
        ----------
        Barnabas Poczos, Zoltan Szabo, Jeff Schneider. Nonparametric 
        divergence estimators for Independent Subspace Analysis. European 
        Signal Processing Conference (EUSIPCO), pages 1849-1853, 2011.
        
        Barnabas Poczos, Jeff Schneider. On the Estimation of 
        alpha-Divergences. International Conference on Artificial
        Intelligence and Statistics (AISTATS), pages 609-617, 2011.
        
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    
        
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        
        y1, y2 = joint_and_product_of_the_marginals_split(y, ds)
        i = self.renyi_co.estimation(y1, y2)
        
        return i       


class MITsallis_DT(InitAlpha, VerCompSubspaceDims):
    """ Tsallis mutual information estimator based on Tsallis divergence. 
    
    The estimation is based on the relation I(y^1,...,y^M) = 
    D(f_y,\prod_{m=1}^M f_{y^m}), where I is the Tsallis mutual 
    information, D is the Tsallis divergence.
    
    Partial initialization comes from 'InitAlpha', verification is from 
    'VerCompSubspaceDims' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True, alpha=0.99,
                 tsallis_co_name='BDTsallis_KnnK', tsallis_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        alpha : float, optional
                Parameter of the Renyi mutual information (default is
                0.99).
        tsallis_co_name : str, optional 
                          You can change it to any Tsallis divergence 
                          estimator (default is 'BDTsallis_KnnK').
        tsallis_co_pars : dictionary, optional
                          Parameters for the Tsallis divergence estimator 
                          (default is None (=> {}); in this case the
                          default parameter values of the Tsallis
                          divergence estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MITsallis_DT()
        >>> co2 = ite.cost.MITsallis_DT(tsallis_co_name='BDTsallis_KnnK')
        >>> co3 = ite.cost.MITsallis_DT(tsallis_co_name='BDTsallis_KnnK',\
                                        alpha=0.4)
        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 2, 'eps': 0.1}
        >>> co4 = ite.cost.MITsallis_DT(mult=True,alpha=0.9,\
                                        tsallis_co_name='BDTsallis_KnnK',\
                                        tsallis_co_pars=dict_ch)
        
        """

        # initialize with 'InitAlpha':
        super().__init__(mult=mult, alpha=alpha)
        
        # initialize the Tsallis divergence estimator:
        tsallis_co_pars = tsallis_co_pars or {}
        tsallis_co_pars['mult'] = mult    # guarantee this property
        tsallis_co_pars['alpha'] = alpha  # -||-
        self.tsallis_co = co_factory(tsallis_co_name, **tsallis_co_pars)
        
    def estimation(self, y, ds):
        """ Estimate Tsallis mutual information.
        
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
            Estimated Tsallis mutual information.
            
        References
        ----------
        Barnabas Poczos, Zoltan Szabo, Jeff Schneider. Nonparametric 
        divergence estimators for Independent Subspace Analysis. European 
        Signal Processing Conference (EUSIPCO), pages 1849-1853, 2011.
        
        Barnabas Poczos, Jeff Schneider. On the Estimation of 
        alpha-Divergences. International Conference on Artificial
        Intelligence and Statistics (AISTATS), pages 609-617, 2011.
        
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    
        
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        
        y1, y2 = joint_and_product_of_the_marginals_split(y, ds)
        i = self.tsallis_co.estimation(y1, y2)
        
        return i       


class MIMMD_CopulaDMMD(InitX, VerCompSubspaceDims, VerOneDSubspaces):
    """ Copula and MMD based kernel dependency estimator.

    MMD stands for maximum mean discrepancy.
    
    The estimation is based on the relation I(Y_1,...,Y_d) = MMD(P_Z,P_U), 
    where (i) Z =[F_1(Y_1);...;F_d(Y_d)] is the copula transformation of
    Y; F_i is the cdf of Y_i, (ii) P_U is the uniform distribution on
    [0,1]^d, (iii) dim(Y_1) = ... = dim(Y_d) = 1.
        
    Partial initialization comes from 'InitX', verification is from 
    'VerCompSubspaceDims' and 'VerOneDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True, mmd_co_name='BDMMD_UStat',
                 mmd_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        mmd_co_name : str, optional 
                      You can change it to any MMD estimator (default is 
                      'BDMMD_UStat').
        mmd_co_pars : dictionary, optional
                      Parameters for the MMD estimator (default is None
                      (=> {}); in this case the default parameter values
                      of the MMD estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> from ite.cost.x_kernel import Kernel
        >>> co1 = ite.cost.MIMMD_CopulaDMMD()
        >>> co2 = ite.cost.MIMMD_CopulaDMMD(mmd_co_name='BDMMD_UStat')
        >>> dict_ch = {'kernel': Kernel({'name': 'RBF','sigma': 0.1})}
        >>> co3 = ite.cost.MIMMD_CopulaDMMD(mmd_co_name='BDMMD_UStat',\
                                            mmd_co_pars=dict_ch)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the MMD estimator:
        mmd_co_pars = mmd_co_pars or {}
        mmd_co_pars['mult'] = mult  # guarantee this property
        self.mmd_co = co_factory(mmd_co_name, **mmd_co_pars) 
        
    def estimation(self, y, ds=None):
        """ Estimate copula and MMD based kernel dependency.
        
        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector, vector of ones
             If ds is not given (ds=None), the vector of ones [ds =
             ones(y.shape[1],dtype='int')] is emulated inside the function.

        Returns
        -------
        i : float
            Estimated copula and MMD based kernel dependency.
           
        References
        ----------
        Barnabas Poczos, Zoubin Ghahramani, Jeff Schneider. Copula-based 
        Kernel Dependency Measures. International Conference on Machine 
        Learning (ICML), 2012.           
        
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    

        if ds is None:  # emulate 'ds = vector of ones'
            ds = ones(y.shape[1], dtype='int')

        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        self.verification_one_dimensional_subspaces(ds)
        
        z = copula_transformation(y)
        u = rand(z.shape[0], z.shape[1])
        
        i = self.mmd_co.estimation(z, u)

        return i


class MIRenyi_HR(InitAlpha, VerCompSubspaceDims, VerOneDSubspaces):
    """ Renyi mutual information estimator based on Renyi entropy.
    
    The estimation is based on the relation I_{alpha}(X) = -H_{alpha}(Z), 
    where Z =[F_1(X_1);...;F_d(X_d)] is the copula transformation of X, 
    F_i is the cdf of X_i; I_{alpha} is the Renyi mutual information, 
    H_{alpha} is the Renyi entropy.
    
    Partial initialization comes from 'InitAlpha', verification is from 
    'VerCompSubspaceDims' and 'VerOneDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
   
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
        alpha : float, \ne 1
                Parameter of the Renyi mutual information.
        renyi_co_name : str, optional 
                        You can change it to any Renyi entropy estimator 
                        (default is 'BHRenyi_KnnK').
        renyi_co_pars : dictionary, optional
                        Parameters for the Renyi entropy estimator
                        (default is None (=> {}); in this case the default
                        parameter values of the Renyi entropy estimator
                        are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MIRenyi_HR()
        >>> co2 = ite.cost.MIRenyi_HR(renyi_co_name='BHRenyi_KnnK')
        >>> dict_ch = {'k': 2, 'eps': 0.4}
        >>> co3 = ite.cost.MIRenyi_HR(renyi_co_name='BHRenyi_KnnK',\
                                      renyi_co_pars=dict_ch)
        
        """

        # initialize with 'InitAlpha':
        super().__init__(mult=mult, alpha=alpha)
        
        # initialize the Renyi entropy estimator:
        renyi_co_pars = renyi_co_pars or {}
        renyi_co_pars['mult'] = mult    # guarantee this property
        renyi_co_pars['alpha'] = alpha  # -||-
        self.renyi_co = co_factory(renyi_co_name, **renyi_co_pars) 
        
    def estimation(self, y, ds=None):
        """ Estimate Renyi mutual information.
        
        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.
        ds : int vector, vector of ones
             If ds is not given (ds=None), the vector of ones [ds = 
             ones(y.shape[1],dtype='int')] is emulated inside the function.
    
        Returns
        -------
        i : float
            Estimated Renyi mutual information.
            
        References
        ----------
        David Pal, Barnabas Poczos, Csaba Szepesvari. Estimation of Renyi 
        Entropy and Mutual Information Based on Generalized
        Nearest-Neighbor Graphs. Advances in Neural Information Processing
        Systems (NIPS), pages 1849-1857, 2010.
        
        Barnabas Poczos, Sergey Krishner, Csaba Szepesvari. REGO:
        Rank-based Estimation of Renyi Information using Euclidean Graph
        Optimization. International Conference on Artificial Intelligence
        and Statistics (AISTATS), pages 605-612, 2010.
            
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    
        
        if ds is None:  # emulate 'ds = vector of ones'
            ds = ones(y.shape[1], dtype='int')
            
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        self.verification_one_dimensional_subspaces(ds)
        
        z = copula_transformation(y)
        i = -self.renyi_co.estimation(z)
        
        return i


class MIShannon_HS(InitX, VerCompSubspaceDims):
    """ Shannon mutual information estimator based on Shannon entropy.
    
    The estimation is based on the relation I(y^1,...,y^M) = \sum_{m=1}^M
    H(y^m) - H([y^1,...,y^M]), where I is the Shannon mutual information,
    H is the Shannon entropy.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerCompSubspaceDims' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True, shannon_co_name='BHShannon_KnnK',
                 shannon_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        shannon_co_name : str, optional 
                          You can change it to any Shannon differential 
                          entropy estimator (default is 'BHShannon_KnnK').
        shannon_co_pars : dictionary, optional
                          Parameters for the Shannon differential entropy 
                          estimator (default is None (=> {}); in this case
                          the default parameter values of the Shannon
                          differential entropy estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MIShannon_HS()
        >>> co2 = ite.cost.MIShannon_HS(shannon_co_name='BHShannon_KnnK')
        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 4, 'eps': 0.1}
        >>> co3 = ite.cost.MIShannon_HS(shannon_co_name='BHShannon_KnnK',\
                                        shannon_co_pars=dict_ch)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the Shannon differential entropy estimator:
        shannon_co_pars = shannon_co_pars or {}
        shannon_co_pars['mult'] = True  # guarantee this property
        self.shannon_co = co_factory(shannon_co_name, **shannon_co_pars) 
        
    def estimation(self, y, ds):
        """ Estimate Shannon mutual information.
        
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
            Estimated Shannon mutual information.
            
        References
        ----------
        Thomas M. Cover, Joy A. Thomas. Elements of Information Theory,
        John Wiley and Sons, New York, USA (1991).
        
        Examples
        --------
        i = co.estimation(y,ds)  
            
        """    
        
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)

        # I = - H([y^1,...,y^M]):
        i = -self.shannon_co.estimation(y)

        # I = I + \sum_{m=1}^M H(y^m):
        idx_start = 0 
        for k in range(len(ds)):
            dim_k = ds[k]
            idx_stop = idx_start + dim_k
            # print("{0}:{1}".format(idx_start,idx_stop))
            i += self.shannon_co.estimation(y[:, idx_start:idx_stop])
            idx_start = idx_stop    

        return i


class MIDistCov_HSIC(InitX, VerCompSubspaceDims, VerSubspaceNumberIsK):
    """ Estimate distance covariance from HSIC.

    The estimation is based on the relation I(y^1,y^2;rho_1,rho_2) =
    2 HSIC(y^1,y^2;k), where HSIC stands for the Hilbert-Schmidt
    independence criterion, y = [y^1; y^2] and k = k_1 x k_2, where k_i-s
    generates rho_i-s, semimetrics of negative type used in distance
    covariance.

    Partial initialization comes from 'InitX', verification is from
    'VerCompSubspaceDims' and 'VerSubspaceNumberIsK' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').


    """

    def __init__(self, mult=True, hsic_co_name='BIHSIC_IChol',
                 hsic_co_pars=None):

        """ Initialize the estimator.

        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the
               estimation. 'False': estimation up to 'proportionality'.
               (default is True)
        hsic_co_name : str, optional
                       You can change it to any HSIC estimator
                       (default is 'BIHSIC_IChol').
        hsic_co_pars : dictionary, optional
                       Parameters for the HSIC estimator (default is
                       None (=> {}); in this case the default parameter
                       values of the HSIC estimator are used.

        Examples
        --------
        >>> import ite
        >>> from ite.cost.x_kernel import Kernel
        >>> co1 = ite.cost.MIDistCov_HSIC()
        >>> co2 = ite.cost.MIDistCov_HSIC(hsic_co_name='BIHSIC_IChol')
        >>> k =  Kernel({'name': 'RBF','sigma': 0.3})
        >>> dict_ch = {'kernel': k, 'eta': 1e-3}
        >>> co3 = ite.cost.MIDistCov_HSIC(hsic_co_name='BIHSIC_IChol',\
                                          hsic_co_pars=dict_ch)

        """

        # initialize with 'InitX':
        super().__init__(mult=mult)

        # initialize the HSIC estimator:
        hsic_co_pars = hsic_co_pars or {}
        hsic_co_pars['mult'] = mult  # guarantee this property
        self.hsic_co = co_factory(hsic_co_name, **hsic_co_pars)

    def estimation(self, y, ds):
        """ Estimate distance covariance.

        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
            One row of y corresponds to one sample.
        ds : int vector
             Dimensions of the individual subspaces in y; ds[i] = i^th
             subspace dimension. Length(ds) = 2.

        Returns
        -------
        i : float
            Estimated distance covariance.

        References
        ----------

        Examples
        --------
        i = co.estimation(y,ds)

        """

        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        self.verification_subspace_number_is_k(ds, 2)

        i = 2 * self.hsic_co.estimation(y, ds)

        return i
