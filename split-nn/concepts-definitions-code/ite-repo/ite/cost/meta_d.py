""" Meta divergence estimators. """

from numpy import sqrt, floor, array, sum

from ite.cost.x_initialization import InitX, InitAlpha
from ite.cost.x_verification import VerEqualDSubspaces, \
                                    VerEqualSampleNumbers
from ite.cost.x_factory import co_factory
from ite.shared import mixture_distribution


class MDBlockMMD(InitX, VerEqualDSubspaces, VerEqualSampleNumbers):
    """ Block MMD estimator using average of U-stat. based MMD estimators.

    MMD stands for maximum mean discrepancy.

    Partial initialization comes from 'InitX', verification is from 
    'VerEqualDSubspaces', 'VerEqualSampleNumbers' (see
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
                      You can change it to any U-statistic based MMD 
                      estimator. (default is 'BDMMD_UStat')
        mmd_co_pars : dictionary, optional
                      Parameters for the U-statistic based MMD estimator  
                      (default is None (=> {}); in this case the default
                      parameter values of the U-statistic based MMD
                      estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> from ite.cost.x_kernel import Kernel
        >>> co1 = ite.cost.MDBlockMMD()
        >>> co2 = ite.cost.MDBlockMMD(mmd_co_name='BDMMD_UStat')
        >>> dict_ch = {'kernel': \
                        Kernel({'name': 'RBF','sigma': 0.1}), 'mult': True}
        >>> co3 = ite.cost.MDBlockMMD(mmd_co_name='BDMMD_UStat',\
                                      mmd_co_pars=dict_ch)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the U-statistic based MMD estimator:
        mmd_co_pars = mmd_co_pars or {}
        mmd_co_pars['mult'] = mult  # guarantee this property
        self.mmd_co = co_factory(mmd_co_name, **mmd_co_pars) 
        
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
        y : float
            Estimated MMD.
            
        References
        ----------
        Wojciech Zaremba, Arthur Gretton, and Matthew Blaschko. B-tests:
        Low variance kernel two-sample tests. In Advances in Neural
        Information Processing Systems (NIPS), pages 755-763, 2013.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        self.verification_equal_sample_numbers(y1, y2)
        
        num_of_samples = y1.shape[0]  # =y2.shape[0]
        b = int(floor(sqrt(num_of_samples)))  # size of a block
        num_of_blocks = int(floor(num_of_samples / b))

        d = 0
        for k in range(num_of_blocks):
            d += self.mmd_co.estimation(y1[k*b:(k+1)*b], y2[k*b:(k+1)*b])

        d /= num_of_blocks
        
        return d


class MDEnergyDist_DMMD(InitX, VerEqualDSubspaces):
    """ Energy distance estimator using MMD (maximum mean discrepancy).
    
    The estimation is based on the relation D(f_1,f_2;rho) = 
    2 [MMD(f_1,f_2;k)]^2, where k is a kernel that generates rho, a
    semimetric of negative type.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, mult=True, mmd_co_name='BDMMD_UStat_IChol',
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
                      'BDMMD_UStat_IChol').
        mmd_co_pars : dictionary, optional
                      Parameters for the MMD estimator (default is None
                      (=> {}); in this case the default parameter values
                      of the MMD estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> from ite.cost.x_kernel import Kernel
        >>> co1 = ite.cost.MDEnergyDist_DMMD()
        >>> co2 =\
                ite.cost.MDEnergyDist_DMMD(mmd_co_name='BDMMD_UStat_IChol')
        >>> dict_ch = {'kernel': \
                       Kernel({'name': 'RBF','sigma': 0.1}), 'eta': 1e-2}
        >>> co3 =\
               ite.cost.MDEnergyDist_DMMD(mmd_co_name='BDMMD_UStat_IChol',\
                                          mmd_co_pars=dict_ch)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the MMD estimator:
        mmd_co_pars = mmd_co_pars or {}
        mmd_co_pars['mult'] = mult  # guarantee this property
        self.mmd_co = co_factory(mmd_co_name, **mmd_co_pars) 

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
        Dino Sejdinovic, Arthur Gretton, Bharath Sriperumbudur, and Kenji 
        Fukumizu. Hypothesis testing using pairwise distances and
        associated kernels. International Conference on Machine Learning
        (ICML), pages 1111-1118, 2012. (semimetric space; energy distance
        <=> MMD, with a suitable kernel)
        
        Russell Lyons. Distance covariance in metric spaces. Annals of 
        Probability, 41:3284-3305, 2013. (energy distance, metric space of 
        negative type; pre-equivalence to MMD).
        
        Gabor J. Szekely and Maria L. Rizzo. A new test for multivariate 
        normality. Journal of Multivariate Analysis, 93:58-80, 2005.
        (energy distance; metric space of negative type)
        
        Gabor J. Szekely and Maria L. Rizzo. Testing for equal
        distributions in high dimension. InterStat, 5, 2004. (energy
        distance; R^d)
        
        Ludwig Baringhaus and C. Franz. On a new multivariate 
        two-sample test. Journal of Multivariate Analysis, 88, 190-206, 
        2004. (energy distance; R^d)

        Lev Klebanov. N-Distances and Their Applications. Charles 
        University, Prague, 2005. (N-distance)
        
        A. A. Zinger and A. V. Kakosyan and L. B. Klebanov. A 
        characterization of distributions by mean values of statistics 
        and certain probabilistic metrics. Journal of Soviet 
        Mathematics, 1992 (N-distance, general case).
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
            
        d = 2 * self.mmd_co.estimation(y1, y2)**2
        
        return d


class MDf_DChi2(InitX, VerEqualDSubspaces):
    """ f-divergence estimator based on Taylor expansion & chi^2 distance.
    
    Assumption: f convex and f(1) = 0.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """
    
    def __init__(self, hess=2, mult=True, chi_square_co_name='BDChi2_KnnK',
                 chi_square_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        hess : float, optional 
               =f^{(2)}(1), the second derivative of f at 1 (default is 2).
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
                estimation. 'False': estimation up to 'proportionality'.        
                (default is True)
        chi_square_co_name : str, optional 
                             You can change it to any Pearson chi square 
                             divergence estimator (default is
                             'BDChi2_KnnK').
        chi_square_co_pars : dictionary, optional
                             Parameters for the Pearson chi-square
                             divergence estimator (default is None
                             (=> {}); in this case the default parameter
                             values of the Pearson chi square divergence
                             estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MDf_DChi2(hess=2)
        >>> co2 = ite.cost.MDf_DChi2(hess=1,\
                                     chi_square_co_name='BDChi2_KnnK')
        >>> dict_ch = {'k': 6}
        >>> co3 = ite.cost.MDf_DChi2(hess=2,\
                                     chi_square_co_name='BDChi2_KnnK',\
                                     chi_square_co_pars=dict_ch)
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the chi^2 divergence estimator:
        chi_square_co_pars = chi_square_co_pars or {}
        chi_square_co_pars['mult'] = mult  # guarantee this property
        self.chi_square_co = co_factory(chi_square_co_name,
                                        **chi_square_co_pars)
        # other attributes (hess):
        self.hess = hess

    def estimation(self, y1, y2):
        """ Estimate f-divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated f-divergence.
            
        References
        ----------
        Frank Nielsen and Richard Nock. On the chi square and higher-order
        chi distances for approximating f-divergences. IEEE Signal
        Processing Letters, 2:10-13, 2014.
        
        Neil S. Barnett, Pietro Cerone, Sever Silvestru Dragomir, and A.
        Sofo. Approximating Csiszar f-divergence by the use of Taylor's
        formula with integral remainder. Mathematical Inequalities and
        Applications, 5:417-432, 2002.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        d = self.hess / 2 * self.chi_square_co.estimation(y1, y2)
        
        return d


class MDJDist_DKL(InitX, VerEqualDSubspaces):
    """ J distance estimator.

    J distance is also known as the symmetrised Kullback-Leibler
    divergence.
    
    The estimation is based on the relation D_J(f_1,f_2) = D(f_1,f_2) + 
    D(f_2,f_1), where D_J is the J distance and D denotes the
    Kullback-Leibler divergence.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
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
                     Parameters for the Kullback-Leibler divergence
                     estimator (default is None (=> {}); in this case the
                     default parameter values of the Kullback-Leibler
                     divergence estimator are used).
                      
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MDJDist_DKL()
        >>> co2 = ite.cost.MDJDist_DKL(kl_co_name='BDKL_KnnK')
        >>> co3 = ite.cost.MDJDist_DKL(kl_co_name='BDKL_KnnK',\
                                       kl_co_pars={'k': 6})
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the KL divergence estimator:
        kl_co_pars = kl_co_pars or {}
        kl_co_pars['mult'] = mult  # guarantee this property
        self.kl_co = co_factory(kl_co_name, **kl_co_pars)
    
    def estimation(self, y1, y2):
        """ Estimate J distance.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated J distance.
            
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        d = self.kl_co.estimation(y1, y2) + self.kl_co.estimation(y2, y1)
        
        return d


class MDJR_HR(InitAlpha, VerEqualDSubspaces):
    """ Jensen-Renyi divergence estimator based on Renyi entropy.
    
    The estimation is based on the relation D_JR(f_1,f_2) = 
    D_{JR,alpha}(f_1,f_2) = H_{R,alpha}(w1*y^1+w2*y^2) - 
    [w1*H_{R,alpha}(y^1) + w2*H_{R,alpha}(y^2)], where y^i has density f_i 
    (i=1,2), w1*y^1+w2*y^2 is the mixture distribution of y^1 and y^2 with 
    w1, w2 positive weights, D_JR is the Jensen-Renyi divergence,
    H_{R,alpha} denotes the Renyi entropy.
    
    
    Partial initialization comes from 'InitAlpha', verification is from 
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """

    def __init__(self, mult=True, alpha=0.99, w=array([1/2, 1/2]),
                 renyi_co_name='BHRenyi_KnnK', renyi_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
                estimation. 'False': estimation up to 'proportionality'.        
                (default is True)
        alpha : float, \ne 1, optional
                Parameter of the Jensen-Renyi divergence (default is 0.99).
        w : ndarray, w = [w1,w2], w_i > 0, w_2 > 0, w1 + w2 = 1, optional.
            Parameters of the Jensen-Renyi divergence (default is w = 
            array([1/2,1/2]) )
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
        >>> co1 = ite.cost.MDJR_HR()
        >>> co2 = ite.cost.MDJR_HR(renyi_co_name='BHRenyi_KnnK', alpha=0.8)
        >>> co3 = ite.cost.MDJR_HR(renyi_co_name='BHRenyi_KnnK',\
                                   renyi_co_pars={'k': 6}, alpha=0.5,\
                                   w=array([1/4,3/4]))
        
        """

        # initialize with 'InitAlpha':
        super().__init__(mult=mult, alpha=alpha)
        
        # initialize the Renyi entropy estimator:
        renyi_co_pars = renyi_co_pars or {}
        renyi_co_pars['mult'] = mult   # guarantee this property
        renyi_co_pars['alpha'] = alpha  # -||-
        self.renyi_co = co_factory(renyi_co_name, **renyi_co_pars)
        
        # other attributes (w):
        # verification:
        if sum(w) != 1:
            raise Exception('sum(w) has to be 1!')
            
        if not all(w > 0):
            raise Exception('The coordinates of w have to be positive!')
            
        if len(w) != 2:
            raise Exception('The length of w has to be 2!')

        self.w = w

    def estimation(self, y1, y2):
        """ Estimate Jensen-Renyi divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated Jensen-Renyi divergence.
            
        References
        ----------
        A.B. Hamza and H. Krim. Jensen-Renyi divergence measure:
        theoretical and computational perspectives. In IEEE International
        Symposium on Information Theory (ISIT), page 257, 2003.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        w = self.w
        mixture_y = mixture_distribution((y1, y2), w)
        d = self.renyi_co.estimation(mixture_y) -\
            (w[0] * self.renyi_co.estimation(y1) +
             w[1] * self.renyi_co.estimation(y2))
     
        return d


class MDJT_HT(InitAlpha, VerEqualDSubspaces):
    """ Jensen-Tsallis divergence estimator based on Tsallis entropy.
    
    The estimation is based on the relation D_JT(f_1,f_2) = 
    D_{JT,alpha}(f_1,f_2) = H_{T,alpha}((y^1+y^2)/2) -
    [1/2*H_{T,alpha}(y^1) + 1/2*H_{T,alpha}(y^2)], where y^i has density
    f_i (i=1,2), (y^1+y^2)/2 is the mixture distribution of y^1 and y^2
    with 1/2-1/2 weights, D_JT is the Jensen-Tsallis divergence,
    H_{T,alpha} denotes the Tsallis entropy.
    
    Partial initialization comes from 'InitAlpha', verification is from 
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """

    def __init__(self, mult=True, alpha=0.99,
                 tsallis_co_name='BHTsallis_KnnK', tsallis_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
                estimation. 'False': estimation up to 'proportionality'.        
                (default is True)
        alpha : float, \ne 1, optional
                Parameter of the Jensen-Tsallis divergence (default is
                0.99).
        tsallis_co_name : str, optional 
                          You can change it to any Tsallis entropy
                          estimator (default is 'BHTsallis_KnnK').
        tsallis_co_pars : dictionary, optional
                          Parameters for the Tsallis entropy estimator
                          (default is None (=> {}); in this case the
                          default parameter values of  the Tsallis entropy
                          estimator are used).
                      
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MDJT_HT()
        >>> co2 = ite.cost.MDJT_HT(tsallis_co_name='BHTsallis_KnnK',\
                                   alpha=0.8)
        >>> co3 = ite.cost.MDJT_HT(tsallis_co_name='BHTsallis_KnnK',\
                                   tsallis_co_pars={'k':6}, alpha=0.5)
        
        """

        # initialize with 'InitAlpha':
        super().__init__(mult=mult, alpha=alpha)
        
        # initialize the Tsallis entropy estimator:
        tsallis_co_pars = tsallis_co_pars or {}
        tsallis_co_pars['mult'] = mult   # guarantee this property
        tsallis_co_pars['alpha'] = alpha  # -||-
        self.tsallis_co = co_factory(tsallis_co_name, **tsallis_co_pars)
        
    def estimation(self, y1, y2):
        """ Estimate Jensen-Tsallis divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated Jensen-Tsallis divergence.
            
        References
        ----------
        J. Burbea and C.R. Rao. On the convexity of some divergence
        measures based on entropy functions. IEEE Transactions on
        Information Theory, 28:489-495, 1982.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)
        
        w = array([1/2, 1/2])
        mixture_y = mixture_distribution((y1, y2), w)
        d = self.tsallis_co.estimation(mixture_y) -\
            (w[0] * self.tsallis_co.estimation(y1) +
             w[1] * self.tsallis_co.estimation(y2))
     
        return d


class MDJS_HS(InitX, VerEqualDSubspaces):
    """ Jensen-Shannon divergence estimator based on Shannon entropy.
    
    The estimation is based on the relation D_JS(f_1,f_2) =
    H(w1*y^1+w2*y^2) - [w1*H(y^1) + w2*H(y^2)], where y^i has density f_i
    (i=1,2), w1*y^1+w2*y^2 is the mixture distribution of y^1 and y^2 with
    w1, w2 positive weights, D_JS is the Jensen-Shannon divergence, H
    denotes the Shannon entropy.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """

    def __init__(self, mult=True, w=array([1/2, 1/2]),
                 shannon_co_name='BHShannon_KnnK', shannon_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
                estimation. 'False': estimation up to 'proportionality'.        
                (default is True)
        w : ndarray, w = [w1,w2], w_i > 0, w_2 > 0, w1 + w2 = 1, optional.
                Parameters of the Jensen-Shannon divergence (default is
                w = array([1/2,1/2]) )
        shannon_co_name : str, optional 
                          You can change it to any Shannon entropy
                          estimator (default is 'BHShannon_KnnK').
        shannon_co_pars : dictionary, optional
                          Parameters for the Shannon entropy estimator 
                          (default is None (=> {}); in this case the
                          default parameter values of the Shannon entropy
                          estimator are used).
                      
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MDJS_HS()
        >>> co2 = ite.cost.MDJS_HS(shannon_co_name='BHShannon_KnnK')
        >>> co3 = ite.cost.MDJS_HS(shannon_co_name='BHShannon_KnnK',\
                                   shannon_co_pars={'k':6,'eps':0.2},\
                                   w=array([1/4,3/4]))
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the Shannon entropy estimator:
        shannon_co_pars = shannon_co_pars or {}
        shannon_co_pars['mult'] = mult   # guarantee this property
        self.shannon_co = co_factory(shannon_co_name, **shannon_co_pars)
        
        # other attributes (w):
        # verification:
        if sum(w) != 1:
            raise Exception('sum(w) has to be 1!')
            
        if not all(w > 0):
            raise Exception('The coordinates of w have to be positive!')
            
        if len(w) != 2:
            raise Exception('The length of w has to be 2!')

        self.w = w

    def estimation(self, y1, y2):
        """ Estimate Jensen-Shannon divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated Jensen-Shannon divergence.
            
        References
        ----------
        Jianhua Lin. Divergence measures based on the Shannon entropy.
        IEEE Transactions on Information Theory, 37:145-151, 1991.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        w = self.w
        mixture_y = mixture_distribution((y1, y2), w)
        d = self.shannon_co.estimation(mixture_y) -\
            (w[0] * self.shannon_co.estimation(y1) +
             w[1] * self.shannon_co.estimation(y2))
     
        return d


class MDK_DKL(InitX, VerEqualDSubspaces):
    """ K divergence estimator based on Kullback-Leibler divergence.
    
    The estimation is based on the relation D_K(f_1,f_2) =
    D(f_1,(f_1+f_2)/2), where D_K is the K divergence, D denotes the
    Kullback-Leibler divergence.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
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
                     Parameters for the Kullback-Leibler divergence
                     estimator (default is None (=> {}); in this case the
                     default parameter values of the Kullback-Leibler
                     divergence estimator are used).
                      
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MDK_DKL()
        >>> co2 = ite.cost.MDK_DKL(kl_co_name='BDKL_KnnK')
        >>> co3 = ite.cost.MDK_DKL(kl_co_name='BDKL_KnnK',\
                                   kl_co_pars={'k':6,'eps':0.2})
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the Kullback-Leibler divergence estimator:
        kl_co_pars = kl_co_pars or {}
        kl_co_pars['mult'] = mult   # guarantee this property
        self.kl_co = co_factory(kl_co_name, **kl_co_pars)
        
    def estimation(self, y1, y2):
        """ Estimate K divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated K divergence.
            
        References
        ----------
        Jianhua Lin. Divergence measures based on the Shannon entropy.
        IEEE Transactions on Information Theory, 37:145-151, 1991.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        num_of_samples1, num_of_samples2 = y1.shape[0], y2.shape[0]
        
        # mixture of y1 and y2 with 1/2, 1/2 weights:
        w = array([1/2, 1/2])
        # samples to the mixture (second part of y1 and y2; =:y1m, y2m):
        # (max) number of samples to the mixture from y1 and from y2:
        num_of_samples1m = int(floor(num_of_samples1 / 2))  
        num_of_samples2m = int(floor(num_of_samples2 / 2)) 
        y1m = y1[num_of_samples1m:]  # broadcasting
        y2m = y2[num_of_samples2m:]  # broadcasting
        mixture_y = mixture_distribution((y1m, y2m), w)

        # with broadcasting:
        d = self.kl_co.estimation(y1[:num_of_samples1m], mixture_y)
     
        return d


class MDL_DKL(InitX, VerEqualDSubspaces):
    """ L divergence estimator based on Kullback-Leibler divergence.
    
    The estimation is based on the relation D_L(f_1,f_2) =
    D(f_1,(f_1+f_2)/2) + D(f_2,(f_1+f_2)/2), where D_L is the L divergence
    and D denotes the Kullback-Leibler divergence.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
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
                     Parameters for the Kullback-Leibler divergence
                     estimator (default is None (=> {}); in this case the
                     default parameter values of the Kullback-Leibler
                     divergence estimator are used).
                      
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MDL_DKL()
        >>> co2 = ite.cost.MDL_DKL(kl_co_name='BDKL_KnnK')
        >>> co3 = ite.cost.MDL_DKL(kl_co_name='BDKL_KnnK',\
                                   kl_co_pars={'k':6,'eps':0.2})
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the Kullback-Leibler divergence estimator:
        kl_co_pars = kl_co_pars or {}
        kl_co_pars['mult'] = mult  # guarantee this property
        self.kl_co = co_factory(kl_co_name, **kl_co_pars)
        
    def estimation(self, y1, y2):
        """ Estimate L divergence.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        d : float
            Estimated L divergence.
            
        References
        ----------
        Jianhua Lin. Divergence measures based on the Shannon entropy.
        IEEE Transactions on Information Theory, 37:145-151, 1991.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        num_of_samples1, num_of_samples2 = y1.shape[0], y2.shape[0]
        
        # mixture of y1 and y2 with 1/2, 1/2 weights:
        w = array([1/2, 1/2])
        # samples to the mixture (second part of y1 and y2; =:y1m, y2m):
        # (max) number of samples to the mixture from y1 and from y2:
        num_of_samples1m = int(floor(num_of_samples1 / 2))  
        num_of_samples2m = int(floor(num_of_samples2 / 2)) 
        y1m = y1[num_of_samples1m:]  # broadcasting
        y2m = y2[num_of_samples2m:]  # broadcasting
        mixture_y = mixture_distribution((y1m, y2m), w)

        # with broadcasting:
        d = self.kl_co.estimation(y1[:num_of_samples1m], mixture_y) +\
            self.kl_co.estimation(y2[:num_of_samples1m], mixture_y)
     
        return d


class MDSymBregman_DB(InitAlpha, VerEqualDSubspaces):
    """ Symmetric Bregman distance estimator from the nonsymmetric one.

    The estimation is based on the relation D_S =
    (D_NS(f1,f2) + D_NS (f2,f1)) / alpha, where D_S is the symmetric
    Bregman distance, D_NS is the nonsymmetric Bregman distance.
    
    Partial initialization comes from 'InitAlpha', verification is from 
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """

    def __init__(self, mult=True, alpha=0.99,
                 bregman_co_name='BDBregman_KnnK', bregman_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
                estimation. 'False': estimation up to 'proportionality'.        
                (default is True)
        alpha : float, \ne 1, optional
                Parameter of the symmetric Bregman distance (default is
                0.99).
        bregman_co_name : str, optional 
                          You can change it to any nonsymmetric Bregman 
                          distance estimator (default is 'BDBregman_KnnK').
        bregman_co_pars : dictionary, optional
                          Parameters for the nonsymmetric Bregman distance 
                          estimator (default is None (=> {}); in this case
                          the default parameter values of the nonsymmetric
                          Bregman distance estimator are used).
                      
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MDSymBregman_DB()
        >>> co2 =\
                ite.cost.MDSymBregman_DB(bregman_co_name='BDBregman_KnnK')
        >>> co3 =\
                ite.cost.MDSymBregman_DB(bregman_co_name='BDBregman_KnnK',\
                                         bregman_co_pars={'k':6,'eps':0.2})
        
        """

        # initialize with 'InitAlpha':
        super().__init__(mult=mult, alpha=alpha)
        
        # initialize the nonsymmetric Bregman distance estimator:
        bregman_co_pars = bregman_co_pars or {}
        bregman_co_pars['mult'] = mult   # guarantee this property
        bregman_co_pars['alpha'] = alpha   # guarantee this property
        self.bregman_co = co_factory(bregman_co_name, **bregman_co_pars)
        
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

        d = (self.bregman_co.estimation(y1, y2) +
             self.bregman_co.estimation(y2, y1)) / self.alpha
     
        return d


class MDKL_HSCE(InitX, VerEqualDSubspaces):
    """ Kullback-Leibler divergence from cross-entropy and Shannon entropy.
    
    The estimation is based on the relation D(f_1,f_2) =
    CE(f_1,f_2) - H(f_1), where D denotes the Kullback-Leibler divergence,
    CE is the cross-entropy, and H stands for the Shannon differential
    entropy.
    
    Partial initialization comes from 'InitX', verification is from 
    'VerEqualDSubspaces' (see 'ite.cost.x_initialization.py',
    'ite.cost.x_verification.py').
   
    """

    def __init__(self, mult=True, shannon_co_name='BHShannon_KnnK',
                 shannon_co_pars=None, ce_co_name='BCCE_KnnK',
                 ce_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
                estimation. 'False': estimation up to 'proportionality'.        
                (default is True)
        shannon_co_name : str, optional 
                          You can change it to any Shannon entropy
                          (default is 'BHShannon_KnnK').
        shannon_co_pars : dictionary, optional
                     Parameters for the Shannon entropy estimator (default
                     is None (=> {}); in this case the default parameter
                     values of the Shannon entropy estimator are used).
        ce_co_name : str, optional 
                     You can change it to any cross-entropy estimator
                     (default is 'BCCE_KnnK').
        ce_co_pars : dictionary, optional
                     Parameters for the cross-entropy estimator (default
                     is None (=> {}); in this case the default parameter
                     values of the cross-entropy estimator are used).
                      
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MDKL_HSCE()
        >>> co2 = ite.cost.MDKL_HSCE(shannon_co_name='BHShannon_KnnK')
        >>> co3 = ite.cost.MDKL_HSCE(shannon_co_name='BHShannon_KnnK',\
                                     shannon_co_pars={'k':6,'eps':0.2})
        >>> co4 = ite.cost.MDKL_HSCE(shannon_co_name='BHShannon_KnnK',\
                                     shannon_co_pars={'k':5,'eps':0.2},\
                                     ce_co_name='BCCE_KnnK',\
                                     ce_co_pars={'k':6,'eps':0.1})
        
        """

        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the Shannon entropy estimator:
        shannon_co_pars = shannon_co_pars or {}
        shannon_co_pars['mult'] = True   # guarantee this property
        self.shannon_co = co_factory(shannon_co_name, **shannon_co_pars)

        # initialize the cross-entropy estimator:
        ce_co_pars = ce_co_pars or {}
        ce_co_pars['mult'] = True  # guarantee this property
        self.ce_co = co_factory(ce_co_name, **ce_co_pars)

    def estimation(self, y1, y2):
        """ Estimate Kullback-Leibler divergence.
        
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
        Jianhua Lin. Divergence measures based on the Shannon entropy.
        IEEE Transactions on Information Theory, 37:145-151, 1991.
        
        Examples
        --------
        d = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        c = self.ce_co.estimation(y1, y2)
        h = self.shannon_co.estimation(y1)
        d = c - h
    
        return d
