""" Meta kernel estimators on distributions. """

from numpy import array, exp, log

from ite.cost.x_factory import co_factory
from ite.cost.x_initialization import InitX, InitAlpha, InitUAlpha, \
                                      InitBagGram
from ite.cost.x_verification import VerEqualDSubspaces
from ite.shared import mixture_distribution


class MKExpJR1_HR(InitUAlpha, InitBagGram, VerEqualDSubspaces):
    """ Exponentiated Jensen-Renyi kernel-1 estimator based on Renyi
    entropy.
    
    The estimation is based on the relation K_EJR1(f_1,f_2) = 
    exp[-u x H_R((y^1+y^2)/2)], where K_EJR1 is the exponentiated
    Jensen-Renyi kernel-1, H_R is the Renyi entropy, (y^1+y^2)/2 is the
    mixture of y^1~f_1 and y^2~f_2 with 1/2-1/2 weights, u>0.
    
    Partial initialization comes from 'InitUAlpha' and 'InitBagGram',
    verification is inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    """
    
    def __init__(self, mult=True, alpha=0.99, u=1,
                 renyi_co_name='BHRenyi_KnnK', renyi_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        alpha: float, 0 < alpha < 1, optional
               Parameter of the exponentiated Jensen-Renyi kernel-1
               (default is 0.99).
        u: float, 0 < u, optional
           Parameter of the exponentiated Jensen-Renyi kernel-1 (default
           is 1).
        renyi_co_name : str, optional 
                        You can change it to any Renyi entropy estimator
                        (default is 'BDKL_KnnK').
        renyi_co_pars : dictionary, optional
                        Parameters for the Renyi entropy estimator
                        (default is None (=> {}); in this case the default
                        parameter values of the Renyi entropy estimator
                        are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MKExpJR1_HR()
        >>> co2 = ite.cost.MKExpJR1_HR(renyi_co_name='BHRenyi_KnnK')
        >>> co3 = ite.cost.MKExpJR1_HR(alpha=0.7,u=1.2,\
                                       renyi_co_name='BHRenyi_KnnK')
        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 4, 'eps': 0.1}
        >>> co4 = ite.cost.MKExpJR1_HR(renyi_co_name='BHRenyi_KnnK',\
                                       renyi_co_pars=dict_ch)
        
        """

        # verification (alpha == 1 is checked via 'InitUAlpha'):
        # if alpha <= 0 or alpha > 1:
        #    raise Exception('0 < alpha < 1 has to hold!')
            
        # initialize with 'InitUAlpha':
        super().__init__(mult=mult, u=u, alpha=alpha)

        # initialize the Renyi entropy estimator:
        renyi_co_pars = renyi_co_pars or {}
        renyi_co_pars['mult'] = True    # guarantee this property
        renyi_co_pars['alpha'] = alpha  # -||-
        self.renyi_co = co_factory(renyi_co_name, **renyi_co_pars) 

        # other attributes (u):
        self.u = u

    def estimation(self, y1, y2):
        """ Estimate the value of the exponentiated Jensen-Renyi kernel-1.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        k : float
            Estimated kernel value.
            
        References
        ----------            
        Andre F. T. Martins, Noah A. Smith, Eric P. Xing, Pedro M. Q.
        Aguiar, and Mario A. T. Figueiredo. Nonextensive information
        theoretical kernels on measures. Journal of Machine Learning
        Research, 10:935-975, 2009.
           
        Examples
        --------
        k = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        # mixture:    
        w = array([1/2, 1/2])
        mixture_y = mixture_distribution((y1, y2), w)

        k = exp(-self.u * self.renyi_co.estimation(mixture_y))
        
        return k


class MKExpJR2_DJR(InitUAlpha, InitBagGram, VerEqualDSubspaces):
    """ Exponentiated Jensen-Renyi kernel-2 estimator based on
    Jensen-Renyi divergence
    
    The estimation is based on the relation K_EJR2(f_1,f_2) = 
    exp[-u x D_JR(f_1,f_2)], where K_EJR2 is the exponentiated
    Jensen-Renyi kernel-2, D_JR is the Jensen-Renyi divergence with
    uniform weights (w=(1/2,1/2)), u>0.
    
    Partial initialization comes from 'InitUAlpha' and 'InitBagGram',
    verification is inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    """
    
    def __init__(self, mult=True, alpha=0.99, u=1, jr_co_name='MDJR_HR',
                 jr_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        alpha: float, 0 < alpha < 1, optional
               Parameter of the exponentiated Jensen-Renyi kernel-2
               (default is 0.99).
        u: float, 0 < u, optional
           Parameter of the exponentiated Jensen-Renyi kernel-2 (default
           is 1).
        jr_co_name : str, optional 
                     You can change it to any Jensen-Renyi divergence 
                     estimator (default is 'MDJR_HR').
        jr_co_pars : dictionary, optional
                     Parameters for the Jensen-Renyi divergence estimator 
                     (default is None (=> {}); in this case the default
                     parameter values of the Jensen-Renyi divergence
                     estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MKExpJR2_DJR()
        >>> co2 = ite.cost.MKExpJR2_DJR(jr_co_name='MDJR_HR')
        >>> co3 = ite.cost.MKExpJR2_DJR(alpha=0.7,u=1.2,\
                                        jr_co_name='MDJR_HR')

        """
        
        # verification (alpha == 1 is checked via 'InitUAlpha'):
        # if alpha <= 0 or alpha > 1:
        #    raise Exception('0 < alpha < 1 has to hold!')
            
        # initialize with 'InitUAlpha':
        super().__init__(mult=mult, u=u, alpha=alpha)
        
        # initialize the Jensen-Renyi divergence estimator:
        jr_co_pars = jr_co_pars or {}
        jr_co_pars['mult'] = True    # guarantee this property
        jr_co_pars['alpha'] = alpha  # -||-
        jr_co_pars['w'] = array([1/2, 1/2])  # uniform weights
        self.jr_co = co_factory(jr_co_name, **jr_co_pars) 
        
        # other attributes (u):
        self.u = u

    def estimation(self, y1, y2):
        """ Estimate the value of the exponentiated Jensen-Renyi kernel-2.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        k : float
            Estimated kernel value.
            
        References
        ----------            
        Andre F. T. Martins, Noah A. Smith, Eric P. Xing, Pedro M. Q.
        Aguiar, and Mario A. T. Figueiredo. Nonextensive information
        theoretical kernels on measures. Journal of Machine Learning
        Research, 10:935-975, 2009.
           
        Examples
        --------
        k = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        k = exp(-self.u * self.jr_co.estimation(y1, y2))
        
        return k


class MKExpJS_DJS(InitX, InitBagGram, VerEqualDSubspaces):
    """ Exponentiated Jensen-Shannon kernel estimator based on
    Jensen-Shannon divergence
    
    The estimation is based on the relation K_JS(f_1,f_2) = 
    exp[-u x D_JS(f_1,f_2)], where K_JS is the exponentiated
    Jensen-Shannon kernel, D_JS is the Jensen-Shannon divergence with
    uniform weights (w=(1/2,1/2)), u>0.
    
    Partial initialization comes from 'InitX' and 'InitBagGram',
    verification is inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    """
    
    def __init__(self, mult=True, u=1, js_co_name='MDJS_HS',
                 js_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        u: float, 0 < u, optional
           Parameter of the exponentiated Jensen-Shannon kernel (default
           is 1).
        js_co_name : str, optional 
                     You can change it to any Jensen-Shannon divergence 
                     estimator (default is 'MDJS_HS').
        js_co_pars : dictionary, optional
                     Parameters for the Jensen-Shannnon divergence
                     estimator (default is None (=> {}); in this case the
                     default parameter values of the Jensen-Shannon
                     divergence estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MKExpJS_DJS()
        >>> co2 = ite.cost.MKExpJS_DJS(u=1.2, js_co_name='MDJS_HS')
        
        """
        
        if u <= 0:
            raise Exception('u has to be positive!')
        
        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the Jensen-Shannon divergence estimator:
        js_co_pars = js_co_pars or {}
        js_co_pars['mult'] = True  # guarantee this property
        js_co_pars['w'] = array([1/2, 1/2])  # uniform weights
        self.js_co = co_factory(js_co_name, **js_co_pars) 
        
        # other attributes (u):
        self.u = u

    def estimation(self, y1, y2):
        """ Estimate the value of the exponentiated Jensen-Shannon kernel.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        k : float
            Estimated kernel value.
            
        References
        ----------    
        Andre F. T. Martins, Noah A. Smith, Eric P. Xing, Pedro M. Q.
        Aguiar, and Mario A. T. Figueiredo. Nonextensive information
        theoretical kernels on measures. Journal of Machine Learning
        Research, 10:935-975, 2009.
        
        Andre F. T. Martins, Pedro M. Q. Aguiar, and Mario A. T.
        Figueiredo. Tsallis kernels on measures. In Information Theory
        Workshop (ITW), pages 298-302, 2008.
           
        Examples
        --------
        k = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        k = exp(-self.u * self.js_co.estimation(y1, y2))
        
        return k


class MKExpJT1_HT(InitUAlpha, InitBagGram, VerEqualDSubspaces):
    """ Exponentiated Jensen-Tsallis kernel-1 estimator based on Tsallis 
    entropy.
    
    The estimation is based on the relation K_EJT1(f_1,f_2) = 
    exp[-u x H_T((y^1+y^2)/2)], where K_EJT1 is the exponentiated 
    Jensen-Tsallis kernel-1, H_T is the Tsallis entropy, (y^1+y^2)/2 is
    the mixture of y^1~f_1 and y^2~f_2 with uniform (1/2,1/2) weights, u>0.
    
    Partial initialization comes from 'InitUAlpha' and 'InitBagGram',
    verification is inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    """
    
    def __init__(self, mult=True, alpha=0.99, u=1,
                 tsallis_co_name='BHTsallis_KnnK', tsallis_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        alpha: float, 0 < alpha <= 2, \ne 1, optional
               Parameter of the exponentiated Jensen-Tsallis kernel-1
               (default is 0.99).
        u: float, 0 < u, optional
           Parameter of the exponentiated Jensen-Tsallis kernel-1 (default
           is 1).
        tsallis_co_name : str, optional 
                          You can change it to any Tsallis entropy
                          estimator (default is 'BHTsallis_KnnK').
        tsallis_co_pars : dictionary, optional
                          Parameters for the Tsallis entropy estimator 
                          (default is None (=> {}); in this case the
                          default parameter values of the Tsallis entropy
                          estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MKExpJT1_HT()
        >>> co2 = ite.cost.MKExpJT1_HT(tsallis_co_name='BHTsallis_KnnK')
        >>> co3 = ite.cost.MKExpJT1_HT(alpha=0.7,u=1.2,\
                                       tsallis_co_name='BHTsallis_KnnK')
        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 4, 'eps': 0.1}
        >>> co4 = ite.cost.MKExpJT1_HT(tsallis_co_name='BHTsallis_KnnK',\
                                       tsallis_co_pars=dict_ch)
        
        """

        # verification (alpha == 1 is checked via 'InitUAlpha'):
        # if alpha <= 0 or alpha > 2:
        #    raise Exception('0 < alpha <= 2 has to hold!')
            
        # initialize with 'InitUAlpha':
        super().__init__(mult=mult, u=u, alpha=alpha)

        # initialize the Tsallis entropy estimator:
        tsallis_co_pars = tsallis_co_pars or {}
        tsallis_co_pars['mult'] = True    # guarantee this property
        tsallis_co_pars['alpha'] = alpha  # -||-
        self.tsallis_co = co_factory(tsallis_co_name, **tsallis_co_pars) 

        # other attributes (u):
        self.u = u

    def estimation(self, y1, y2):
        """ Estimate exponentiated Jensen-Tsallis kernel-1.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        k : float
            Estimated kernel value.
            
        References
        ----------            
        Andre F. T. Martins, Noah A. Smith, Eric P. Xing, Pedro M. Q.
        Aguiar, and Mario A. T. Figueiredo. Nonextensive information
        theoretical kernels on measures. Journal of Machine Learning
        Research, 10:935-975, 2009.
           
        Examples
        --------
        k = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        # mixture:    
        w = array([1/2, 1/2])
        mixture_y = mixture_distribution((y1, y2), w)

        k = exp(-self.u * self.tsallis_co.estimation(mixture_y))
        
        return k


class MKExpJT2_DJT(InitUAlpha, InitBagGram, VerEqualDSubspaces):
    """ Exponentiated Jensen-Tsallis kernel-2 estimator based on 
    Jensen-Tsallis divergence.
    
    The estimation is based on the relation K_EJT2(f_1,f_2) = 
    exp[-u x D_JT(f_1,f_2)], where K_EJT2 is the exponentiated
    Jensen-Tsallis kernel-2, D_JT is the Jensen-Tsallis divergence, u>0.
    
    Partial initialization comes from 'InitUAlpha' and 'InitBagGram',
    verification is inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    """
    
    def __init__(self, mult=True, alpha=0.99, u=1, jt_co_name='MDJT_HT',
                 jt_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        alpha: float, 0 < alpha <= 2, \ne 1, optional
               Parameter of the exponentiated Jensen-Tsallis kernel-2
               (default is 0.99).
        u: float, 0 < u, optional
           Parameter of the exponentiated Jensen-Tsallis kernel-2 (default
           is 1).
        jt_co_name : str, optional 
                     You can change it to any Jensen-Tsallis divergence 
                     estimator (default is 'MDJT_HT').
        jt_co_pars : dictionary, optional
                     Parameters for the Jensen-Tsallis divergence
                     estimator (default is None (=> {}); in this case the
                     default parameter values of the Jensen-Tsallis
                     divergence estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MKExpJT2_DJT()
        >>> co2 = ite.cost.MKExpJT2_DJT(jt_co_name='MDJT_HT')
        >>> co3 = ite.cost.MKExpJT2_DJT(alpha=0.7,u=1.2,\
                                        jt_co_name='MDJT_HT')

        """
        
        # verification (alpha == 1 is checked via 'InitUAlpha'):
        # if alpha <= 0 or alpha > 2:
        #    raise Exception('0 < alpha <= 2 has to hold!')
            
        # initialize with 'InitUAlpha':
        super().__init__(mult=mult, u=u, alpha=alpha)
        
        # initialize the Jensen-Tsallis divergence estimator:
        jt_co_pars = jt_co_pars or {}
        jt_co_pars['mult'] = True    # guarantee this property
        jt_co_pars['alpha'] = alpha  # -||-
        self.jt_co = co_factory(jt_co_name, **jt_co_pars) 
        
        # other attributes (u):
        self.u = u

    def estimation(self, y1, y2):
        """ Estimate exponentiated Jensen-Tsallis kernel-2.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        k : float
            Estimated kernel value.
            
        References
        ----------            
        Andre F. T. Martins, Noah A. Smith, Eric P. Xing, Pedro M. Q.
        Aguiar, and Mario A. T. Figueiredo. Nonextensive information
        theoretical kernels on measures. Journal of Machine Learning
        Research, 10:935-975, 2009.
           
        Examples
        --------
        k = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        k = exp(-self.u * self.jt_co.estimation(y1, y2))
        
        return k


class MKJS_DJS(InitX, InitBagGram, VerEqualDSubspaces):
    """ Jensen-Shannon kernel estimator based on Jensen-Shannon divergence.

    The estimation is based on the relation K_JS(f_1,f_2) = log(2) - 
    D_JS(f_1,f_2), where K_JS is the Jensen-Shannon kernel, and D_JS is
    the Jensen-Shannon divergence with uniform weights (w=(1/2,1/2)).
    
    Partial initialization comes from 'InitX' and 'InitBagGram',
    verification is inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    """
    
    def __init__(self, mult=True, js_co_name='MDJS_HS', js_co_pars=None):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        js_co_name : str, optional 
                     You can change it to any Jensen-Shannon divergence 
                     estimator (default is 'MDJS_HS').
        js_co_pars : dictionary, optional
                     Parameters for the Jensen-Shannnon divergence
                     estimator (default is None (=> {}); in this case the
                     default parameter values of the Jensen-Shannon
                     divergence estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MKJS_DJS()
        >>> co2 = ite.cost.MKJS_DJS(js_co_name='MDJS_HS')
        
        """
        
        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # initialize the Jensen-Shannon divergence estimator:
        js_co_pars = js_co_pars or {}
        js_co_pars['mult'] = True   # guarantee this property
        js_co_pars['w'] = array([1/2, 1/2])  # uniform weights
        self.js_co = co_factory(js_co_name, **js_co_pars) 
        
    def estimation(self, y1, y2):
        """ Estimate the value of the Jensen-Shannon kernel.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        k : float
            Estimated kernel value.
            
        References
        ----------    
        Andre F. T. Martins, Noah A. Smith, Eric P. Xing, Pedro M. Q.
        Aguiar, and Mario A. T. Figueiredo. Nonextensive information
        theoretical kernels on measures. Journal of Machine Learning
        Research, 10:935-975, 2009.
        
        Andre F. T. Martins, Pedro M. Q. Aguiar, and Mario A. T.
        Figueiredo. Tsallis kernels on measures. In Information Theory
        Workshop (ITW), pages 298-302, 2008.
           
        Examples
        --------
        k = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        k = log(2) - self.js_co.estimation(y1, y2)
        
        return k


class MKJT_HT(InitAlpha, InitBagGram, VerEqualDSubspaces):
    """ Jensen-Tsallis kernel estimator based on Tsallis entropy.
    
    The estimation is based on the relation K_JT(f_1,f_2) = log_{alpha}(2)
    - T_alpha(f_1,f_2), where (i) K_JT is the Jensen-Tsallis kernel, (ii)
    log_{alpha} is the alpha-logarithm, (iii) T_alpha is the
    Jensen-Tsallis alpha-difference (that can be expressed in terms of the
    Tsallis entropy)
    
    Partial initialization comes from 'InitAlpha' and 'InitBagGram',
    verification is inherited from 'VerEqualDSubspaces' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
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
        alpha: float, 0 < alpha <= 2, \ne 1, optional
               Parameter of the Jensen-Tsallis kernel (default is 0.99).
        tsallis_co_name : str, optional 
                          You can change it to any Tsallis entropy
                          estimator (default is 'BHTsallis_KnnK').
        tsallis_co_pars : dictionary, optional
                          Parameters for the Tsallis entropy estimator 
                          (default is None (=> {}); in this case the
                          default parameter values of the Tsallis entropy
                          estimator are used).
                     
        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.MKJT_HT()
        >>> co2 = ite.cost.MKJT_HT(tsallis_co_name='BHTsallis_KnnK')
        >>> co3 = ite.cost.MKJT_HT(alpha=0.7,\
                                   tsallis_co_name='BHTsallis_KnnK')
        >>> dict_ch = {'knn_method': 'cKDTree', 'k': 4, 'eps': 0.1}
        >>> co4 = ite.cost.MKJT_HT(tsallis_co_name='BHTsallis_KnnK',\
                                   tsallis_co_pars=dict_ch)
        
        """

        # verification (alpha == 1 is checked via 'InitAlpha'):
        if alpha <= 0 or alpha > 2:
            raise Exception('0 < alpha <= 2 has to hold!')

        # initialize with 'InitAlpha':
        super().__init__(mult=mult, alpha=alpha)

        # initialize the Tsallis entropy estimator:
        tsallis_co_pars = tsallis_co_pars or {}
        tsallis_co_pars['mult'] = True    # guarantee this property
        tsallis_co_pars['alpha'] = alpha  # -||-
        self.tsallis_co = co_factory(tsallis_co_name, **tsallis_co_pars) 
        
        # other attribute (log_alpha_2 = alpha-logarithm of 2):
        self.alpha = alpha        
        self.log_alpha_2 = (2**(1 - alpha) - 1) / (1 - alpha)
        
    def estimation(self, y1, y2):
        """ Estimate the value of the Jensen-Tsallis kernel.
        
        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.
    
        Returns
        -------
        k : float
            Estimated kernel value.
            
        References
        ----------            
        Andre F. T. Martins, Noah A. Smith, Eric P. Xing, Pedro M. Q.
        Aguiar, and Mario A. T. Figueiredo. Nonextensive information
        theoretical kernels on measures. Journal of Machine Learning
        Research, 10:935-975, 2009.
           
        Examples
        --------
        k = co.estimation(y1,y2)  
            
        """    
        
        # verification:
        self.verification_equal_d_subspaces(y1, y2)

        # Jensen-Tsallis alpha-difference (jt):
        a = self.alpha

        w = array([1/2, 1/2])
        mixture_y = mixture_distribution((y1, y2), w)  # mixture
        jt = \
            self.tsallis_co.estimation(mixture_y) -\
            (w[0]**a * self.tsallis_co.estimation(y1) +
             w[1]**a * self.tsallis_co.estimation(y2))
    
        k = self.log_alpha_2 - jt
        
        return k
