""" Base association measure estimators. """

from numpy import mean, prod, triu, ones, dot, sum, maximum, all
from scipy.special import binom

from ite.cost.x_initialization import InitX
from ite.cost.x_verification import VerOneDSubspaces, VerCompSubspaceDims
from ite.shared import copula_transformation


class BASpearman1(InitX, VerOneDSubspaces, VerCompSubspaceDims):
    """ Estimator of the first multivariate extension of Spearman's rho.

    Initialization is inherited from 'InitX', verification capabilities
    come from 'VerOneDSubspaces' and 'VerCompSubspaceDims' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').

    Examples
    --------
    >>> import ite
    >>> co = ite.cost.BASpearman1()

    """

    def estimation(self, y, ds=None):
        """ Estimate the first multivariate extension of Spearman's rho.
        
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
            Estimated first multivariate extension of Spearman's rho.

        References
        ----------
        Friedrich Shmid, Rafael Schmidt, Thomas Blumentritt, Sandra
        Gaiser, and Martin Ruppert. Copula Theory and Its Applications,
        Chapter Copula based Measures of Multivariate Association. Lecture
        Notes in Statistics. Springer, 2010.
        
        Friedrich Schmid and Rafael Schmidt. Multivariate extensions of 
        Spearman's rho and related statistics. Statistics & Probability 
        Letters, 77:407-416, 2007.
        
        Roger B. Nelsen. Nonparametric measures of multivariate
        association. Lecture Notes-Monograph Series, Distributions with
        Fixed Marginals and Related Topics, 28:223-232, 1996.
        
        Edward F. Wolff. N-dimensional measures of dependence.
        Stochastica, 4:175-188, 1980.
        
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
        
        dim = y.shape[1]  # dimension
        u = copula_transformation(y)
        h = (dim + 1) / (2**dim - (dim + 1))  # h_rho(dim)
        a = h * (2**dim * mean(prod(1 - u, axis=1)) - 1)
        
        return a


class BASpearman2(InitX, VerOneDSubspaces, VerCompSubspaceDims):
    """ Estimator of the second multivariate extension of Spearman's rho.

    Initialization is inherited from 'InitX', verification capabilities
    come from 'VerOneDSubspaces' and 'VerCompSubspaceDims' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').

    Examples
    --------
    >>> import ite
    >>> co = ite.cost.BASpearman2()

    """
    
    def estimation(self, y, ds=None):
        """ Estimate the second multivariate extension of Spearman's rho.
        
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
            Estimated second multivariate extension of Spearman's rho.

        References
        ----------
        Friedrich Shmid, Rafael Schmidt, Thomas Blumentritt, Sandra
        Gaiser, and Martin Ruppert. Copula Theory and Its Applications,
        Chapter Copula based Measures of Multivariate Association. Lecture
        Notes in Statistics. Springer, 2010.
        
        Friedrich Schmid and Rafael Schmidt. Multivariate extensions of 
        Spearman's rho and related statistics. Statistics & Probability 
        Letters, 77:407-416, 2007.
        
        Roger B. Nelsen. Nonparametric measures of multivariate
        association. Lecture Notes-Monograph Series, Distributions with
        Fixed Marginals and Related Topics, 28:223-232, 1996.
        
        Harry Joe. Multivariate concordance. Journal of Multivariate
        Analysis, 35:12-30, 1990.
        
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
        
        dim = y.shape[1]  # dimension
        u = copula_transformation(y)
        h = (dim + 1) / (2**dim - (dim + 1))  # h_rho(dim)
        
        a = h * (2**dim * mean(prod(u, axis=1)) - 1)
        
        return a


class BASpearman3(InitX, VerOneDSubspaces, VerCompSubspaceDims):
    """ Estimator of the third multivariate extension of Spearman's rho.

    Initialization is inherited from 'InitX', verification capabilities
    come from 'VerOneDSubspaces' and 'VerCompSubspaceDims' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').

    Examples
    --------
    >>> import ite
    >>> co = ite.cost.BASpearman3()

    """

    def estimation(self, y, ds=None):
        """ Estimate the third multivariate extension of Spearman's rho.
        
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
            Estimated third multivariate extension of Spearman's rho.

        References
        ----------
        Friedrich Shmid, Rafael Schmidt, Thomas Blumentritt, Sandra
        Gaiser, and Martin Ruppert. Copula Theory and Its Applications,
        Chapter Copula based Measures of Multivariate Association. Lecture
        Notes in Statistics. Springer, 2010.
        
        Roger B. Nelsen. An Introduction to Copulas (Springer Series in 
        Statistics). Springer, 2006.
        
        Roger B. Nelsen. Distributions with Given Marginals and
        Statistical Modelling, chapter Concordance and copulas: A survey,
        pages 169-178. Kluwer Academic Publishers, Dordrecht, 2002.
        
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
        
        dim = y.shape[1]  # dimension
        u = copula_transformation(y)
        h = (dim + 1) / (2**dim - (dim + 1))  # h_rho(d)
        
        a1 = h * (2**dim * mean(prod(1 - u, axis=1)) - 1)
        a2 = h * (2**dim * mean(prod(u, axis=1)) - 1)
        a = (a1 + a2) / 2
        
        return a        


class BASpearman4(InitX, VerOneDSubspaces, VerCompSubspaceDims):
    """ Estimator of the fourth multivariate extension of Spearman's rho.

    Initialization is inherited from 'InitX', verification capabilities
    come from 'VerOneDSubspaces' and 'VerCompSubspaceDims'; (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').

    Examples
    --------
    >>> import ite
    >>> co = ite.cost.BASpearman4()

    """
    
    def estimation(self, y, ds=None):
        """ Estimate the fourth multivariate extension of Spearman's rho.
        
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
            Estimated fourth multivariate extension of Spearman's rho.

        References
        ----------
        Friedrich Shmid, Rafael Schmidt, Thomas Blumentritt, Sandra
        Gaiser, and Martin Ruppert. Copula Theory and Its Applications,
        Chapter Copula based Measures of Multivariate Association. Lecture
        Notes in Statistics. Springer, 2010.

        Friedrich Schmid and Rafael Schmidt. Multivariate extensions of 
        Spearman's rho and related statistics. Statistics & Probability 
        Letters, 77:407-416, 2007.
        
        Maurice G. Kendall. Rank correlation methods. London, Griffin,
        1970.
       
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

        num_of_samples, dim = y.shape  # number of samples, dimension
        u = copula_transformation(y)
        
        m_triu = triu(ones((dim, dim)), 1)  # upper triangular mask
        b = binom(dim, 2)
        a = 12 * sum(dot((1 - u).T, (1 - u)) * m_triu) /\
            (b * num_of_samples) - 3
        
        return a


class BASpearmanCondLT(InitX, VerOneDSubspaces, VerCompSubspaceDims):
    """ Estimate multivariate conditional version of Spearman's rho.

    The measure weights the lower tail of the copula.

    Partial initialization comes from 'InitX'; verification capabilities
    are inherited from 'VerOneDSubspaces' and 'VerCompSubspaceDims' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').

    """
   
    def __init__(self, mult=True, p=0.5):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        p : float, 0<p<=1, optional 
            (default is 0.5)

        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.BASpearmanCondLT()
        >>> co2 = ite.cost.BASpearmanCondLT(p=0.4)

        """
        
        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # p:
        self.p = p
     
    def estimation(self, y, ds=None):
        """ Estimate multivariate conditional version of Spearman's rho.
        
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
            Estimated multivariate conditional version of Spearman's rho.

        References
        ----------
        Friedrich Schmid and Rafael Schmidt. Multivariate conditional
        versions of Spearman's rho and related measures of tail dependence.
        Journal of Multivariate Analysis, 98:1123-1140, 2007.
        
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
        
        num_of_samples, dim = y.shape  # number of samples, dimension
        u = copula_transformation(y)
        c1 = (self.p**2 / 2)**dim
        c2 = self.p**(dim + 1) / (dim + 1)
        
        a = (mean(prod(maximum(self.p - u, 0), axis=1)) - c1) / (c2 - c1)
        
        return a


class BASpearmanCondUT(InitX, VerOneDSubspaces, VerCompSubspaceDims):
    """ Estimate multivariate conditional version of Spearman's rho.

    The measure weights the upper tail of the copula.

    Partial initialization comes from 'InitX'; verification capabilities
    are inherited from 'VerOneDSubspaces' and 'VerCompSubspaceDims' (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').
    
    """
   
    def __init__(self, mult=True, p=0.5):
        """ Initialize the estimator. 
        
        Parameters
        ----------
        mult : bool, optional
               'True': multiplicative constant relevant (needed) in the 
               estimation. 'False': estimation up to 'proportionality'.        
               (default is True)
        p : float, 0<p<=1, optional 
            (default is 0.5)

        Examples
        --------
        >>> import ite
        >>> co1 = ite.cost.BASpearmanCondUT()
        >>> co2 = ite.cost.BASpearmanCondUT(p=0.4)

        """
        
        # initialize with 'InitX':
        super().__init__(mult=mult)
        
        # p:
        self.p = p
     
    def estimation(self, y, ds=None):
        """ Estimate multivariate conditional version of Spearman's rho.
        
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
            Estimated multivariate conditional version of Spearman's rho.

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
       
        num_of_samples, dim = y.shape  # number of samples, dimension
        u = copula_transformation(y)
    
        c = mean(prod(1 - maximum(u, 1 - self.p), axis=1))
        c1 = (self.p * (2 - self.p) / 2)**dim
        c2 = self.p**dim * (dim + 1 - self.p * dim) / (dim + 1)

        a = (c - c1) / (c2 - c1)        
        
        return a


class BABlomqvist(InitX, VerOneDSubspaces, VerCompSubspaceDims):
    """ Estimator of the multivariate extension of Blomqvist's beta.

    Blomqvist's beta is also known as the medial correlation coefficient.

    Initialization is inherited from 'InitX', verification capabilities
    come from 'VerOneDSubspaces' and 'VerCompSubspaceDims'
    ('ite.cost.x_classes.py').

    Initialization is inherited from 'InitX', verification capabilities
    come from 'VerOneDSubspaces' and 'VerCompSubspaceDims'  (see
    'ite.cost.x_initialization.py', 'ite.cost.x_verification.py').

    Examples
    --------
    >>> import ite
    >>> co = ite.cost.BABlomqvist()

    """
    
    def estimation(self, y, ds=None):
        """ Estimate multivariate extension of Blomqvist's beta.
        
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
            Estimated multivariate extension of Blomqvist's beta.

        References
        ----------
        Friedrich Schmid, Rafael Schmidt, Thomas Blumentritt, Sandra
        Gaiser, and Martin Ruppert. Copula Theory and Its Applications,
        Chapter Copula based Measures of Multivariate Association. Lecture
        Notes in Statistics. Springer, 2010. (multidimensional case,
        len(ds)>=2)
        
        Manuel Ubeda-Flores. Multivariate versions of Blomqvist's beta and 
        Spearman's footrule. Annals of the Institute of Statistical 
        Mathematics, 57:781-788, 2005.
        
        Nils Blomqvist. On a measure of dependence between two random 
        variables. The Annals of Mathematical Statistics, 21:593-600, 1950. 
        (2D case, statistical properties)
        
        Frederick Mosteller. On some useful ''inefficient'' statistics.
        Annals of Mathematical Statistics, 17:377--408, 1946. (2D case,
        def)


        Examples
        --------
        a = co.estimation(y,ds)  

        """
        
        if ds is None:  # emulate 'ds = vector of ones'
            ds = ones(y.shape[1], dtype='int')
        
        # verification:
        self.verification_compatible_subspace_dimensions(y, ds)
        self.verification_one_dimensional_subspaces(ds)

        num_of_samples, dim = y.shape  # number of samples, dimension
        u = copula_transformation(y)

        h = 2**(dim - 1) / (2**(dim - 1) - 1)  # h(dim)
        c1 = mean(all(u <= 1/2, axis=1))  # C(1/2)
        c2 = mean(all(u > 1/2, axis=1))  # \bar{C}(1/2)
        a = h * (c1 + c2 - 2**(1 - dim))
        
        return a                     
