""" Analytical expressions of information theoretical quantities. """

from scipy.linalg import det, inv
from numpy import log, prod, absolute, exp, pi, trace, dot, cumsum, \
                  hstack, ix_, sqrt, eye, diag, array, sum
                  
from ite.shared import compute_h2


def analytical_value_h_shannon(distr, par):
    """ Analytical value of the Shannon entropy for the given distribution.
    
    Parameters
    ----------    
    distr : str
            Name of the distribution.
    par : dictionary
          Parameters of the distribution. If distr = 'uniform': par["a"], 
          par["b"], par["l"] <- lxU[a,b]. If distr = 'normal' : par["cov"] 
          is the covariance matrix.
          
    Returns
    -------
    h : float
        Analytical value of the Shannon entropy.
           
    """
    
    if distr == 'uniform':
        # par = {"a": a, "b": b, "l": l}
        h = log(prod(par["b"] - par["a"])) + log(absolute(det(par["l"]))) 
    elif distr == 'normal':
        # par = {"cov": c}
        dim = par["cov"].shape[0]  # =c.shape[1]
        h = 1/2 * log((2 * pi * exp(1))**dim * det(par["cov"]))
        # = 1/2 * log(det(c)) + d / 2 * log(2*pi) + d / 2
    else:
        raise Exception('Distribution=?')
    
    return h    


def analytical_value_c_cross_entropy(distr1, distr2, par1, par2):
    """ Analytical value of the cross-entropy for the given distributions.
    
    Parameters
    ----------    
    distr1, distr2 : str
                     Name of the distributions.
    par1, par2 : dictionaries
                 Parameters of the distribution. If distr1 = distr2 =
                 'normal': par1["mean"], par1["cov"] and par2["mean"],
                 par2["cov"] are the means and the covariance matrices.
          
    Returns
    -------
    c : float
        Analytical value of the cross-entropy.
           
    """
    
    if distr1 == 'normal' and distr2 == 'normal':
        # covariance matrices, expectations:
        c1, m1 = par1['cov'], par1['mean']
        c2, m2 = par2['cov'], par2['mean']
        dim = len(m1)
            
        invc2 = inv(c2)
        diffm = m1 - m2
             
        c = 1/2 * (dim * log(2*pi) + log(det(c2)) + trace(dot(invc2, c1)) +
                   dot(diffm, dot(invc2, diffm)))
    else:
        raise Exception('Distribution=?')
        
    return c


def analytical_value_d_kullback_leibler(distr1, distr2, par1, par2):
    """ Analytical value of the KL divergence for the given distributions.
    
    Parameters
    ----------    
    distr1, distr2 : str-s
                    Names of the distributions.
    par1, par2 : dictionary-s
                 Parameters of the distributions. If distr1 = distr2 =
                 'normal': par1["mean"], par1["cov"] and par2["mean"],
                 par2["cov"] are the means and the covariance matrices.
          
    Returns
    -------
    d : float
        Analytical value of the Kullback-Leibler divergence.
           
    """
    
    if distr1 == 'normal' and distr2 == 'normal':
        # covariance matrices, expectations:
        c1, m1 = par1['cov'], par1['mean']
        c2, m2 = par2['cov'], par2['mean']
        dim = len(m1)    
        
        invc2 = inv(c2)
        diffm = m1 - m2
        
        d = 1/2 * (log(det(c2)/det(c1)) + trace(dot(invc2, c1)) +
                   dot(diffm, dot(invc2, diffm)) - dim)
    else:
        raise Exception('Distribution=?')
        
    return d


def analytical_value_i_shannon(distr, par):
    """ Analytical value of mutual information for the given distribution.
    
    Parameters
    ----------    
    distr : str
            Name of the distribution.
    par : dictionary
          Parameters of the distribution. If distr = 'normal': par["ds"], 
          par["cov"] are the vector of component dimensions and the (joint) 
          covariance matrix. 
                    
    Returns
    -------
    i : float
        Analytical value of the Shannon mutual information.
           
    """
    
    if distr == 'normal':
        c, ds = par["cov"], par["ds"]
        # 0,d_1,d_1+d_2,...,d_1+...+d_{M-1}; starting indices of the
        # subspaces:
        cum_ds = cumsum(hstack((0, ds[:-1])))
        i = 1
        for m in range(len(ds)):
            idx = range(cum_ds[m], cum_ds[m] + ds[m])
            i *= det(c[ix_(idx, idx)])
            
        i = log(i / det(c)) / 2     
    else:
        raise Exception('Distribution=?')
    
    return i


def analytical_value_h_renyi(distr, alpha, par):
    """ Analytical value of the Renyi entropy for the given distribution.
    
    Parameters
    ----------    
    distr : str
            Name of the distribution.
    alpha : float, alpha \ne 1
            Parameter of the Renyi entropy.
    par : dictionary
          Parameters of the distribution. If distr = 'uniform': par["a"], 
          par["b"], par["l"] <- lxU[a,b]. If distr = 'normal' : par["cov"]
          is the covariance matrix.
          
    Returns
    -------
    h : float
        Analytical value of the Renyi entropy.
          
    References
    ----------
    Kai-Sheng Song. Renyi information, loglikelihood and an intrinsic 
    distribution measure. Journal of Statistical Planning and Inference
    93: 51-69, 2001.
    
    """
    
    if distr == 'uniform':
        # par = {"a": a, "b": b, "l": l}
        # We also apply the transformation rule of the Renyi entropy in
        # case of linear transformations:
        h = log(prod(par["b"] - par["a"])) + log(absolute(det(par["l"]))) 
    elif distr == 'normal': 
        # par = {"cov": c}
        dim = par["cov"].shape[0]  # =c.shape[1]
        h = log((2*pi)**(dim / 2) * sqrt(absolute(det(par["cov"])))) -\
            dim * log(alpha) / 2 / (1 - alpha)        
    else:
        raise Exception('Distribution=?')
    
    return h    


def analytical_value_h_tsallis(distr, alpha, par):
    """ Analytical value of the Tsallis entropy for the given distribution.
    
    Parameters
    ----------    
    distr : str
            Name of the distribution.
    alpha : float, alpha \ne 1
            Parameter of the Tsallis entropy.
    par : dictionary
          Parameters of the distribution. If distr = 'uniform': par["a"], 
          par["b"], par["l"] <- lxU[a,b]. If distr = 'normal' : par["cov"]
          is the covariance matrix.
          
    Returns
    -------
    h : float
        Analytical value of the Tsallis entropy.
           
    """
    
    # Renyi entropy:
    h = analytical_value_h_renyi(distr, alpha, par)
    
    # Renyi entropy -> Tsallis entropy:
    h = (exp((1 - alpha) * h) - 1) / (1 - alpha)
    
    return h    


def analytical_value_k_prob_product(distr1, distr2, rho, par1, par2):
    """ Analytical value of the probability product kernel.

    Parameters
    ----------    
    distr1, distr2 : str
                     Name of the distributions.
    rho: float, >0
         Parameter of the probability product kernel.
    par1, par2 : dictionary-s
                 Parameters of the distributions. If distr1 = distr2 = 
                 'normal': par1["mean"], par1["cov"] and par2["mean"], 
                 par2["cov"] are the means and the covariance matrices.
          
    Returns
    -------
    k : float
         Analytical value of the probability product kernel.
           
    """
    
    if distr1 == 'normal' and distr2 == 'normal':
        # covariance matrices, expectations:
        c1, m1 = par1['cov'], par1['mean']
        c2, m2 = par2['cov'], par2['mean']
        dim = len(m1)
        
        # inv1, inv2, inv12:
        inv1, inv2 = inv(c1), inv(c2)
        inv12 = inv(inv1+inv2)
        
        m12 = dot(inv1, m1) + dot(inv2, m2)
        exp_arg = \
            dot(m1, dot(inv1, m1)) + dot(m2, dot(inv2, m2)) -\
            dot(m12, dot(inv12, m12))
        
        k = (2 * pi)**((1 - 2 * rho) * dim / 2) * rho**(-dim / 2) *\
            absolute(det(inv12))**(1 / 2) * \
            absolute(det(c1))**(-rho / 2) * \
            absolute(det(c2))**(-rho / 2) * exp(-rho / 2 * exp_arg)
    else:
        raise Exception('Distribution=?')
        
    return k


def analytical_value_k_expected(distr1, distr2, kernel, par1, par2):
    """ Analytical value of expected kernel for the given distributions.
    
    Parameters
    ----------    
    distr1, distr2 : str
                     Names of the distributions.
    kernel: Kernel class.
    par1, par2 : dictionary-s
                 Parameters of the distributions. If distr1 = distr2 = 
                 'normal': par1["mean"], par1["cov"] and par2["mean"], 
                 par2["cov"] are the means and the covariance matrices.
          
    Returns
    -------
    k : float
        Analytical value of the expected kernel.

    References
    ----------
    Krikamol Muandet, Kenji Fukumizu, Francesco Dinuzzo, and Bernhard 
    Scholkopf. Learning from distributions via support measure machines.
    In Advances in Neural Information Processing Systems (NIPS), pages
    10-18, 2011.
          
    """
    
    if distr1 == 'normal' and distr2 == 'normal':
    
        # covariance matrices, expectations:
        c1, m1 = par1['cov'], par1['mean']
        c2, m2 = par2['cov'], par2['mean']

        if kernel.name == 'RBF':
            dim = len(m1)
            gam = 1 / kernel.sigma ** 2
            diffm = m1 - m2
            exp_arg = dot(dot(diffm, inv(c1 + c2 + eye(dim) / gam)), diffm)
            k = exp(-exp_arg / 2) / \
                sqrt(absolute(det(gam * c1 + gam * c2 + eye(dim))))

        elif kernel.name == 'polynomial':
            if kernel.exponent == 2:
                if kernel.c == 1:
                    k = (dot(m1, m2) + 1)**2 + sum(c1 * c2) + \
                        dot(m1, dot(c2, m1)) + dot(m2, dot(c1, m2))
                else:
                    raise Exception('The offset of the polynomial kernel' +
                                    ' (c) should be one!')

            elif kernel.exponent == 3:
                if kernel.c == 1:
                    k = (dot(m1, m2) + 1)**3 + \
                        6 * dot(dot(c1, m1), dot(c2, m2)) + \
                        3 * (dot(m1, m2) + 1) * (sum(c1 * c2) +
                                                 dot(m1, dot(c2, m1)) +
                                                 dot(m2, dot(c1, m2)))
                else:
                    raise Exception('The offset of the polynomial kernel' +
                                    ' (c) should be one!')

            else:
                raise Exception('The exponent of the polynomial kernel ' +
                                'should be either 2 or 3!')
        else:
            raise Exception('Kernel=?')

    else:
        raise Exception('Distribution=?')
        
    return k


def analytical_value_d_mmd(distr1, distr2, kernel, par1, par2):
    """ Analytical value of MMD for the given distributions.

    Parameters
    ----------
    distr1, distr2 : str
                     Names of the distributions.
    kernel: Kernel class.
    par1, par2 : dictionary-s
                 Parameters of the distributions. If distr1 = distr2 =
                 'normal': par1["mean"], par1["cov"] and par2["mean"],
                 par2["cov"] are the means and the covariance matrices.

    Returns
    -------
    d : float
        Analytical value of MMD.

    """

    d_pp = analytical_value_k_expected(distr1, distr1, kernel, par1, par1)
    d_qq = analytical_value_k_expected(distr2, distr2, kernel, par2, par2)
    d_pq = analytical_value_k_expected(distr1, distr2, kernel, par1, par2)
    d = sqrt(d_pp + d_qq - 2 * d_pq)

    return d


def analytical_value_h_sharma_mittal(distr, alpha, beta, par):
    """ Analytical value of the Sharma-Mittal entropy.

    Parameters
    ----------    
    distr : str
            Name of the distribution.
    alpha : float, 0 < alpha \ne 1
            Parameter of the Sharma-Mittal entropy.
    beta : float, beta \ne 1
           Parameter of the Sharma-Mittal entropy.
           
    par : dictionary
          Parameters of the distribution. If distr = 'normal' : par["cov"] 
          = covariance matrix.
          
    Returns
    -------
    h : float
        Analytical value of the Sharma-Mittal entropy.

    References
    ----------   
    Frank Nielsen and Richard Nock. A closed-form expression for the 
    Sharma-Mittal entropy of exponential families. Journal of Physics A: 
    Mathematical and Theoretical, 45:032003, 2012.
        
    """
    
    if distr == 'normal': 
        # par = {"cov": c}
        c = par['cov']
        dim = c.shape[0]  # =c.shape[1]
        h = (((2*pi)**(dim / 2) * sqrt(absolute(det(c))))**(1 - beta) /
             alpha**(dim * (1 - beta) / (2 * (1 - alpha))) - 1) / \
            (1 - beta)
        
    else:
        raise Exception('Distribution=?')
    
    return h    


def analytical_value_h_phi(distr, par, c):
    """ Analytical value of the Phi entropy for the given distribution.
    
    Parameters
    ----------    
    distr : str
            Name of the distribution.
    par : dictionary
          Parameters of the distribution. If distr = 'uniform': par.a,
          par.b in U[a,b].
    c : float, >=1
        Parameter of the Phi-entropy: phi = lambda x: x**c
      
    Returns
    -------
    h : float
        Analytical value of the Phi entropy.
           
    """
    
    if distr == 'uniform': 
        a, b = par['a'], par['b']
        h = 1 / (b-a)**c
    else:
        raise Exception('Distribution=?')
    
    return h   


def analytical_value_d_chi_square(distr1, distr2, par1, par2):
    """ Analytical value of chi^2 divergence for the given distributions.

    Parameters
    ----------    
    distr1, distr2 : str-s.
                     Names of distributions.
    par1, par2 : dictionary-s.
                 Parameters of distributions. If (distr1, distr2) =
                 ('uniform', 'uniform'), then both distributions are
                 uniform: distr1 = U[0,a] with a = par1['a'], distr2 =
                 U[0,b] with b = par2['a']. If (distr1, distr2) =
                 ('normalI', 'normalI'), then distr1 = N(m1,I) where m1 =
                 par1['mean'], distr2 = N(m2,I), where m2 = par2['mean'].

    Returns
    -------
    d : float
        Analytical value of the (Pearson) chi^2 divergence.
        
    References
    ----------       
    Frank Nielsen and Richard Nock. On the chi square and higher-order chi 
    distances for approximating f-divergence. IEEE Signal Processing
    Letters, 2:10-13, 2014.
    
    """
    
    if distr1 == 'uniform' and distr2 == 'uniform':
        a = par1['a']
        b = par2['a']
        d = prod(b) / prod(a) - 1
    elif distr1 == 'normalI' and distr2 == 'normalI':
        m1 = par1['mean']
        m2 = par2['mean']
        diffm = m2 - m1
        d = exp(dot(diffm, diffm)) - 1
    else:
        raise Exception('Distribution=?')
        
    return d


def analytical_value_d_l2(distr1, distr2, par1, par2):
    """ Analytical value of the L2 divergence for the given distributions.

    Parameters
    ----------    
    distr1, distr2 : str-s
                     Names of distributions.
    par1, par2 : dictionary-s
                 Parameters of distributions. If (distr1, distr2) =
                 ('uniform', 'uniform'), then both distributions are
                 uniform: distr1 = U[0,a] with a = par1['a'], distr2 =
                 U[0,b] with b = par2['a'].

    Returns
    -------
    d : float
        Analytical value of the L2 divergence.
           
    """
    
    if distr1 == 'uniform' and distr2 == 'uniform':
        a = par1['a']
        b = par2['a']
        d = sqrt(1 / prod(b) - 1 / prod(a))  
            
    else:
        raise Exception('Distribution=?')
        
    return d


def analytical_value_d_renyi(distr1, distr2, alpha, par1, par2):
    """ Analytical value of Renyi divergence for the given distributions.

    Parameters
    ----------    
    distr1, distr2 : str-s
                     Names of distributions.
    alpha : float, \ne 1
            Parameter of the Sharma-Mittal divergence.
    par1, par2 : dictionary-s
                 Parameters of distributions.
                 If (distr1,distr2) = ('normal','normal'), then distr1 =
                 N(m1,c1), where m1 = par1['mean'], c1 = par1['cov'],
                 distr2 = N(m2,c2), where m2 = par2['mean'], c2 =
                 par2['cov'].

    Returns
    -------
    d : float
        Analytical value of the Renyi divergence.

    References
    ----------
    Manuel Gil. On Renyi Divergence Measures for Continuous Alphabet
    Sources. Phd Thesis, Queen’s University, 2011.
           
    """
    
    if distr1 == 'normal' and distr2 == 'normal':
        # covariance matrices, expectations:
        c1, m1 = par1['cov'], par1['mean']
        c2, m2 = par2['cov'], par2['mean']

        mix_c = alpha * c2 + (1 - alpha) * c1
        diffm = m1 - m2
        
        d = alpha * (1/2 * dot(dot(diffm, inv(mix_c)), diffm) -
                     1 / (2 * alpha * (alpha - 1)) *
                     log(absolute(det(mix_c)) /
                     (det(c1)**(1 - alpha) * det(c2)**alpha)))
            
    else:
        raise Exception('Distribution=?')
        
    return d    


def analytical_value_d_tsallis(distr1, distr2, alpha, par1, par2):
    """ Analytical value of Tsallis divergence for the given distributions.

    Parameters
    ----------    
    distr1, distr2 : str-s
                     Names of distributions.
    alpha : float, \ne 1
            Parameter of the Sharma-Mittal divergence.
    par1, par2 : dictionary-s
                 Parameters of distributions.
                 If (distr1,distr2) = ('normal','normal'), then distr1 =
                 N(m1,c1), where m1 = par1['mean'], c1 = par1['cov'],
                 distr2 = N(m2,c2), where m2 = par2['mean'], c2 =
                 par2['cov'].

    Returns
    -------
    d : float
        Analytical value of the Tsallis divergence.

          
    """
    
    if distr1 == 'normal' and distr2 == 'normal':
        d = analytical_value_d_renyi(distr1, distr2, alpha, par1, par2)
        d = (exp((alpha - 1) * d) - 1) / (alpha - 1)
    else:
        raise Exception('Distribution=?')
        
    return d


def analytical_value_d_sharma_mittal(distr1, distr2, alpha, beta, par1,
                                     par2):
    """ Analytical value of the Sharma-Mittal divergence.

    Parameters
    ----------    
    distr1, distr2 : str-s
                     Names of distributions.
    alpha : float, 0 < alpha \ne 1
            Parameter of the Sharma-Mittal divergence.
    beta : float, beta \ne 1
           Parameter of the Sharma-Mittal divergence.
    par1, par2 : dictionary-s
                 Parameters of distributions.
                 If (distr1,distr2) = ('normal','normal'), then distr1 =
                 N(m1,c1), where m1 = par1['mean'], c1 = par1['cov'],
                 distr2 = N(m2,c2), where m2 = par2['mean'], c2 =
                 par2['cov'].

    Returns
    -------
    D : float
        Analytical value of the Tsallis divergence.

    References
    ----------          
    Frank Nielsen and Richard Nock. A closed-form expression for the 
    Sharma-Mittal entropy of exponential families. Journal of Physics A: 
    Mathematical and Theoretical, 45:032003, 2012.
    
    """
    
    if distr1 == 'normal' and distr2 == 'normal':
        # covariance matrices, expectations:
        c1, m1 = par1['cov'], par1['mean']
        c2, m2 = par2['cov'], par2['mean']
        
        c = inv(alpha * inv(c1) + (1 - alpha) * inv(c2))
        diffm = m1 - m2
    
        # Jensen difference divergence, c2:
        j = (log(absolute(det(c1))**alpha * absolute(det(c2))**(1 -
                                                                alpha) /
                 absolute(det(c))) + alpha * (1 - alpha) *
             dot(dot(diffm, inv(c)), diffm)) / 2
        c2 = exp(-j)
        
        d = (c2**((1 - beta) / (1 - alpha)) - 1) / (beta - 1)

    else:
        raise Exception('Distribution=?')
        
    return d


def analytical_value_d_bregman(distr1, distr2, alpha, par1, par2):
    """ Analytical value of Bregman divergence for the given distributions.

    Parameters
    ----------    
    distr1, distr2 : str-s
                     Names of distributions.
    alpha : float, \ne 1
            Parameter of the Bregman divergence.
    par1, par2 : dictionary-s
                 Parameters of distributions. If (distr1, distr2) =
                 ('uniform', 'uniform'), then both distributions are
                 uniform: distr1 = U[0,a] with a = par1['a'], distr2 =
                 U[0,b] with b = par2['a'].

    Returns
    -------
    d : float
        Analytical value of the Bregman divergence.
           
    """
    
    if distr1 == 'uniform' and distr2 == 'uniform':
        a = par1['a']
        b = par2['a']
        d = \
            -1 / (alpha - 1) * prod(b)**(1 - alpha) +\
            1 / (alpha - 1) * prod(a)**(1 - alpha)
    else:
        raise Exception('Distribution=?')
        
    return d


def analytical_value_d_jensen_renyi(distr1, distr2, w, par1, par2):
    """ Analytical value of the Jensen-Renyi divergence.

    Parameters
    ----------    
    distr1, distr2 : str-s
                     Names of distributions.
    w    : vector, w[i] > 0 (for all i), sum(w) = 1
           Weight used in the Jensen-Renyi divergence.                     
    par1, par2 : dictionary-s
                 Parameters of distributions. If (distr1, distr2) =
                 ('normal', 'normal'), then both distributions are normal:
                 distr1 = N(m1,s1^2 I) with m1 = par1['mean'], s1 =
                 par1['std'], distr2 = N(m2,s2^2 I) with m2 =
                 par2['mean'], s2 = par2['std'].

    Returns
    -------
    d : float
        Analytical value of the Jensen-Renyi divergence.
        
    References           
    ----------
    Fei Wang, Tanveer Syeda-Mahmood, Baba C. Vemuri, David Beymer, and
    Anand Rangarajan. Closed-Form Jensen-Renyi Divergence for Mixture of
    Gaussians and Applications to Group-Wise Shape Registration. Medical
    Image Computing and Computer-Assisted Intervention, 12: 648–655, 2009.
    
    """
    
    if distr1 == 'normal' and distr2 == 'normal':
        m1, s1 = par1['mean'], par1['std']
        m2, s2 = par2['mean'], par2['std']
        term1 = compute_h2(w, (m1, m2), (s1, s2))
        term2 = \
            w[0] * compute_h2((1,), (m1,), (s1,)) +\
            w[1] * compute_h2((1,), (m2,), (s2,))

        # H2(\sum_i wi yi) - \sum_i w_i H2(yi), where H2 is the quadratic
        # Renyi entropy:
        d = term1 - term2

    else:
        raise Exception('Distribution=?')
        
    return d


def analytical_value_i_renyi(distr, alpha, par):
    """ Analytical value of the Renyi mutual information.

    Parameters
    ----------    
    distr : str
            Name of the distribution.
    alpha : float
            Parameter of the Renyi mutual information.
    par : dictionary
          Parameters of the distribution. If distr = 'normal': par["cov"]
          is the covariance matrix.
                    
    Returns
    -------
    i : float
        Analytical value of the Renyi mutual information.
           
    """
    
    if distr == 'normal':
        c = par["cov"]        

        t1 = -alpha / 2 * log(det(c))
        t2 = -(1 - alpha) / 2 * log(prod(diag(c)))
        t3 = log(det(alpha * inv(c) + (1 - alpha) * diag(1 / diag(c)))) / 2
        i = 1 / (alpha - 1) * (t1 + t2 - t3)            
    else:
        raise Exception('Distribution=?')
    
    return i


def analytical_value_k_ejr1(distr1, distr2, u, par1, par2):
    """ Analytical value of the Jensen-Renyi kernel-1.

    Parameters
    ----------
    distr1, distr2 : str-s
                     Names of distributions.
    u    : float, >0
           Parameter of the Jensen-Renyi kernel-1 (alpha = 2: fixed).
    par1, par2 : dictionary-s
                 Parameters of distributions. If (distr1, distr2) =
                 ('normal', 'normal'), then both distributions are normal:
                 distr1 = N(m1,s1^2 I) with m1 = par1['mean'], s1 =
                 par1['std'], distr2 = N(m2,s2^2 I) with m2 =
                 par2['mean'], s2 = par2['std'].

    References
    ----------
    Fei Wang, Tanveer Syeda-Mahmood, Baba C. Vemuri, David Beymer, and
    Anand Rangarajan. Closed-Form Jensen-Renyi Divergence for Mixture of
    Gaussians and Applications to Group-Wise Shape Registration. Medical
    Image Computing and Computer-Assisted Intervention, 12: 648–655, 2009.
    
    """ 
    
    if distr1 == 'normal' and distr2 == 'normal':
        m1, s1 = par1['mean'], par1['std']
        m2, s2 = par2['mean'], par2['std']
        w = array([1/2, 1/2])
        h = compute_h2(w, (m1, m2), (s1, s2))  # quadratic Renyi entropy
        k = exp(-u * h)
    else:
        raise Exception('Distribution=?')
        
    return k


def analytical_value_k_ejr2(distr1, distr2, u, par1, par2):
    """ Analytical value of the Jensen-Renyi kernel-2.

    Parameters
    ----------
    distr1, distr2 : str-s
                     Names of distributions.
    u    : float, >0
           Parameter of the Jensen-Renyi kernel-2 (alpha = 2: fixed).
    par1, par2 : dictionary-s
                 Parameters of distributions. If (distr1, distr2) =
                 ('normal', 'normal'), then both distributions are normal:
                 distr1 = N(m1,s1^2 I) with m1 = par1['mean'], s1 =
                 par1['std'], distr2 = N(m2,s2^2 I) with m2 =
                 par2['mean'], s2 = par2['std'].

    """ 
    
    if distr1 == 'normal' and distr2 == 'normal':
        w = array([1/2, 1/2])
        d = analytical_value_d_jensen_renyi(distr1, distr2, w, par1, par2)
        k = exp(-u * d)
    else:
        raise Exception('Distribution=?')
        
    return k    


def analytical_value_k_ejt1(distr1, distr2, u, par1, par2):
    """ Analytical value of the Jensen-Tsallis kernel-1.

    Parameters
    ----------
    distr1, distr2 : str-s
                     Names of distributions.
    u    : float, >0
           Parameter of the Jensen-Tsallis kernel-1 (alpha = 2: fixed).
    par1, par2 : dictionary-s
                 Parameters of distributions. If (distr1, distr2) =
                 ('normal', 'normal'), then both distributions are normal:
                 distr1 = N(m1,s1^2 I) with m1 = par1['mean'], s1 =
                 par1['std'], distr2 = N(m2,s2^2 I) with m2 =
                 par2['mean'], s2 = par2['std'].

    References
    ----------
    Fei Wang, Tanveer Syeda-Mahmood, Baba C. Vemuri, David Beymer, and
    Anand Rangarajan. Closed-Form Jensen-Renyi Divergence for Mixture of
    Gaussians and Applications to Group-Wise Shape Registration. Medical
    Image Computing and Computer-Assisted Intervention, 12: 648–655, 2009.
    (Renyi entropy)
    
    """ 
   
    if distr1 == 'normal' and distr2 == 'normal':
        m1, s1 = par1['mean'], par1['std']
        m2, s2 = par2['mean'], par2['std']
        w = array([1/2, 1/2])
        h = compute_h2(w, (m1, m2), (s1, s2))  # quadratic Renyi entropy
        # quadratic Renyi entropy -> quadratic Tsallis entropy:
        h = 1 - exp(-h)
        k = exp(-u * h)
    else:
        raise Exception('Distribution=?')
        
    return k

    
def analytical_value_k_ejt2(distr1, distr2, u, par1, par2):
    """ Analytical value of the Jensen-Tsallis kernel-2.

    Parameters
    ----------
    distr1, distr2 : str-s
                     Names of distributions.
    u    : float, >0
           Parameter of the Jensen-Tsallis kernel-2 (alpha = 2: fixed).
    par1, par2 : dictionary-s
                 Parameters of distributions. If (distr1, distr2) =
                 ('normal', 'normal'), then both distributions are normal:
                 distr1 = N(m1,s1^2 I) with m1 = par1['mean'], s1 =
                 par1['std'], distr2 = N(m2,s2^2 I) with m2 =
                 par2['mean'], s2 = par2['std'].

    References
    ----------
    Fei Wang, Tanveer Syeda-Mahmood, Baba C. Vemuri, David Beymer, and
    Anand Rangarajan. Closed-Form Jensen-Renyi Divergence for Mixture of
    Gaussians and Applications to Group-Wise Shape Registration. Medical
    Image Computing and Computer-Assisted Intervention, 12: 648–655, 2009.
    (analytical value of the Jensen-Renyi divergence)
    
    """ 
    
    if distr1 == 'normal' and distr2 == 'normal':
        m1, s1 = par1['mean'], par1['std']
        m2, s2 = par2['mean'], par2['std']
        w = array([1/2, 1/2])
        # quadratic Renyi entropy -> quadratic Tsallis entropy:
        term1 = 1 - exp(-compute_h2(w, (m1, m2), (s1, s2)))
        term2 = \
            w[0] * (1 - exp(-compute_h2((1, ), (m1, ), (s1,)))) +\
            w[1] * (1 - exp(-compute_h2((1,), (m2,), (s2,))))
        # H2(\sum_i wi Yi) - \sum_i w_i H2(Yi), where H2 is the quadratic
        # Tsallis entropy:
        d = term1 - term2

        k = exp(-u * d)
    else:
        raise Exception('Distribution=?')
        
    return k


def analytical_value_d_hellinger(distr1, distr2, par1, par2):
    """ Analytical value of Hellinger distance for the given distributions.

    Parameters
    ----------
    distr1, distr2 : str-s
                    Names of the distributions.
    par1, par2 : dictionary-s
                 Parameters of the distributions. If distr1 = distr2 =
                 'normal': par1["mean"], par1["cov"] and par2["mean"],
                 par2["cov"] are the means and the covariance matrices.

    Returns
    -------
    d : float
        Analytical value of the Hellinger distance.

    """

    if distr1 == 'normal' and distr2 == 'normal':
        # covariance matrices, expectations:
        c1, m1 = par1['cov'], par1['mean']
        c2, m2 = par2['cov'], par2['mean']

        # "https://en.wikipedia.org/wiki/Hellinger_distance": Examples:
        diffm = m1 - m2
        avgc = (c1 + c2) / 2
        inv_avgc = inv(avgc)
        d = 1 - det(c1)**(1/4) * det(c2)**(1/4) / sqrt(det(avgc)) * \
            exp(-dot(diffm, dot(inv_avgc, diffm))/8)  # D^2

        d = sqrt(d)
    else:
        raise Exception('Distribution=?')

    return d


def analytical_value_cond_h_shannon(distr, par):
    """ Analytical value of the conditional Shannon entropy.

    Parameters
    ----------
    distr : str-s
            Names of the distributions; 'normal'.
    par : dictionary
          Parameters of the distribution. If distr is 'normal': par["cov"]
          and par["dim1"] are the covariance matrix and the dimension of
          y1.

    Returns
    -------
    cond_h : float
             Analytical value of the conditional Shannon entropy.

    """

    if distr == 'normal':
        # h12 (=joint entropy):
        h12 = analytical_value_h_shannon(distr, par)

        # h2 (=entropy of the conditioning variable):
        c, dim1 = par['cov'], par['dim1']  # covariance matrix, dim(y1)
        par = {"cov": c[dim1:, dim1:]}
        h2 = analytical_value_h_shannon(distr, par)

        cond_h = h12 - h2

    else:
        raise Exception('Distribution=?')

    return cond_h


def analytical_value_cond_i_shannon(distr, par):
    """ Analytical value of the conditional Shannon mutual information.

     Parameters
     ----------
     distr : str-s
             Names of the distributions; 'normal'.
     par : dictionary
           Parameters of the distribution. If distr is 'normal':
           par["cov"] and par["ds"] are the (joint) covariance matrix and
           the vector of subspace dimensions.

     Returns
     -------
     cond_i : float
              Analytical value of the conditional Shannon mutual
              information.

    """

    # initialization:
    ds = par['ds']
    len_ds = len(ds)
    # 0,d_1,d_1+d_2,...,d_1+...+d_M; starting indices of the subspaces:
    cum_ds = cumsum(hstack((0, ds[:-1])))
    idx_condition = range(cum_ds[len_ds - 1],
                          cum_ds[len_ds - 1] + ds[len_ds - 1])

    if distr == 'normal':
        c = par['cov']

        # h_joint:
        h_joint = analytical_value_h_shannon(distr, par)

        # h_cross:
        h_cross = 0
        for m in range(len_ds-1):  # non-conditioning subspaces
            idx_m = range(cum_ds[m], cum_ds[m] + ds[m])
            idx_m_and_condition = hstack((idx_m, idx_condition))
            par = {"cov": c[ix_(idx_m_and_condition, idx_m_and_condition)]}
            h_cross += analytical_value_h_shannon(distr, par)

        # h_condition:
        par = {"cov": c[ix_(idx_condition, idx_condition)]}
        h_condition = analytical_value_h_shannon(distr, par)

        cond_i = -h_joint + h_cross - (len_ds - 2) * h_condition

    else:
        raise Exception('Distribution=?')

    return cond_i
