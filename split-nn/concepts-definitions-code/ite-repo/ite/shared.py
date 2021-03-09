from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import pdist, squareform
# from scipy.spatial.distance cdist
from scipy.special import gamma
from scipy.linalg import eigh
from scipy.stats import rankdata

# from scipy.special import gammaln  # estimate_d_temp1/2/3
from numpy.random import permutation, choice
from numpy import pi, cumsum, hstack, zeros, sum, ix_, mean, newaxis, \
                  sqrt, dot, median, exp, min, floor, log, eye, absolute, \
                  array, max, any, place, inf, isinf, where, diag
from scipy.linalg import det, inv

# scipy.spatial.distance.cdist is slightly slow; you can obtain some
# speed-up in case of larger dimensions by using
# ite.shared.cdist_large_dim:
# from ite.shared import cdist_large_dim


def knn_distances(y, q, y_equals_to_q, knn_method='cKDTree', knn_k=3,
                  knn_eps=0, knn_p=2):
    """ Compute the k-nearest neighbors (kNN-s) of Q in y.
    
    Parameters
    ----------
    q : (number of samples in q, dimension)-ndarray
        Query points.
    y : (number of samples in y, dimension)-ndarray
        Data from which the kNN-s are searched.
    y_equals_to_q : boolean
                    'True' if y is equal to q; otherwise it is 'False'.
    knn_method : str, optional
                 kNN computation method; 'cKDTree' or 'KDTree'. (default
                 is 'cKDTree')
    knn_k : int, >= 1, optional
            kNN_k-nearest neighbors. If 'y_equals_to_q' = True, then  
            'knn_k' + 1 <= 'num_of_samples in y'; otherwise 'knn_k' <= 
            'num_of_samples in y'. (default is 3)
    knn_eps : float, >= 0, optional
              The kNN_k^th returned value is guaranteed to be no further
              than (1+eps) times the distance to the real knn_k. (default
              is 0, i.e. the exact kNN-s are computed)
    knn_p   : float, 1 <= p <= infinity, optional
              Which Minkowski p-norm to use. (default is 2, i.e. Euclidean
              norm is taken)

    Returns
    -------        
    distances : array of floats
                The distances to the kNNs; size: 'number of samples in q'
                x 'knn_k'.
    indices : array of integers
              indices[iq,ik] = distance of the iq^th point in q and the
              ik^th NN in q (iq = 1,...,number of samples in q; ik =
              1,...,k); it has the same shape as 'distances'.
    
    """
    
    if knn_method == 'cKDTree':
        tree = cKDTree(y)    
    elif knn_method == 'KDTree':
        tree = KDTree(y)

    if y_equals_to_q:
        if knn_k+1 > y.shape[0]:
            raise Exception("'knn_k' + 1 <= 'num_of_samples in y' " + 
                            "is not satisfied!")
                            
        # distances, indices: |q| x (knn_k+1):                                
        distances, indices = tree.query(q, k=knn_k+1, eps=knn_eps, p=knn_p)
        
        # exclude the points themselves => distances, indices: |q| x knn_k:
        distances, indices = distances[:, 1:], indices[:, 1:]
    else: 
        if knn_k > y.shape[0]:
            raise Exception("'knn_k' <= 'num_of_samples in y' " + 
                            "is not satisfied!")
                            
        # distances, indices: |q| x knn_k:                            
        distances, indices = tree.query(q, k=knn_k, eps=knn_eps, p=knn_p) 
        
    return distances, indices


def volume_of_the_unit_ball(d):
    """ Volume of the d-dimensional unit ball.
    
    Parameters
    ----------
    d : int
        dimension.
        
    Returns
    -------         
    vol : float
          volume.
        
    """
    
    vol = pi**(d/2) / gamma(d/2+1)  # = 2 * pi^(d/2) / ( d*gamma(d/2) )
    
    return vol


def joint_and_product_of_the_marginals_split(z, ds):
    """ Split to samples from the joint and the product of the marginals.
    
    Parameters
    ----------
    z : (number of samples, dimension)-ndarray
        Sample points.
    ds : int vector
         Dimension of the individual subspaces in z; ds[i] = i^th subspace
         dimension.     
     
    Returns
    -------  
    x : (number of samplesx, dimension)-ndarray
        Samples from the joint.
    y : (number of samplesy, dimension)-ndarray
        Sample from the product of the marginals; it is independent of x.
         
    """
    
    # verification (sum(ds) = z.shape[1]):
    if sum(ds) != z.shape[1]:
        raise Exception('sum(ds) must be equal to z.shape[1]; in other ' +
                        'words the subspace dimensions do not sum to the' +
                        ' total dimension!')
        
    # 0,d_1,d_1+d_2,...,d_1+...+d_{M-1}; starting indices of the subspaces:
    cum_ds = cumsum(hstack((0, ds[:-1])))
    
    num_of_samples, dim = z.shape                                   
    num_of_samples2 = num_of_samples//2  # integer division
    
    # x, y:
    x = z[:num_of_samples2, :]
    y = zeros((num_of_samples2, dim))  # preallocation
    for m in range(len(ds)):
        idx = range(cum_ds[m], cum_ds[m] + ds[m])
        y[:, idx] = z[ix_(num_of_samples2 + permutation(num_of_samples2),
                          idx)]
    
    return x, y

    
def estimate_i_alpha(y, co):
    """ Estimate i_alpha = \int p^{\alpha}(y)dy.
    
    The Renyi and Tsallis entropies are simple functions of this quantity. 
    
    Parameters
    ----------
    y : (number of samples, dimension)-ndarray
        One row of y corresponds to one sample.
    co : cost object; details below.
    co.knn_method : str
                    kNN computation method; 'cKDTree' or 'KDTree'.
    co.k : int, >= 1
           k-nearest neighbors.
    co.eps : float, >= 0
             the k^th returned value is guaranteed to be no further than 
             (1+eps) times the distance to the real kNN.
    co.alpha : float
               alpha in the definition of i_alpha
               
    Returns
    -------
    i_alpha : float
              Estimated i_alpha value.
    
    Examples
    --------
    i_alpha = estimate_i_alpha(y,co)
    
    """
    
    num_of_samples, dim = y.shape
    distances_yy = knn_distances(y, y, True, co.knn_method, co.k, co.eps,
                                 2)[0]
    v = volume_of_the_unit_ball(dim)

    # Solution-1 (normal k):
    c = (gamma(co.k)/gamma(co.k + 1 - co.alpha))**(1 / (1 - co.alpha))

    # Solution-2 (if k is 'extreme large', say self.k=180 [ =>
    #            gamma(self.k)=inf], then use this alternative form of
    #            'c', after importing gammaln). Note: we used the
    #            'gamma(a) / gamma(b) = exp(gammaln(a) - gammaln(b))'
    #            identity.
    # c = exp(gammaln(co.k) - gammaln(co.k+1-co.alpha))**(1 / (1-co.alpha))

    s = sum(distances_yy[:, co.k-1]**(dim * (1 - co.alpha)))
    i_alpha = \
        (num_of_samples - 1) / num_of_samples * v**(1 - co.alpha) * \
        c**(1 - co.alpha) * s / (num_of_samples - 1)**co.alpha
              
    return i_alpha


def copula_transformation(y):
    """ Compute the copula transformation of signal y.
    
    Parameters
    ----------
    y : (number of samples, dimension)-ndarray
        One row of y corresponds to one sample.   
        
    Returns
    -------
    z : (number of samples, dimension)-ndarray
        Estimated copula transformed variable.

    Examples
    --------
    z = copula_transformation(y)
        
    """

    # rank transformation (z):
    num_of_samples, dim = y.shape
    z = zeros((num_of_samples, dim))
    for k in range(0, dim):
        z[:, k] = rankdata(y[:, k])

    return z / y.shape[0]


def estimate_d_temp1(y1, y2, co):
    """ Estimate d_temp1 = \int p^{\alpha}(u)q^{1-\alpha}(u)du.
    
    For example, the Renyi and the Tsallis divergences are simple
    functions of this quantity.

    Parameters
    ----------
    y1 : (number of samples1, dimension)-ndarray
         One row of y1 corresponds to one sample.
    y2 : (number of samples2, dimension)-ndarray
         One row of y2 corresponds to one sample.
    co : cost object; details below.
    co.knn_method : str
                    kNN computation method; 'cKDTree' or 'KDTree'.
    co.k : int, >= 1
           k-nearest neighbors.
    co.eps : float, >= 0
             the k^th returned value is guaranteed to be no further than 
             (1+eps) times the distance to the real kNN.         
    
    Returns
    -------
    d_temp2 : float
              Estimated d_temp2 value.
            
    Examples
    --------
    d_temp2 = estimate_d_temp2(y1,y2,co)
     
    """
    
    # initialization:
    num_of_samples1, dim1 = y1.shape
    num_of_samples2, dim2 = y2.shape
    
    # verification:
    if dim1 != dim2:
        raise Exception('The dimension of the samples in y1 and y2 must' +
                        ' be equal!')
    # k, knn_method, eps, dim (= dim1 = dim2):
    k, knn_method, eps, alpha, dim = \
        co.k, co.knn_method, co.eps, co.alpha, dim1
                     
    # kNN distances:                     
    dist_k_y1y1 = knn_distances(y1, y1, True, knn_method, k, eps,
                                2)[0][:, -1]
    dist_k_y2y1 = knn_distances(y2, y1, False, knn_method, k, eps,
                                2)[0][:, -1]

    # b:
    # Solution-I ('normal' k):
    b = gamma(k)**2 / (gamma(k - alpha + 1) * gamma(k + alpha - 1))
    # Solution-II (if k is 'extreme large', say k=180 [=> gamma(k)=inf],
    #              then use this alternative form of 'b'; the identity
    #              used is gamma(a)^2 / (gamma(b) * gamma(c)) =
    #              = exp( 2 * gammaln(a) - gammaln(b) - gammaln(c) )
    # b = exp( 2 * gammaln(k) - gammaln(k - alpha + 1) -
    #         gammaln(k + alpha - 1))

    d_temp1 = mean(((num_of_samples1 - 1) / num_of_samples2 *
                   (dist_k_y1y1 / dist_k_y2y1)**dim)**(1 - alpha)) * b
    
    return d_temp1


def estimate_d_temp2(y1, y2, co):
    """ Estimate d_temp2 = \int p^a(u)q^b(u)p(u)du.
    
    For example, the Hellinger distance and the Bhattacharyya distance are 
    simple functions of this quantity.

    Parameters
    ----------
    y1 : (number of samples1, dimension)-ndarray
         One row of y1 corresponds to one sample.
    y2 : (number of samples2, dimension)-ndarray
         One row of y2 corresponds to one sample.
    co : cost object; details below.
    co.knn_method : str
                    kNN computation method; 'cKDTree' or 'KDTree'.
    co.k : int, >= 1
           k-nearest neighbors.
    co.eps : float, >= 0
             the k^th returned value is guaranteed to be no further than 
             (1+eps) times the distance to the real kNN.         
    co._a : float
    co._b : float
    
    Returns
    -------
    d_temp2 : float
              Estimated d_temp2 value.
            
    Examples
    --------
    d_temp2 = estimate_d_temp2(y1,y2,co)
     
    """
   
    # initialization:
    num_of_samples1, dim1 = y1.shape
    num_of_samples2, dim2 = y2.shape
    
    # verification:
    if dim1 != dim2:
        raise Exception('The dimension of the samples in y1 and y2 must' +
                        ' be equal!')

    # k, knn_method, eps, a, b, dim:
    k, knn_method, eps, a, b, dim = \
        co.k, co.knn_method, co.eps, co._a, co._b, dim1  # =dim2

    # kNN distances:                     
    dist_k_y1y1 = knn_distances(y1, y1, True, knn_method, k, eps,
                                2)[0][:, -1]
    dist_k_y2y1 = knn_distances(y2, y1, False, knn_method, k, eps,
                                2)[0][:, -1]
         
    # b2 computation:
    c = volume_of_the_unit_ball(dim)
    # Solution-I ('normal' k):
    b2 = c**(-(a+b)) * gamma(k)**2 / (gamma(k-a) * gamma(k-b))
    # Solution-II (if k is 'extreme large', say k=180 [=> gamma(k)=inf],
    #              then use this alternative form of 'b2'; the identity
    #              used is gamma(a)^2 / (gamma(b) * gamma(c)) =
    #              = exp( 2 * gammaln(a) - gammaln(b) - gammaln(c) )
    # b2 = c**(-(a+b)) * exp( 2 * gammaln(k) - gammaln(k-a) -gammaln(k-b) )
    
    # b2 -> d_temp2:
    d_temp2 = \
        (num_of_samples1 - 1)**(-a) * num_of_samples2**(-b) * b2 *\
        mean(dist_k_y1y1**(-dim * a) * dist_k_y2y1**(-dim * b))
    
    return d_temp2


def estimate_d_temp3(y1, y2, co):
    """ Estimate d_temp3 = \int p(u)q^{a-1}(u)du.
    
    For example, the Bregman distance can be computed based on this
    quantity.

    Parameters
    ----------
    y1 : (number of samples1, dimension)-ndarray
         One row of y1 corresponds to one sample.
    y2 : (number of samples2, dimension)-ndarray
         One row of y2 corresponds to one sample.
    co : cost object; details below.
    co.knn_method : str
                    kNN computation method; 'cKDTree' or 'KDTree'.
    co.k : int, >= 1
           k-nearest neighbors.
    co.eps : float, >= 0
             the k^th returned value is guaranteed to be no further than 
             (1+eps) times the distance to the real kNN.         
    
    Returns
    -------
    d_temp3 : float
              Estimated d_temp3 value.
            
    Examples
    --------
    d_temp2 = estimate_d_temp2(y1,y2,co)
     
    """
   
    # initialization:
    num_of_samples1, dim1 = y1.shape
    num_of_samples2, dim2 = y2.shape

    # verification:
    if dim1 != dim2:
        raise Exception('The dimension of the samples in y1 and y2 must' +
                        ' be equal!')

    dim, a, k, knn_method, eps = \
        dim1, co.alpha, co.k, co.knn_method, co.eps
    
    # kNN distances: 
    distances_y2y1 = knn_distances(y2, y1, False, knn_method, k, eps, 2)[0]

    # 'ca' computation:
    v = volume_of_the_unit_ball(dim)
    # Solution-I ('normal' k):
    ca = gamma(k) / gamma(k + 1 - a)  # C^a
    # Solution-II (if k is 'extreme large', say k=180 [=> gamma(k)=inf],
    #              then use this alternative form of 'ca'; the identity
    #              used is gamma(a)^2 / (gamma(b) * gamma(c)) =
    #              = exp( 2 * gammaln(a) - gammaln(b) - gammaln(c) )
    # ca = exp(gammaln(k) - gammaln(k + 1 - a))

    d_temp3 = \
        num_of_samples2**(1 - a) * ca * v**(1 - a) * \
        mean(distances_y2y1[:, co.k-1]**(dim * (1 - a)))
       
    return d_temp3    


def cdist_large_dim(y1, y2):
    """ Pairwise Euclidean distance computation.
    
    Parameters
    ----------
    y1 : (number of samples1, dimension)-ndarray
         One row of y1 corresponds to one sample.
    y2 : (number of samples2, dimension)-ndarray
         One row of y2 corresponds to one sample.
         
    Returns
    -------
    d : ndarray
        (number of samples1) x (number of samples2)+sized distance matrix:
        d[i,j] = euclidean_distance(y1[i,:],y2[j,:]).
        
    Notes
    -----
    The function provides a faster pairwise distance computation method
    than scipy.spatial.distance.cdist, if the dimension is 'large'.
    
    Examples
    --------
    d = cdist_large_dim(y1,y2)
    
    """ 
        
    d = sqrt(sum(y1**2, axis=1)[:, newaxis] + sum(y2**2, axis=1)
             - 2 * dot(y1, y2.T))

    return d


def compute_dcov_dcorr_statistics(y, alpha):
    """ Compute the statistics to distance covariance/correlation.  
    
    Parameters
    ----------
    y : (number of samples, dimension)-ndarray
        One row of y corresponds to one sample.
    alpha : float
            0 < alpha < 2
    Returns
    -------
    c : (number of samples, dimension)-ndarray
        Computed statistics.    
        
    """
    d = squareform(pdist(y))**alpha
    ck = mean(d, axis=0)
    c = d - ck - ck[:, newaxis] + mean(ck)
    
    return c


def median_heuristic(y):
    """  Estimate RBF bandwith using median heuristic. 
    
    Parameters
    ----------
    y : (number of samples, dimension)-ndarray
        One row of y corresponds to one sample.

    Returns
    -------
    bandwidth : float
                Estimated RBF bandwith.
    
    """
    
    num_of_samples = y.shape[0]  # number of samples
    # if y contains more samples, then it is subsampled to this cardinality
    num_of_samples_used = 100

    # subsample y (if necessary; select '100' random y columns):
    if num_of_samples > num_of_samples_used:
        idx = choice(num_of_samples, num_of_samples_used, replace=False)
        y = y[idx]  # broadcasting
    
    dist_vector = pdist(y)  # pairwise Euclidean distances
    bandwith = median(dist_vector) / sqrt(2)
    
    return bandwith


def mixture_distribution(ys, w):
    """  Sampling from mixture distribution.

    The samples are generated from the given samples of the individual
    distributions and the mixing weights.

    Parameters
    ----------
    ys : tuple of ndarrays 
         ys[i]: samples from i^th distribution, ys[i][j,:]: j^th sample
         from the i^th distribution. Requirement: the samples (ys[i][j,:])
         have the same dimensions (for all i, j).
    w : vector, w[i] > 0 (for all i), sum(w) = 1
        Mixing weights. Requirement: len(y) = len(w).
    
    """
    
    # verification:
    if sum(w) != 1:
        raise Exception('sum(w) has to be 1!')
            
    if not(all(w > 0)):
        raise Exception('The coordinates of w have to be positive!')
    
    if len(w) != len(ys):
        raise Exception('len(w)=len(ys) has to hold!')

    # number of samples, dimensions:
    num_of_samples_v = array([y.shape[0] for y in ys])
    dim_v = array([y.shape[1] for y in ys])
    if len(set(dim_v)) != 1:  # test if all the dimensions are identical
        raise Exception('All the distributions in ys need to have the ' +
                        'same dimensionality!')
                         
    # take the maximal number of samples (t) for which 't*w1<=t1, ..., 
    # t*wM<=tM', then tm:=floor(t*wm), i.e. compute the trimmed number of 
    # samples:     
    t = min(num_of_samples_v / w)
    tw = tuple(int(e) for e in floor(t * w))
    
    # mix ys[i]-s:
    num_of_samples = sum(tw)
    mixture = zeros((num_of_samples, dim_v[0]))
    idx_start = 0 
    for k in range(len(ys)):
        tw_k = tw[k]
        idx_stop = idx_start + tw_k
        # trim the 'irrelevant' part, the result is added to the mixture:
        mixture[idx_start:idx_stop] = ys[k][:tw_k]  # broadcasting
        
        idx_start = idx_stop
    
    # permute the samples to obtain the mixture (the weights have been
    # taken into account in the trimming part):
    mixture = permutation(mixture)  # permute along the first dimension
    
    return mixture


def compute_h2(ws, ms, ss):
    """ Compute quadratic Renyi entropy for the mixture of Gaussians model.


    Weights, means and standard deviations are given as input.
    
    Parameters
    ----------
    ws : tuple of floats, ws[i] > 0 (for all i), sum(ws) = 1
         Weights.
    ms : tuple of vectors.
         Means: ms[i] = i^th mean.
    ss : tuple of floats, ss[i] > 0 (for all i).
         Standard deviations: ss[i] = i^th std.
         Requirement: len(ws) = len(ms) = len(ss)
         
    Returns     
    -------
    h2 : float,
         Computed quadratic Renyi entropy.
         
    """
    
    # Verification:
    if sum(ws) != 1:
        raise Exception('sum(w) has to be 1!')

    if not(all(tuple(i > j for i, j in zip(ws, zeros(len(ws)))))):
        raise Exception('The coordinates of w have to be positive!')
        
    if len(ws) != len(ms) or len(ws) != len(ss):
        raise Exception('len(ws)=len(ms)=len(ss) has hold!')
        
    # initialization:    
    num_of_comps = len(ws)    # number of componnents
    id_mtx = eye(ms[0].size)  # identity matrix
    term = 0

    # without -log():    
    for n1 in range(num_of_comps):
        for n2 in range(num_of_comps):
            term += ws[n1] * ws[n2] *\
                    normal_density_at_zero(ms[n1] - ms[n2],
                                           (ss[n1]**2 + ss[n2]**2) *
                                           id_mtx)

    h2 = -log(term)
    
    return h2


def normal_density_at_zero(m, c):
    """ Compute the normal density with given mean and covariance at zero. 

    Parameters
    ----------    
    m : vector
        Mean.
    c : ndarray
        Covariance matrix. Assumption: c is square matrix and its size is 
        compatible with that of m.
        
    Returns
    -------
    g : float
        Computed density value.
        
    """
  
    dim = len(m)
    g = 1 / ((2 * pi)**(dim / 2) * sqrt(absolute(det(c)))) *\
        exp(-1/2 * dot(dot(m, inv(c)), m))
            
    return g  


def replace_infs_with_max(m):
    """ Replace the inf elements of matrix 'm' with its largest element.

    The 'largest' is selected from the non-inf entries. If 'm' does not
    contain inf-s, then the output of the function equals to its input.

    Parameters
    ----------
    m : (d1, d2)-ndarray
        Matrix what we want to 'clean'.

    Returns
    -------
    m : float
        Original 'm' but its Inf elements replaced with the max non-Inf
        entry.

    Examples
    --------
    >>> from numpy import inf, array
    >>> m = array([[0.0,1.0,inf], [3.0,inf,5.0]])
    >>> m = replace_infs_with_max(m)
    inf elements: changed to the maximal non-inf one.
    >>> print(m)
    [[ 0.  1.  5.]
     [ 3.  5.  5.]]
    >>> m = array([[0.0,1.0,2.0], [3.0,4.0,5.0]])
    >>> m = replace_infs_with_max(m)
    >>> print(m)
    [[ 0.  1.  2.]
     [ 3.  4.  5.]]

    """

    if any(isinf(m)):
        place(m, m == inf, -inf)  # they will not be maximal
        max_value = max(m)
        place(m, m == -inf, max_value)
        print('inf elements: changed to the maximal non-inf one.')

    return m


def compute_matrix_r_kcca_kgv(y, ds, kernel, tol, kappa):
    """ Computation of the 'r' matrix of KCCA/KGV.

    KCCA is kernel canononical correlation analysis, KGV stands for kernel
    generalized variance.

    This function is a Python implementation, and an extension for the
    subspace case [ds(i)>=1] of 'contrast_tca_kgv.m' which was written by
    Francis Bach for the TCA topic
    (see "http://www.di.ens.fr/~fbach/tca/tca1_0.tar.gz").

    References
    ----------
    Francis R. Bach, Michael I. Jordan. Beyond independent components:
    trees and clusters. Journal of Machine Learning Research, 4:1205-1233,
    2003.

    Parameters
    ----------
    y : (number of samples, dimension)-ndarray
        One row of y corresponds to one sample.
    ds : int vector
         Dimensions of the individual subspaces in y; ds[i] = i^th subspace
         dimension.
    kernel: Kernel.
            See 'ite.cost.x_kernel.py'
    tol: float, > 0
         Tolerance parameter; smaller 'tol' means larger-sized Gram factor
         and better approximation.
    kappa: float, >0
           Regularization parameter.

    """

    # initialization:
    num_of_samples = y.shape[0]
    num_of_subspaces = len(ds)
    # 0,d_1,d_1+d_2,...,d_1+...+d_{M-1}; starting indices of the subspaces:
    cum_ds = cumsum(hstack((0, ds[:-1])))

    sizes = zeros(num_of_subspaces, dtype='int')
    us = list()
    eigs_reg = list()  # regularized eigenvalues

    for m in range(num_of_subspaces):
        # centered g:
        idx = range(cum_ds[m], cum_ds[m] + ds[m])
        g = kernel.ichol(y[:, idx], tol)
        g = g - mean(g, axis=0)  # center the Gram matrix: dot(g,g.T)

        # select the 'relevant' ('>= tol') eigenvalues (eigh =>
        # eigenvalues are real and are in increasing order),
        # eigenvectors[:,i] = i^th eigenvector;
        eigenvalues, eigenvectors = eigh(dot(g.T, g))
        relevant_indices = where(eigenvalues >= tol)
        if relevant_indices[0].size == 0:  # empty
            relevant_indices = array([0])
        eigenvalues = eigenvalues[relevant_indices]

        # append:
        r1 = eigenvectors[:, relevant_indices[0]]
        r2 = diag(sqrt(1 / eigenvalues))
        us.append(dot(g, dot(r1, r2)))
        eigs_reg.append(eigenvalues / (num_of_samples * kappa +
                                       eigenvalues))
        sizes[m] = len(eigenvalues)

    # 'us', 'eigenvalues_regularized' -> 'rkappa':
    rkappa = eye(sum(sizes))
    # 0,d_1,d_1+d_2,...,d_1+...+d_{M-1}; starting indices of the block:
    cum_sizes = cumsum(hstack((0, sizes[:-1])))
    for i in range(1, num_of_subspaces):
        for j in range(i):
            newbottom = dot(dot(diag(eigs_reg[i]), dot(us[i].T, us[j])),
                            diag(eigs_reg[j]))
            idx_i = range(cum_sizes[i], cum_sizes[i] + sizes[i])
            idx_j = range(cum_sizes[j], cum_sizes[j] + sizes[j])
            rkappa[ix_(idx_i, idx_j)] = newbottom
            rkappa[ix_(idx_j, idx_i)] = newbottom.T

    return rkappa
