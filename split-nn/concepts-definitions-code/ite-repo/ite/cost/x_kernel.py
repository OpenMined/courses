""" Kernel class.

It provides Gram matrix computation and incomplete Cholesky decomposition
capabilities.

"""

from scipy.spatial.distance import pdist, cdist, squareform
from numpy import sum, sqrt, exp, dot, ones, array, zeros, argmax, \
                  hstack, newaxis, copy, argsort


class Kernel(object):
    """ Kernel class """

    def __init__(self, par=None):
        """ Initialization.

        Parameters
        ----------
        par : dictionary, optional
              Name of the kernel and its parameters (default is
              {'name': 'RBF','sigma': 1}). The name of the kernel comes
              from 'RBF', 'exponential', 'Cauchy', 'student', 'Matern3p2',
              'Matern5p2', 'polynomial', 'ratquadr' (rational quadratic),
              'invmquadr' (inverse multiquadr).

        Examples
        --------
        >>> from ite.cost.x_kernel import Kernel
        >>> k1 = Kernel({'name': 'RBF','sigma': 1})
        >>> k2 = Kernel({'name': 'exponential','sigma': 1})
        >>> k3 = Kernel({'name': 'Cauchy','sigma': 1})
        >>> k4 = Kernel({'name': 'student','d': 1})
        >>> k5 = Kernel({'name': 'Matern3p2','l': 1})
        >>> k6 = Kernel({'name': 'Matern5p2','l': 1})
        >>> k7 = Kernel({'name': 'polynomial','exponent': 2,'c': 1})
        >>> k8 = Kernel({'name': 'ratquadr','c': 1})
        >>> k9 = Kernel({'name': 'invmquadr','c': 1})

        from numpy.random import rand
        num_of_samples, dim = 5, 2
        y1, y2 = rand(num_of_samples, dim), rand(num_of_samples+1, dim)
        y1b = rand(num_of_samples, dim)
        k1.gram_matrix1(y1)
        k1.gram_matrix2(y1, y2)
        k1.sum(y1,y1b)
        k1.gram_matrix_diagonal(y1)

        """

        # if par is None:
        #     par = {'name': 'RBF', 'sigma': 0.01}
        if par is None:
            par = {'name': 'RBF', 'sigma': 1}
        # if par is None:
        #    par = {'name': 'exponential', 'sigma': 1}
        # if par is None:
        #     par = {'name': 'Cauchy', 'sigma': 1}
        # if par is None:
        #     par = {'name': 'student', 'd': 1}
        # if par is None:
        #     par = {'name': 'Matern3p2', 'l': 1}
        # if par is None:
        #     par = {'name': 'Matern5p2', 'l': 1}
        # if par is None:
        #     par = {'name': 'polynomial','exponent': 2, 'c': 1}
        # if par is None:
        #     par = {'name': 'polynomial', 'exponent': 3, 'c': 1}
        # if par is None:
        #     par = {'name': 'ratquadr', 'c': 1}
        # if par is None:
        #     par = {'name': 'invmquadr', 'c': 1}

        # name:
        name = par['name']
        self.name = name

        # other attributes:
        if name == 'RBF' or name == 'exponential' or name == 'Cauchy':
            self.sigma = par['sigma']
        elif name == 'student':
            self.d = par['d']
        elif name == 'Matern3p2' or name == 'Matern5p2':
            self.l = par['l']
        elif name == 'polynomial':
            self.c = par['c']
            self.exponent = par['exponent']
        elif name == 'ratquadr' or name == 'invmquadr':
            self.c = par['c']
        else:
            raise Exception('kernel=?')

    def __str__(self):
        """ String representation of the kernel.

        Examples
        --------
        print(kernel)

        """

        return ''.join((self.__class__.__name__, ' -> ',
                        str(self.__dict__)))

    def gram_matrix1(self, y):
        """  Compute the Gram matrix = [k(y[i,:],y[j,:])]; i, j: running.

        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
            One row of y corresponds to one sample.

        Returns
        -------
        g : ndarray.
            Gram matrix of y.

        Examples
        --------
        g = k.gram_matrix1(y)

        """

        if self.name == 'RBF':
            sigma = self.sigma
            g = squareform(pdist(y))
            g = exp(-g ** 2 / (2 * sigma ** 2))
        elif self.name == 'exponential':
            sigma = self.sigma
            g = squareform(pdist(y))
            g = exp(-g / (2 * sigma ** 2))
        elif self.name == 'Cauchy':
            sigma = self.sigma
            g = squareform(pdist(y))
            g = 1 / (1 + g ** 2 / sigma ** 2)
        elif self.name == 'student':
            d = self.d
            g = squareform(pdist(y))
            g = 1 / (1 + g ** d)
        elif self.name == 'Matern3p2':
            l = self.l
            g = squareform(pdist(y))
            g = (1 + sqrt(3) * g / l) * exp(-sqrt(3) * g / l)
        elif self.name == 'Matern5p2':
            l = self.l
            g = squareform(pdist(y))
            g = (1 + sqrt(5) * g / l + 5 * g ** 2 / (3 * l ** 2)) * \
                exp(-sqrt(5) * g / l)
        elif self.name == 'polynomial':
            c = self.c
            exponent = self.exponent
            g = (dot(y, y.T) + c) ** exponent
        elif self.name == 'ratquadr':
            c = self.c
            g = squareform(pdist(y)) ** 2
            g = 1 - g / (g + c)
        elif self.name == 'invmquadr':
            c = self.c
            g = squareform(pdist(y))
            g = 1 / sqrt(g ** 2 + c ** 2)
        else:
            raise Exception('kernel=?')

        return g

    def gram_matrix2(self, y1, y2):
        """  Compute the Gram matrix = [k(y1[i,:],y2[j,:])]; i, j: running.

        Parameters
        ----------
        y1 : (number of samples1, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples2, dimension)-ndarray
             One row of y2 corresponds to one sample.

        Returns
        -------
        g : ndarray.
            Gram matrix of y1 and y2.

        Examples
        --------
        g = k.gram_matrix2(y1,y2)

        """

        if self.name == 'RBF':
            sigma = self.sigma
            g = cdist(y1, y2)  # alternative: g = cdist_large_dim(y1,y2)
            g = exp(-g ** 2 / (2 * sigma ** 2))
        elif self.name == 'exponential':
            sigma = self.sigma
            g = cdist(y1, y2)  # alternative: g = cdist_large_dim(y1,y2)
            g = exp(-g / (2 * sigma ** 2))
        elif self.name == 'Cauchy':
            sigma = self.sigma
            g = cdist(y1, y2)  # alternative: g = cdist_large_dim(y1,y2)
            g = 1 / (1 + g ** 2 / sigma ** 2)
        elif self.name == 'student':
            d = self.d
            g = cdist(y1, y2)  # alternative: g = cdist_large_dim(y1,y2)
            g = 1 / (1 + g ** d)
        elif self.name == 'Matern3p2':
            l = self.l
            g = cdist(y1, y2)  # alternative: g = cdist_large_dim(y1,y2)
            g = (1 + sqrt(3) * g / l) * exp(-sqrt(3) * g / l)
        elif self.name == 'Matern5p2':
            l = self.l
            g = cdist(y1, y2)  # alternative: g = cdist_large_dim(y1,y2)
            g = (1 + sqrt(5) * g / l + 5 * g ** 2 / (3 * l ** 2)) * \
                exp(-sqrt(5) * g / l)
        elif self.name == 'polynomial':
            c = self.c
            exponent = self.exponent
            g = (dot(y1, y2.T) + c) ** exponent
        elif self.name == 'ratquadr':
            c = self.c
            # alternative: g = cdist_large_dim(y1,y2)**2
            g = cdist(y1, y2) ** 2
            g = 1 - g / (g + c)
        elif self.name == 'invmquadr':
            c = self.c
            g = cdist(y1, y2)  # alternative: g = cdist_large_dim(y1,y2)
            g = 1 / sqrt(g ** 2 + c ** 2)
        else:
            raise Exception('kernel=?')

        return g

    def sum(self, y1, y2):
        """ Compute \sum_i k(y1[i,:],y2[i,:]).

        Parameters
        ----------
        y1 : (number of samples, dimension)-ndarray
             One row of y1 corresponds to one sample.
        y2 : (number of samples, dimension)-ndarray
             One row of y2 corresponds to one sample. There has to be the
             same number of samples in y1 and y2.

        Returns
        -------
        s : float
            s = \sum_i k(y1[i,:],y2[i,:]).

        """

        # verification:
        if y1.shape[0] != y1.shape[0]:
            raise Exception('There should be the same number of samples '
                            'in y1 and y2!')

        if self.name == 'RBF':
            sigma = self.sigma
            dist2 = sum((y1 - y2) ** 2, axis=1)
            s = sum(exp(-dist2 / (2 * sigma ** 2)))
        elif self.name == 'exponential':
            sigma = self.sigma
            dist = sqrt(sum((y1 - y2) ** 2, axis=1))
            s = sum(exp(-dist / (2 * sigma ** 2)))
        elif self.name == 'Cauchy':
            sigma = self.sigma
            dist2 = sum((y1 - y2) ** 2, axis=1)
            s = sum(1 / (1 + dist2 / sigma ** 2))
        elif self.name == 'student':
            d = self.d
            dist2 = sqrt(sum((y1 - y2) ** 2, axis=1))
            s = sum(1 / (1 + dist2 ** d))
        elif self.name == 'Matern3p2':
            l = self.l
            dist = sqrt(sum((y1 - y2) ** 2, axis=1))
            s = sum((1 + sqrt(3) * dist / l) * exp(-sqrt(3) * dist / l))
        elif self.name == 'Matern5p2':
            l = self.l
            dist = sqrt(sum((y1 - y2) ** 2, axis=1))
            s = sum((1 + sqrt(5) * dist / l + 5 * dist ** 2 /
                     (3 * l ** 2)) * exp(-sqrt(5) * dist / l))
        elif self.name == 'polynomial':
            c = self.c
            exponent = self.exponent
            s = sum((sum(y1 * y2, axis=1) + c) ** exponent)
        elif self.name == 'ratquadr':
            c = self.c
            dist2 = sum((y1 - y2) ** 2, axis=1)
            s = sum(1 - dist2 / (dist2 + c))
        elif self.name == 'invmquadr':
            c = self.c
            dist2 = sum((y1 - y2) ** 2, axis=1)
            s = sum(1 / sqrt(dist2 + c ** 2))
        else:
            raise Exception('kernel=?')

        return s

    def gram_matrix_diagonal(self, y):
        """ Diagonal of the Gram matrix: [k(y[i,:],y[i,:])]; i is running.

        Parameters
        ----------
        y : (number of samples, dimension)-ndarray
             One row of y corresponds to one sample.

        Returns
        -------
        diag_g : num_of_samples-ndarray
                 Diagonal of the Gram matrix.

        """

        num_of_samples = y.shape[0]

        if self.name == 'RBF' or\
           self.name == 'exponential' or\
           self.name == 'Cauchy' or\
           self.name == 'student' or\
           self.name == 'ratquadr' or\
           self.name == 'Matern3p2' or\
           self.name == 'Matern5p2':
            diag_g = ones(num_of_samples, dtype='float')
        elif self.name == 'polynomial':
            diag_g = (sum(y**2, axis=1) + self.c)**self.exponent
        elif self.name == 'invmquadr':
            diag_g = ones(num_of_samples, dtype='float') / self.c
        else:
            raise Exception('kernel=?')

        return diag_g

    def ichol(self, y, tol):
        """ Incomplete Cholesky decomposition defined by the data & kernel.

        If 'a' is the true Gram matrix: a \approx dot(g_hat, g_hat.T).

        Parameters
        ----------
        y   : (number of samples, dimension)-ndarray
        tol : float, > 0
              Tolerance parameter; smaller 'tol' means larger sized Gram
              factor and better approximation.

        Returns
        -------
        g_hat : (number_of_samples, smaller dimension)-ndarray
                Incomplete Cholesky(/Gram) factor.

        Notes
        -----
        Symmetric pivoting is used and the algorithms stops when the sum
        of the remaining pivots is less than 'tol'.

        This function is a Python implementation for general kernels of
        'chol_gauss.m', 'chol_poly.m', 'chol_hermite.m' which were written
        by Francis Bach for the TCA topic (see
        "http://www.di.ens.fr/~fbach/tca/tca1_0.tar.gz").

        References
        ----------
        Francis R. Bach, Michael I. Jordan. Beyond independent components:
        trees and clusters. Journal of Machine Learning Research, 4:1205-
        1233, 2003.

        """

        num_of_samples = y.shape[0]
        pvec = array(range(num_of_samples))
        g_diag = self.gram_matrix_diagonal(y)
        # .copy(): so that if 'g_diag' changes 'k_diag' should not also do
        # so:
        k_diag = g_diag.copy()
        # 'i' and 'jast' follow 'Matlab indexing'; the rest is adjusted
        # accordingly:

        i = 1

        while sum(g_diag[i-1:num_of_samples]) > tol:
            # g update with a new zero column to the right:
            if i == 1:
                g = zeros((num_of_samples, 1))
            else:
                g = hstack((g, zeros((num_of_samples, 1))))

            if i > 1:
                # argmax returns the index of the (first) max
                jast = argmax(g_diag[i-1:num_of_samples]) + (i - 1) + 1
                pvec[i-1], pvec[jast-1] = pvec[jast-1], pvec[i-1]
                # swap the 2 rows of g:
                t = copy(g[i-1])  # broadcast
                g[i-1] = g[jast-1]  # broadcast
                g[jast-1] = t  # broadcast
            else:
                jast = 1

            g[i-1, i-1] = sqrt(g_diag[jast-1])
            if i < num_of_samples:
                # with broadcasting:
                yq = y[pvec[i-1]]
                newacol = \
                    self.gram_matrix2(y[pvec[i:num_of_samples]],
                                      yq[newaxis, :]).T
                if i > 1:
                    g[i:num_of_samples, i-1] = \
                        1 / g[i-1, i-1] * \
                        (newacol - dot(g[i:num_of_samples, 0:i-1],
                                       g[i-1, 0:i-1].T))
                else:
                    g[i:num_of_samples, i-1] = 1 / g[i-1, i-1] * newacol

            if i < num_of_samples:
                g_diag[i:num_of_samples] = \
                    k_diag[pvec[i:num_of_samples]] \
                    - sum(g[i:num_of_samples]**2, axis=1)  # broadcast

            i += 1

        # permute the rows of 'g' in accord with 'pvec':
        pvec = argsort(pvec)

        return g[pvec]  # broadcast
