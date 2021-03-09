""" Verification and exception classes for estimators.

In other words, for entropy / mutual information / divergence / cross
quantity / association / distribution kernel estimators.

The verification classes are not called directly, but they are used by
inheritance: the cost objects get them as method(s) for checking before
estimation; for example in case of divergence measures whether the samples
(in y1 and y2) have the same dimension. Each verification class is
accompanied by an exception class (ExceptionX, classX); if the required
property is violated (classX) and exception (ExceptionX) is raised.

"""


class ExceptionOneDSignal(Exception):
    """ Exception for VerOneDSignal '"""

    def __str__(self):
        return 'The samples must be one-dimensional for this estimator!'


class VerOneDSignal(object):
    """ Verification class with 'one-dimensional signal' capability. """

    def verification_one_d_signal(self, y):
        """ Verify if y is one-dimensional.

        If this is not the case, an ExceptionOneDSignal exception is
        raised.

        Examples
        --------
        >>> from numpy.random import rand
        >>> import ite
        >>> Ver = ite.cost.x_verification.VerOneDSignal() # <-> 'simple co'
        >>> y = rand(100,1) # 100 samples from an 1D random variable
        >>> Ver.verification_one_d_signal(y)

        """

        if (y.ndim != 2) or (y.shape[1] != 1):
            raise ExceptionOneDSignal()


class ExceptionOneDSubspaces(Exception):
    """ Exception for VerOneDSubspaces """

    def __str__(self):
        return 'The subspaces must be one-dimensional for this estimator!'


class VerOneDSubspaces(object):
    """ Verification class with 'one-dimensional subspaces' capability. """

    def verification_one_dimensional_subspaces(self, ds):
        """ Verify if ds encodes one-dimensional subspaces.

        If this is not the case, an ExceptionOneDSubspaces exception is
        raised.

        Examples
        --------
        >>> from numpy import ones
        >>> import ite
        >>> Ver = ite.cost.x_verification.VerOneDSubspaces() # 'simple co'
        >>> ds = ones(4)
        >>> Ver.verification_one_dimensional_subspaces(ds)

        """

        if not(all(ds == 1)):
            raise ExceptionOneDSubspaces()


class ExceptionCompSubspaceDims(Exception):
    """ Exception for VerCompSubspaceDims """

    def __str__(self):
        return 'The subspace dimensions are not compatible with y!'


class VerCompSubspaceDims(object):
    """ Verification with 'compatible subspace dimensions' capability.

    """

    def verification_compatible_subspace_dimensions(self, y, ds):
        """ Verify if y and ds are compatible.

        If this is not the case, an ExceptionCompSubspaceDims exception is
        raised.

        Examples
        --------
        >>> from numpy import array
        >>> from numpy.random import rand
        >>> import ite
        >>> Ver = ite.cost.x_verification.VerCompSubspaceDims() # simple co
        >>> ds = array([2, 2]) # 2 pieces of 2-dimensional subspaces
        >>> y = rand(100, 4)
        >>> Ver.verification_compatible_subspace_dimensions(y, ds)

        """

        if y.shape[1] != sum(ds):
            raise ExceptionCompSubspaceDims()


class ExceptionSubspaceNumberIsK(Exception):
    """ Exception for VerSubspaceNumberIsK """

    def __init__(self, k):
        self.k = k

    def __str__(self):
        return 'The number of subspaces must be ' + str(self.k) + \
                ' for this estimator!'


class VerSubspaceNumberIsK(object):
    """ Verification class with 'the # of subspaces is k' capability. """

    def verification_subspace_number_is_k(self, ds, k):
        """ Verify if the number of subspaces is k.

        If this is not the case, an ExceptionSubspaceNumberIsK exception is
        raised.

        Examples
        --------
        >>> from numpy import array
        >>> from numpy.random import rand
        >>> import ite
        >>> Ver = ite.cost.x_verification.VerSubspaceNumberIsK() # 'co'
        >>> ds = array([3, 3]) # 2 pieces of 3-dimensional subspaces
        >>> y = rand(1000, 6)
        >>> Ver.verification_subspace_number_is_k(ds, 2)

        """

        if len(ds) != k:
            raise ExceptionSubspaceNumberIsK(k)


class ExceptionEqualDSubspaces(Exception):
    """ Exception for VerEqualDSubspaces """

    def __str__(self):
        return 'The dimension of the samples in y1 and y2 must be equal!'


class VerEqualDSubspaces(object):
    """ Verification class with 'equal subspace dimensions' capability. """

    def verification_equal_d_subspaces(self, y1, y2):
        """ Verify if y1 and y2 have the same dimensionality.

        If this is not the case, an ExceptionEqualDSubspaces exception is
        raised.

        Examples
        --------
        >>> from numpy.random import rand
        >>> import ite
        >>> Ver = ite.cost.x_verification.VerEqualDSubspaces() # 'co'
        >>> y1 = rand(100, 2)
        >>> y2 = rand(200, 2)
        >>> Ver.verification_equal_d_subspaces(y1, y2)

        """

        d1, d2 = y1.shape[1], y2.shape[1]

        if d1 != d2:
            raise ExceptionEqualDSubspaces()


class ExceptionEqualSampleNumbers(Exception):
    """ Exception for VerEqualSampleNumbers """

    def __str__(self):
        return 'There must be equal number of samples in y1 and' + \
               ' y2 for this estimator!'


class VerEqualSampleNumbers(object):
    """ Verification class with 'the # of samples is equal' capability. """

    def verification_equal_sample_numbers(self, y1, y2):
        """ Verify if y1 and y2 have the same dimensionality.

        If this is not the case, an ExceptionEqualDSubspaces exception is
        raised.

        Examples
        --------
        >>> from numpy.random import rand
        >>> import ite
        >>> Ver = ite.cost.x_verification.VerEqualSampleNumbers() # 'co'
        >>> y1 = rand(100, 2)
        >>> y2 = rand(100, 2)
        >>> Ver.verification_equal_sample_numbers(y1, y2)

        """

        num_of_samples1, num_of_samples2 = y1.shape[0], y2.shape[0]

        if num_of_samples1 != num_of_samples2:
            raise ExceptionEqualSampleNumbers()


class ExceptionEvenSampleNumbers(Exception):
    """ Exception for VerEvenSampleNumbers """

    def __str__(self):
        return 'The number of samples must be even for this' +\
               ' estimator!'


class VerEvenSampleNumbers(object):
    """ Verification class with 'even sample numbers' capability.

    Assumption: y1.shape[0] = y2.shape[0]. (see class
    'VerEqualSampleNumbers' above)

    """

    def verification_even_sample_numbers(self, y1):
        """
        Examples
        --------
        >>> from numpy.random import rand
        >>> import ite
        >>> Ver = ite.cost.x_verification.VerEvenSampleNumbers() # 'co'
        >>> y1 = rand(100, 2)
        >>> y2 = rand(100, 2)
        >>> Ver.verification_even_sample_numbers(y1)
        """

        num_of_samples = y1.shape[0]  # = y2.shape[0] by assumption
        if num_of_samples % 2 != 0:  # check if num_of_samples is even
            raise ExceptionEvenSampleNumbers()

# Template:
# class ExceptionX(Exception):
#    """ Exception for  X """
#
#    def __str__(self):
#        return 'XY'
#
