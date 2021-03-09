""" Factory for information theoretical estimators.

For entropy / mutual information / divergence / cross quantity /
association / distribution kernel estimators.

"""

# assumption: the 'cost_name' entropy estimator is in module 'ite.cost'
import ite.cost


def co_factory(cost_name, **kwargs):
    """ Creates any entropy / mutual information / divergence / cross 
    quantity / association / distribution kernel estimator by its name and
    its parameters.
    
    Parameters
    ----------
    cost_name : str
                Name of the cost object to be created.
    kwargs : dictionary
             It can be used to override default parameter values in 
             the estimator (if needed).
    Returns
    -------
    co : class 
         Initialized estimator (cost object).

    Examples
    --------
    >>> import ite
    >>> co1 = ite.cost.co_factory('BHShannon_KnnK')

    >>> dict_par = {'mult': False,'k': 2}
    >>> co2 = ite.cost.co_factory('BHShannon_KnnK', **dict_par) # mapping\
                                                                # unpacking
    
    """
    
    co = getattr(ite.cost, cost_name)(**kwargs)     
    # print(co) # commented out so that doctests should not give errors
    
    return co
