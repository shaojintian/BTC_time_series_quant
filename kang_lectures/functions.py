"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from joblib import wrap_non_picklable_objects
import talib

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def make_function(*, function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)
    
    # np.log1p(x) log(e,(x + 1)) == np.nan = 0

def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def norm(x):
    factors_data = pd.DataFrame(x, columns=['factor'])
    factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
    factors_mean = factors_data.cumsum() / np.arange(1, factors_data.shape[0] + 1)[:, np.newaxis]
    factors_std = factors_data.expanding().std()
    factor_value = (factors_data - factors_mean) / factors_std
    factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
    factor_value = factor_value.clip(-6, 6)
    x = np.nan_to_num(factor_value['factor'].values)
    return x


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return norm(1 / (1 + np.exp(-x1)))

def _tanh(x1):
    with np.errstate(over='ignore', under='ignore'):
        return norm(np.tanh(x1))

def _elu(x1):
    with np.errstate(over='ignore', under='ignore'):
        x = (np.where(x1 > 0, x1, 1 * (np.exp(x1) - 1)))
        return norm(x)

# close,2.6 volume-5000 量纲的差别--作业2：各位详细的理解和体会一下量纲这个物理学的概念
# 均值为0，方差为1的数据，

def _ta_ht_trendline(x1):
    x1 = x1.flatten()
    x = (np.nan_to_num(talib.HT_TRENDLINE(x1)))
    return norm(x)

def _ta_ht_dcperiod(x1):
    x1 = x1.flatten()
    x = (np.nan_to_num(talib.HT_DCPERIOD(x1)))
    return norm(x)


def _ta_ht_dcphase(x1):
    x1 = x1.flatten()
    x = (np.nan_to_num(talib.HT_DCPHASE(x1)))
    return norm(x)

def _ta_sar(x1, x2):
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = (np.nan_to_num(talib.SAR(x1, x2)))
    return norm(x)


def _ta_bop(x1, x2, x3, x4):
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x4 = x4.flatten()
    x = (np.nan_to_num(talib.BOP(x1, x2, x3, x4)))
    return norm(x)

def _ta_bop_0(x1, x2, x3, x4):
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x4 = x4.flatten()
    x = (np.nan_to_num(talib.BOP(x1, x2, x3, x4)))
    return norm(x)



def _ta_ad(x1, x2, x3, x4):
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x4 = x4.flatten()
    x = (np.nan_to_num(talib.AD(x1, x2, x3, x4)))
    return norm(x)

# ma(df.close, 8)

def _ta_obv(x1, x2):
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = (np.nan_to_num(talib.OBV(x1, x2)))
    return norm(x)


def _ta_trange(x1, x2, x3):
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = (np.nan_to_num(talib.TRANGE(x1, x2, x3)))
    return norm(x)

# 截至20230522只有这些因子，需要把带有t的引进

def _ts_cov_20(x1, x2):
    t = 20
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = norm(np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).cov(pd.Series(x2))))
    return norm(x)

def _ts_cov_40(x1, x2):
    t = 40
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = (np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).cov(pd.Series(x2))))
    return norm(x)

def _ts_corr_20(x1, x2):
    t = 20
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = (np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).corr(pd.Series(x2))))
    return norm(x)

def _ts_corr_40(x1, x2):
    t = 40
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = (np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).corr(pd.Series(x2))))
    return norm(x)

def _ts_day_min_10(x1):  # the i_th element is the interval between the min_value time point and t_i in the n-period time series from the past (t_i is not included)
    t = 10
    x1 = pd.Series(x1.flatten())
    x = (np.nan_to_num(x1.rolling(window=t+1).apply(lambda x: t - x.values.tolist()[: -1].index(min(x.values.tolist()[: -1])))))
    return norm(x)

# sliding_window_view 
from numpy.lib.stride_tricks import sliding_window_view

def _ts_day_min_20(x1):  # the i_th element is the interval between the min_value time point and t_i in the n-period time series from the past (t_i is not included)
    t = 20
    x1 = pd.Series(x1.flatten())
    x = (np.nan_to_num(x1.rolling(window=t+1).apply(lambda x: t - x.values.tolist()[: -1].index(min(x.values.tolist()[: -1])))))
    return norm(x)

def _ts_day_min_40(x1):  # the i_th element is the interval between the min_value time point and t_i in the n-period time series from the past (t_i is not included)
    t = 40
    x1 = pd.Series(x1.flatten())
    x = (np.nan_to_num(x1.rolling(window=t+1).apply(lambda x: t - x.values.tolist()[: -1].index(min(x.values.tolist()[: -1])))))
    return norm(x)

def _ts_day_max_10(x1):  # the i_th element is the interval between the max_value time point and t_i in the n-period time series from the past (t_i is not included)
    t = 10
    x1 = pd.Series(x1.flatten())
    x = (np.nan_to_num(x1.rolling(window=t+1).apply(lambda x: t - x.values.tolist()[: -1].index(max(x.values.tolist()[: -1])))))
    return norm(x)

def _ts_day_max_20(x1):  # the i_th element is the interval between the max_value time point and t_i in the n-period time series from the past (t_i is not included)
    t = 20
    x1 = pd.Series(x1.flatten())
    x = (np.nan_to_num(x1.rolling(window=t+1).apply(lambda x: t - x.values.tolist()[: -1].index(max(x.values.tolist()[: -1])))))
    return norm(x)

def _ts_day_max_40(x1):  # the i_th element is the interval between the max_value time point and t_i in the n-period time series from the past (t_i is not included)
    t = 40
    x1 = pd.Series(x1.flatten())
    x = (np.nan_to_num(x1.rolling(window=t+1).apply(lambda x: t - x.values.tolist()[: -1].index(max(x.values.tolist()[: -1])))))
    return norm(x)

def _ts_sma_8(x1):  # the i_th element is the simple moving average of the elements in the n-period time series from the past
    t = 8
    x1 = x1.flatten()
    x = (np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).mean()))
    return norm(x)

def _ts_sma_21(x1):  # the i_th element is the simple moving average of the elements in the n-period time series from the past
    t = 21
    x1 = x1.flatten()
    x = (np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).mean()))
    return norm(x)

def _ts_sma_55(x1):  # the i_th element is the simple moving average of the elements in the n-period time series from the past
    t = 55
    x1 = x1.flatten()
    x = (np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).mean()))
    return norm(x)

def _ts_wma_8(x1):  # the i_th element is the weighted moving average of the elements in the n-period time series from the past
    t = 8
    x1 = x1.flatten()
    weight_list = np.arange(1, t + 1)
    weight_list = weight_list / np.sum(weight_list)
    x = np.nan_to_num(pd.Series(x1).rolling(window=t).apply(lambda price_list: np.dot(price_list, weight_list), raw=True))
    return norm(x)

def _ts_wma_21(x1):  # the i_th element is the weighted moving average of the elements in the n-period time series from the past
    t = 21
    x1 = x1.flatten()
    weight_list = np.arange(1, t + 1)
    weight_list = weight_list / np.sum(weight_list)
    x = np.nan_to_num(pd.Series(x1).rolling(window=t).apply(lambda price_list: np.dot(price_list, weight_list), raw=True))
    return norm(x)

def _ts_wma_55(x1):  # the i_th element is the weighted moving average of the elements in the n-period time series from the past
    t = 55
    x1 = x1.flatten()
    weight_list = np.arange(1, t + 1)
    weight_list = weight_list / np.sum(weight_list)
    x = np.nan_to_num(pd.Series(x1).rolling(window=t).apply(lambda price_list: np.dot(price_list, weight_list), raw=True))
    return norm(x)

def _ts_lag_3(x1):
    t = 3
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).shift(periods=t))
    return norm(x)

def _ts_lag_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).shift(periods=t))
    return norm(x)

def _ts_lag_17(x1):
    t = 17
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).shift(periods=t))
    return norm(x)

def _ts_delta_3(x1):
    t = 3
    x1 = x1.flatten()
    x = np.nan_to_num(x1 - np.nan_to_num(pd.Series(x1).shift(periods=t)))
    return norm(x)

def _ts_delta_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(x1 - np.nan_to_num(pd.Series(x1).shift(periods=t)))
    return norm(x)

def _ts_delta_17(x1):
    t = 17
    x1 = x1.flatten()
    x = np.nan_to_num(x1 - np.nan_to_num(pd.Series(x1).shift(periods=t)))
    return norm(x)

def _ts_sum_3(x1):  # the i_th element is the sum of the elements in the n-period time series from the past
    t = 3
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.sum))
    return norm(x)

def _ts_sum_8(x1):  # the i_th element is the sum of the elements in the n-period time series from the past
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.sum))
    return norm(x)

def _ts_sum_17(x1):  # the i_th element is the sum of the elements in the n-period time series from the past
    t = 17
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.sum))
    return norm(x)

def _ts_prod_3(x1):  # the i_th element is the production of the elements in the n-period time series from the past
    t = 3
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.prod))
    return norm(x)

def _ts_prod_8(x1):  # the i_th element is the production of the elements in the n-period time series from the past
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.prod))
    return norm(x)

def _ts_prod_17(x1):  # the i_th element is the production of the elements in the n-period time series from the past
    t = 17
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.prod))
    return norm(x)

def _ts_std_10(x1):  # the i_th element is the standard deviation of the elements in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).std())
    return norm(x)

def _ts_std_20(x1):  # the i_th element is the standard deviation of the elements in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).std())
    return norm(x)

def _ts_std_40(x1):  # the i_th element is the standard deviation of the elements in the n-period time series from the past
    t = 40
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).std())
    return norm(x)

def _ts_skew_10(x1):  # the i_th element is the skewness of the elements in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).skew())
    return norm(x)

def _ts_skew_20(x1):  # the i_th element is the skewness of the elements in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).skew())
    return norm(x)

def _ts_skew_40(x1):  # the i_th element is the skewness of the elements in the n-period time series from the past
    t = 40
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).skew())
    return norm(x)

def _ts_kurt_10(x1):  # the i_th element is the kurtosis of the elements in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).kurt())
    return norm(x)

def _ts_kurt_20(x1):  # the i_th element is the kurtosis of the elements in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).kurt())
    return norm(x)

def _ts_kurt_40(x1):  # the i_th element is the kurtosis of the elements in the n-period time series from the past
    t = 40
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).kurt())
    return norm(x)

def _ts_min_5(x1):  # the i_th element is the minimum value in the n-period time series from the past
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return norm(x)

def _ts_min_10(x1):  # the i_th element is the minimum value in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return norm(x)

def _ts_min_20(x1):  # the i_th element is the minimum value in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return norm(x)

def _ts_max_5(x1):  # the i_th element is the maximum value in the n-period time series from the past
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max())
    return norm(x)

def _ts_max_10(x1):  # the i_th element is the maximum value in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max())
    return norm(x)

def _ts_max_20(x1):  # the i_th element is the maximum value in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max())
    return norm(x)

def _ts_range_5(x1):
    t = 5
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max()) - np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return norm(x)

def _ts_range_10(x1):
    t = 10
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max()) - np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return norm(x)

def _ts_range_20(x1):
    t = 20
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max()) - np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return norm(x)

def _ts_argmin_5(x1):  # the i_th element is the location of the minimum value in the n-period time series from the past
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmin) + 1)
    return norm(x)

def _ts_argmin_10(x1):  # the i_th element is the location of the minimum value in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmin) + 1)
    return norm(x)

def _ts_argmin_20(x1):  # the i_th element is the location of the minimum value in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmin) + 1)
    return norm(x)

def _ts_argmax_5(x1):  # the i_th element is the location of the maximum value in the n-period time series from the past
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmax) + 1)
    return norm(x)

def _ts_argmax_10(x1):  # the i_th element is the location of the maximum value in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmax) + 1)
    return norm(x)

def _ts_argmax_20(x1):  # the i_th element is the location of the maximum value in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmax) + 1)
    return norm(x)

def _ts_argrange_5(x1):
    t = 5
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmin) + 1)
    return norm(x)

def _ts_argrange_10(x1):
    t = 10
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmin) + 1)
    return norm(x)

def _ts_argrange_20(x1):
    t = 20
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(np.argmin) + 1)
    return norm(x)

def _ts_rank_5(x1):  # the i_th element is the quantile of the last element in the n-period time series from the past
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(lambda x: x.rank(pct=True).values[-1]))
    return norm(x)

def _ts_rank_10(x1):  # the i_th element is the quantile of the last element in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(lambda x: x.rank(pct=True).values[-1]))
    return norm(x)

def _ts_rank_20(x1):  # the i_th element is the quantile of the last element in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(lambda x: x.rank(pct=True).values[-1]))
    return norm(x)

def _ts_mean_return_5(x1):
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(lambda x: x.pct_change(1).mean()))
    return norm(x)

def _ts_mean_return_10(x1):
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(lambda x: x.pct_change(1).mean()))
    return norm(x)

def _ts_mean_return_20(x1):
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).apply(lambda x: x.pct_change(1).mean()))
    return norm(x)

def _ta_beta_5(x1, x2):
    t = 5
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(talib.BETA(x1, x2, timeperiod=t))
    return norm(x)

def _ta_beta_10(x1, x2):
    t = 10
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(talib.BETA(x1, x2, timeperiod=t))
    return norm(x)

def _ta_beta_20(x1, x2):
    t = 20
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(talib.BETA(x1, x2, timeperiod=t))
    return norm(x)

def _ta_lr_slope_5(x1):
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_SLOPE(x1, timeperiod=t))
    return norm(x)

def _ta_lr_slope_10(x1):
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_SLOPE(x1, timeperiod=t))
    return norm(x)

def _ta_lr_slope_20(x1):
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_SLOPE(x1, timeperiod=t))
    return norm(x)

def _ta_lr_intercept_5(x1):
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_INTERCEPT(x1, timeperiod=t))
    return norm(x)

def _ta_lr_intercept_10(x1):
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_INTERCEPT(x1, timeperiod=t))
    return norm(x)

def _ta_lr_intercept_20(x1):
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_INTERCEPT(x1, timeperiod=t))
    return norm(x)

def _ta_lr_angle_5(x1):
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_ANGLE(x1, timeperiod=t))
    return norm(x)

def _ta_lr_angle_10(x1):
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_ANGLE(x1, timeperiod=t))
    return norm(x)

def _ta_lr_angle_20(x1):
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_ANGLE(x1, timeperiod=t))
    return norm(x)

def _ta_tsf_5(x1):
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TSF(x1, timeperiod=t))
    return norm(x)

def _ta_tsf_10(x1):
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TSF(x1, timeperiod=t))
    return norm(x)

def _ta_tsf_20(x1):
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TSF(x1, timeperiod=t))
    return norm(x)

def _ta_ema_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(talib.EMA(x1, timeperiod=t))
    return norm(x)

def _ta_ema_21(x1):
    t = 21
    x1 = x1.flatten()
    x = np.nan_to_num(talib.EMA(x1, timeperiod=t))
    return norm(x)

def _ta_ema_55(x1):
    t = 55
    x1 = x1.flatten()
    x = np.nan_to_num(talib.EMA(x1, timeperiod=t))
    return norm(x)

def _ta_dema_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(talib.DEMA(x1, timeperiod=t))
    return norm(x)

def _ta_dema_21(x1):
    t = 21
    x1 = x1.flatten()
    x = np.nan_to_num(talib.DEMA(x1, timeperiod=t))
    return norm(x)

def _ta_dema_55(x1):
    t = 55
    x1 = x1.flatten()
    x = np.nan_to_num(talib.DEMA(x1, timeperiod=t))
    return norm(x)

def _ta_kama_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(talib.KAMA(x1, timeperiod=t))
    return norm(x)

def _ta_kama_21(x1):
    t = 21
    x1 = x1.flatten()
    x = np.nan_to_num(talib.KAMA(x1, timeperiod=t))
    return norm(x)

def _ta_kama_55(x1):
    t = 55
    x1 = x1.flatten()
    x = np.nan_to_num(talib.KAMA(x1, timeperiod=t))
    return norm(x)

def _ta_tema_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TEMA(x1, timeperiod=t))
    return norm(x)

def _ta_tema_21(x1):
    t = 21
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TEMA(x1, timeperiod=t))
    return norm(x)

def _ta_tema_55(x1):
    t = 55
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TEMA(x1, timeperiod=t))
    return norm(x)

def _ta_trima_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TRIMA(x1, timeperiod=t))
    return norm(x)

def _ta_trima_21(x1):
    t = 21
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TRIMA(x1, timeperiod=t))
    return norm(x)

def _ta_trima_55(x1):
    t = 55
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TRIMA(x1, timeperiod=t))
    return norm(x)

def _ta_rsi_6(x1):
    t = 6
    x1 = x1.flatten()
    x = np.nan_to_num(talib.RSI(x1, timeperiod=t))
    return norm(x)

def _ta_rsi_12(x1):
    t = 12
    x1 = x1.flatten()
    x = np.nan_to_num(talib.RSI(x1, timeperiod=t))
    return norm(x)

def _ta_rsi_24(x1):
    t = 24
    x1 = x1.flatten()
    x = np.nan_to_num(talib.RSI(x1, timeperiod=t))
    return norm(x)

def _ta_cmo_14(x1):
    t = 14
    x1 = x1.flatten()
    x = np.nan_to_num(talib.CMO(x1, timeperiod=t))
    return norm(x)

def _ta_cmo_25(x1):
    t = 25
    x1 = x1.flatten()
    x = np.nan_to_num(talib.CMO(x1, timeperiod=t))
    return norm(x)

def _ta_mom_12(x1):
    t = 12
    x1 = x1.flatten()
    x = np.nan_to_num(talib.MOM(x1, timeperiod=t))
    return norm(x)

def _ta_mom_25(x1):
    t = 25
    x1 = x1.flatten()
    x = np.nan_to_num(talib.MOM(x1, timeperiod=t))
    return norm(x)

# def _ta_roc_14(x1):
#     t = 14
#     x1 = x1.flatten()
#     return np.nan_to_num(talib.ROC(x1, timeperiod=t))

# def _ta_roc_25(x1):
#     t = 25
#     x1 = x1.flatten()
#     return np.nan_to_num(talib.ROC(x1, timeperiod=t))

def _ta_rocp_14(x1):
    t = 14
    x1 = x1.flatten()
    x = np.nan_to_num(talib.ROCP(x1, timeperiod=t))
    return norm(x)

def _ta_rocp_25(x1):
    t = 25
    x1 = x1.flatten()
    x = np.nan_to_num(talib.ROCP(x1, timeperiod=t))
    return norm(x)

def _ta_rocr_14(x1):
    t = 14
    x1 = x1.flatten()
    x = np.nan_to_num(talib.ROCR(x1, timeperiod=t))
    return norm(x)

def _ta_rocr_25(x1):
    t = 25
    x1 = x1.flatten()
    x = np.nan_to_num(talib.ROCR(x1, timeperiod=t))
    return norm(x)

def _ta_trix_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TRIX(x1, timeperiod=t))
    return norm(x)

def _ta_trix_21(x1):
    t = 21
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TRIX(x1, timeperiod=t))
    return norm(x)

def _ta_trix_55(x1):
    t = 55
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TRIX(x1, timeperiod=t))
    return norm(x)

def _ta_adx_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.ADX(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_adx_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.ADX(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_adxr_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.ADXR(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_adxr_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.ADXR(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_aroonosc_14(x1, x2):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(talib.AROONOSC(x1, x2, timeperiod=t))
    return norm(x)

def _ta_aroonosc_25(x1, x2):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(talib.AROONOSC(x1, x2, timeperiod=t))
    return norm(x)

def _ta_cci_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.CCI(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_cci_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.CCI(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_dx_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.DX(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_dx_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.DX(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_mfi_14(x1, x2, x3, x4):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x4 = x4.flatten()
    x = np.nan_to_num(talib.MFI(x1, x2, x3, x4, timeperiod=t))
    return norm(x)

def _ta_mfi_25(x1, x2, x3, x4):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x4 = x4.flatten()
    x = np.nan_to_num(talib.MFI(x1, x2, x3, x4, timeperiod=t))
    return norm(x)

def _ta_minus_di_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.MINUS_DI(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_minus_di_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.MINUS_DI(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_minus_dm_14(x1, x2):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = norm(np.nan_to_num(talib.MINUS_DM(x1, x2, timeperiod=t)))
    return norm(x)

def _ta_minus_dm_25(x1, x2):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = norm(np.nan_to_num(talib.MINUS_DM(x1, x2, timeperiod=t)))
    return norm(x)

def _ta_willr_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.WILLR(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_willr_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.WILLR(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_atr_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.ATR(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_atr_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.ATR(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_natr_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.NATR(x1, x2, x3, timeperiod=t))
    return norm(x)

def _ta_natr_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.NATR(x1, x2, x3, timeperiod=t))
    return norm(x)


# 已加入因子：_ts_cov_20，_ts_cov_40, _ts_corr_20, _ts_corr_40, _ts_day_min_10, _ts_day_min_20, _ts_day_min_40
# _ts_day_max_10, _ts_day_max_20, _ts_day_max_40, _ts_sma_8, _ts_sma_21, _ts_sma_55
# _ts_wma_8, _ts_wma_21, _ts_wma_55, _ts_lag_3, _ts_lag_8, _ts_lag_17
# _ts_delta_3, _ts_delta_8, _ts_delta_17, _ts_sum_3, _ts_sum_8, _ts_sum_17
# _ts_prod_3, _ts_prod_8, _ts_prod_17, _ts_std_10, _ts_std_20, _ts_std_40
# _ts_skew_10, _ts_skew_20, _ts_skew_40, _ts_kurt_10, _ts_kurt_20, _ts_kurt_40
# _ts_min_5, _ts_min_10, _ts_min_20, _ts_max_5, _ts_max_10, _ts_max_20
# _ts_range_5, _ts_range_10, _ts_range_20, _ts_argmin_5, _ts_argmin_10, _ts_argmin_20
# _ts_argmax_5, _ts_argmax_10, _ts_argmax_20, _ts_argrange_5, _ts_argrange_10, _ts_argrange_20
# _ts_rank_5, _ts_rank_10, _ts_rank_20, _ts_mean_return_5, _ts_mean_return_10, _ts_mean_return_20
# _ta_beta_5, _ta_beta_10, _ta_beta_20,_ta_lr_intercept_5, _ta_lr_intercept_10, _ta_lr_intercept_20
# _ta_tsf_5, _ta_tsf_10, _ta_tsf_20,_ta_ema_8, _ta_ema_21, _ta_ema_55
# _ta_dema_8, _ta_dema_21, _ta_dema_55, _ta_kama_8, _ta_kama_21, _ta_kama_55
# _ta_tema_8, _ta_tema_21, _ta_tema_55, _ta_trima_8, _ta_trima_21, _ta_trima_55
# _ta_rsi_6, _ta_rsi_12, _ta_rsi_24, _ta_cmo_14, _ta_cmo_25, _ta_mom_12, _ta_mom_25
# _ta_roc_14, _ta_roc_25, _ta_rocp_14, _ta_rocp_25, _ta_rocr_14, _ta_rocr_25
# _ta_trix_8, _ta_trix_21, _ta_trix_55, _ta_adx_14, _ta_adx_25
# _ta_adxr_14, _ta_adxr_25, _ta_aroonosc_14, _ta_aroonosc_25, _ta_cci_14, _ta_cci_25
# _ta_dx_14, _ta_dx_25, _ta_mfi_14, _ta_mfi_25, _ta_minus_di_14, _ta_minus_di_25
# _ta_minus_dm_14, _ta_minus_dm_25, _ta_willr_14, _ta_willr_25, 
# _ta_atr_14, _ta_atr_25, _ta_natr_14, _ta_natr_25


add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)

# 增加部分的函数
tanh1 = _Function(function=_tanh, name='tanh', arity=1)
elu1 = _Function(function=_elu, name='elu', arity=1)
ta_ht_trendline = _Function(function=_ta_ht_trendline, name='TA_HT_TRENDLINE', arity=1)
ta_ht_dcperiod = _Function(function=_ta_ht_dcperiod, name='TA_HT_DCPERIOD', arity=1)
ta_ht_dcphase = _Function(function=_ta_ht_dcphase, name='TA_HT_DCPHASE', arity=1)
ta_sar = _Function(function=_ta_sar, name='TA_SAR', arity=2)
ta_bop = _Function(function=_ta_bop, name='TA_BOP', arity=4)
ta_ad = _Function(function=_ta_ad, name='TA_AD', arity=4)
ta_obv = _Function(function=_ta_obv, name='TA_OBV', arity=2)
ta_trange = _Function(function=_ta_trange, name='TA_TRANGE', arity=3)

# 5月23日加入的因子：
# 1-10:
ts_cov_20 = _Function(function=_ts_cov_20, name='TS_COV_20', arity=2)
ts_cov_40 = _Function(function=_ts_cov_40, name='TS_COV_40', arity=2)
ts_corr_20 = _Function(function=_ts_corr_20, name='TS_CORR_20', arity=2)
ts_corr_40 = _Function(function=_ts_corr_40, name='TS_CORR_40', arity=2)
ts_day_min_10 = _Function(function=_ts_day_min_10, name='TS_DAY_MIN_10', arity=1)
ts_day_min_20 = _Function(function=_ts_day_min_20, name='TS_DAY_MIN_20', arity=1)
ts_day_min_40 = _Function(function=_ts_day_min_40, name='TS_DAY_MIN_40', arity=1)
ts_day_max_10 = _Function(function=_ts_day_max_10, name='TS_DAY_MAX_10', arity=1)
ts_day_max_20 = _Function(function=_ts_day_max_20, name='TS_DAY_MAX_20', arity=1)
ts_day_max_40 = _Function(function=_ts_day_max_40, name='TS_DAY_MAX_40', arity=1)

# 11-19:
ts_sma_8 = _Function(function=_ts_sma_8, name='ts_sma_8', arity=1)
ts_sma_21 = _Function(function=_ts_sma_21, name='ts_sma_21', arity=1)
ts_sma_55 = _Function(function=_ts_sma_55, name='ts_sma_55', arity=1)
ts_wma_8 = _Function(function=_ts_wma_8, name='ts_wma_8', arity=1)
ts_wma_21 = _Function(function=_ts_wma_21, name='ts_wma_21', arity=1)
ts_wma_55 = _Function(function=_ts_wma_55, name='ts_wma_55', arity=1)
ts_lag_3 = _Function(function=_ts_lag_3, name='ts_lag_3', arity=1)
ts_lag_8 = _Function(function=_ts_lag_8, name='ts_lag_8', arity=1)
ts_lag_17 = _Function(function=_ts_lag_17, name='ts_lag_17', arity=1)

# 20-32
ts_delta_3 = _Function(function=_ts_delta_3, name='ts_delta_3', arity=1)
ts_delta_8 = _Function(function=_ts_delta_8, name='ts_delta_8', arity=1)
ts_delta_17 = _Function(function=_ts_delta_17, name='ts_delta_17', arity=1)
ts_sum_3 = _Function(function=_ts_sum_3, name='ts_sum_3', arity=1)
ts_sum_8 = _Function(function=_ts_sum_8, name='ts_sum_8', arity=1)
ts_sum_17 = _Function(function=_ts_sum_17, name='ts_sum_17', arity=1)
ts_prod_3 = _Function(function=_ts_prod_3, name='ts_prod_3', arity=1)
ts_prod_8 = _Function(function=_ts_prod_8, name='ts_prod_8', arity=1)
ts_prod_17 = _Function(function=_ts_prod_17, name='ts_prod_17', arity=1)
ts_std_10 = _Function(function=_ts_std_10, name='ts_std_10', arity=1)
ts_std_20 = _Function(function=_ts_std_20, name='ts_std_20', arity=1)
ts_std_40 = _Function(function=_ts_std_40, name='ts_std_40', arity=1)

# 33-44
ts_skew_10 = _Function(function=_ts_skew_10, name='ts_skew_10', arity=1)
ts_skew_20 = _Function(function=_ts_skew_20, name='ts_skew_20', arity=1)
ts_skew_40 = _Function(function=_ts_skew_40, name='ts_skew_40', arity=1)
ts_kurt_10 = _Function(function=_ts_kurt_10, name='ts_kurt_10', arity=1)
ts_kurt_20 = _Function(function=_ts_kurt_20, name='ts_kurt_20', arity=1)
ts_kurt_40 = _Function(function=_ts_kurt_40, name='ts_kurt_40', arity=1)
ts_min_5 = _Function(function=_ts_min_5, name='ts_min_5', arity=1)
ts_min_10 = _Function(function=_ts_min_10, name='ts_min_10', arity=1)
ts_min_20 = _Function(function=_ts_min_20, name='ts_min_20', arity=1)
ts_max_5 = _Function(function=_ts_max_5, name='ts_max_5', arity=1)
ts_max_10 = _Function(function=_ts_max_10, name='ts_max_10', arity=1)
ts_max_20 = _Function(function=_ts_max_20, name='ts_max_20', arity=1)

# 45-56
ts_range_5 = _Function(function=_ts_range_5, name='ts_range_5', arity=1)
ts_range_10 = _Function(function=_ts_range_10, name='ts_range_10', arity=1)
ts_range_20 = _Function(function=_ts_range_20, name='ts_range_20', arity=1)
ts_argmin_5 = _Function(function=_ts_argmin_5, name='ts_argmin_5', arity=1)
ts_argmin_10 = _Function(function=_ts_argmin_10, name='ts_argmin_10', arity=1)
ts_argmin_20 = _Function(function=_ts_argmin_20, name='ts_argmin_20', arity=1)
ts_argmax_5 = _Function(function=_ts_argmax_5, name='ts_argmax_5', arity=1)
ts_argmax_10 = _Function(function=_ts_argmax_10, name='ts_argmax_10', arity=1)
ts_argmax_20 = _Function(function=_ts_argmax_20, name='ts_argmax_20', arity=1)
ts_argrange_5 = _Function(function=_ts_argrange_5, name='ts_argrange_5', arity=1)
ts_argrange_10 = _Function(function=_ts_argrange_10, name='ts_argrange_10', arity=1)
ts_argrange_20 = _Function(function=_ts_argrange_20, name='ts_argrange_20', arity=1)

# 57-68
ts_rank_5 = _Function(function=_ts_rank_5, name='ts_rank_5', arity=1)
ts_rank_10 = _Function(function=_ts_rank_10, name='ts_rank_10', arity=1)
ts_rank_20 = _Function(function=_ts_rank_20, name='ts_rank_20', arity=1)
ts_mean_return_5 = _Function(function=_ts_mean_return_5, name='ts_mean_return_5', arity=1)
ts_mean_return_10 = _Function(function=_ts_mean_return_10, name='ts_mean_return_10', arity=1)
ts_mean_return_20 = _Function(function=_ts_mean_return_20, name='ts_mean_return_20', arity=1)
ta_beta_5 = _Function(function=_ta_beta_5, name='ta_beta_5', arity=2)
ta_beta_10 = _Function(function=_ta_beta_10, name='ta_beta_10', arity=2)
ta_beta_20 = _Function(function=_ta_beta_20, name='ta_beta_20', arity=2)
ta_lr_slope_5 = _Function(function=_ta_lr_slope_5, name='ta_lr_slope_5', arity=1)
ta_lr_slope_10 = _Function(function=_ta_lr_slope_10, name='ta_lr_slope_10', arity=1)
ta_lr_slope_20 = _Function(function=_ta_lr_slope_20, name='ta_lr_slope_20', arity=1)

# 69-80
ta_lr_intercept_5 = _Function(function=_ta_lr_intercept_5, name='ta_lr_intercept_5', arity=1)
ta_lr_intercept_10 = _Function(function=_ta_lr_intercept_10, name='ta_lr_intercept_10', arity=1)
ta_lr_intercept_20 = _Function(function=_ta_lr_intercept_20, name='ta_lr_intercept_20', arity=1)
ta_lr_angle_5 = _Function(function=_ta_lr_angle_5, name='ta_lr_angle_5', arity=1)
ta_lr_angle_10 = _Function(function=_ta_lr_angle_10, name='ta_lr_angle_10', arity=1)
ta_lr_angle_20 = _Function(function=_ta_lr_angle_20, name='ta_lr_angle_20', arity=1)
ta_tsf_5 = _Function(function=_ta_tsf_5, name='ta_tsf_5', arity=1)
ta_tsf_10 = _Function(function=_ta_tsf_10, name='ta_tsf_10', arity=1)
ta_tsf_20 = _Function(function=_ta_tsf_20, name='ta_tsf_20', arity=1)
ta_ema_8 = _Function(function=_ta_ema_8, name='ta_ema_8', arity=1)
ta_ema_21 = _Function(function=_ta_ema_21, name='ta_ema_21', arity=1)
ta_ema_55 = _Function(function=_ta_ema_55, name='ta_ema_55', arity=1)

# 81-92
ta_dema_8 = _Function(function=_ta_dema_8, name='ta_dema_8', arity=1)
ta_dema_21 = _Function(function=_ta_dema_21, name='ta_dema_21', arity=1)
ta_dema_55 = _Function(function=_ta_dema_55, name='ta_dema_55', arity=1)
ta_kama_8 = _Function(function=_ta_kama_8, name='ta_dema_8', arity=1)
ta_kama_21 = _Function(function=_ta_kama_21, name='ta_dema_21', arity=1)
ta_kama_55 = _Function(function=_ta_kama_55, name='ta_dema_55', arity=1)
ta_tema_8 = _Function(function=_ta_tema_8, name='ta_tema_8', arity=1)
ta_tema_21 = _Function(function=_ta_tema_21, name='ta_tema_21', arity=1)
ta_tema_55 = _Function(function=_ta_tema_55, name='ta_tema_55', arity=1)
ta_trima_8 = _Function(function=_ta_trima_8, name='ta_trima_8', arity=1)
ta_trima_21 = _Function(function=_ta_trima_21, name='ta_trima_21', arity=1)
ta_trima_55 = _Function(function=_ta_trima_55, name='ta_trima_55', arity=1)

# 93-105
ta_rsi_6 = _Function(function=_ta_rsi_6, name='ta_rsi_6', arity=1)
ta_rsi_12 = _Function(function=_ta_rsi_12, name='ta_rsi_12', arity=1)
ta_rsi_24 = _Function(function=_ta_rsi_24, name='ta_rsi_24', arity=1)
ta_cmo_14 = _Function(function=_ta_cmo_14, name='ta_cmo_14', arity=1)
ta_cmo_25 = _Function(function=_ta_cmo_25, name='ta_cmo_25', arity=1)
ta_mom_12 = _Function(function=_ta_mom_12, name='ta_mom_12', arity=1)
ta_mom_25 = _Function(function=_ta_mom_25, name='ta_mom_25', arity=1)
# ta_roc_14 = _Function(function=_ta_roc_14, name='_ta_roc_14', arity=1)
# ta_roc_25 = _Function(function=_ta_roc_25, name='_ta_roc_25', arity=1)
ta_rocp_14 = _Function(function=_ta_rocp_14, name='ta_rocp_14', arity=1)
ta_rocp_25 = _Function(function=_ta_rocp_25, name='ta_rocp_25', arity=1)
ta_rocr_14 = _Function(function=_ta_rocr_14, name='ta_rocr_14', arity=1)
ta_rocr_25 = _Function(function=_ta_rocr_25, name='ta_rocr_25', arity=1)

# 106-116
ta_trix_8 = _Function(function=_ta_trix_8, name='ta_trix_8', arity=1)
ta_trix_21 = _Function(function=_ta_trix_21, name='ta_trix_21', arity=1)
ta_trix_55 = _Function(function=_ta_trix_55, name='ta_trix_55', arity=1)
ta_adx_14 = _Function(function=_ta_adx_14, name='ta_adx_14', arity=3)
ta_adx_25 = _Function(function=_ta_adx_25, name='ta_adx_25', arity=3)
ta_adxr_14 = _Function(function=_ta_adxr_14, name='ta_adxr_14', arity=3)
ta_adxr_25 = _Function(function=_ta_adxr_25, name='ta_adxr_25', arity=3)
ta_aroonosc_14 = _Function(function=_ta_aroonosc_14, name='ta_aroonosc_14', arity=2)
ta_aroonosc_25 = _Function(function=_ta_aroonosc_25, name='ta_aroonosc_25', arity=2)
ta_cci_14 = _Function(function=_ta_cci_14, name='ta_cci_14', arity=3)
ta_cci_25 = _Function(function=_ta_cci_25, name='ta_cci_25', arity=3)

# 117-
ta_dx_14 = _Function(function=_ta_dx_14, name='ta_dx_14', arity=3)
ta_dx_25 = _Function(function=_ta_dx_25, name='ta_dx_25', arity=3)
ta_mfi_14 = _Function(function=_ta_mfi_14, name='ta_mfi_14', arity=4)
ta_mfi_25 = _Function(function=_ta_mfi_25, name='ta_mfi_25', arity=4)
ta_minus_di_14 = _Function(function=_ta_minus_di_14, name='ta_minus_di_14', arity=3)
ta_minus_di_25 = _Function(function=_ta_minus_di_25, name='ta_minus_di_25', arity=3)
ta_minus_dm_14 = _Function(function=_ta_minus_dm_14, name='ta_minus_dm_14', arity=2)
ta_minus_dm_25 = _Function(function=_ta_minus_dm_25, name='ta_minus_dm_25', arity=2)
ta_willr_14 = _Function(function=_ta_willr_14, name='ta_willr_14', arity=3)
ta_willr_25 = _Function(function=_ta_willr_25, name='ta_willr_25', arity=3)
ta_atr_14 = _Function(function=_ta_atr_14, name='ta_atr_14', arity=3)
ta_atr_25 = _Function(function=_ta_atr_25, name='ta_atr_25', arity=3)
ta_natr_14 = _Function(function=_ta_natr_14, name='ta_natr_14', arity=3)
ta_natr_25 = _Function(function=_ta_natr_25, name='ta_natr_25', arity=3)



_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 # 下面对应的是增加部分
                 'tanh': tanh1,
                 'elu': elu1,
                 'TA_HT_TRENDLINE': ta_ht_trendline,
                 'TA_HT_DCPERIOD': ta_ht_dcperiod,
                 'TA_HT_DCPHASE': ta_ht_dcphase,
                 'TA_SAR': ta_sar,
                 'TA_BOP': ta_bop,
                 'TA_AD': ta_ad,
                 'TA_OBV': ta_obv,
                 'TA_TRANGE': ta_trange,
                 # 5月23日加入
                 # 1-10：
                 'TS_COV_20' : ts_cov_20,
                 'TS_COV_40' : ts_cov_40,
                 'TS_CORR_20' : ts_corr_20,
                 'TS_CORR_40' : ts_corr_40,
                 'TS_DAY_MIN_10' : ts_day_min_10,
                 'TS_DAY_MIN_20' : ts_day_min_20,
                 'TS_DAY_MIN_40' : ts_day_min_40,
                 'TS_DAY_MAX_10' : ts_day_max_10,
                 'TS_DAY_MAX_20' : ts_day_max_20,
                 'TS_DAY_MAX_40' : ts_day_max_40,
                 # 11-19:
                 'ts_sma_8' : ts_sma_8,
                 'ts_sma_21' : ts_sma_21,
                 'ts_sma_55' : ts_sma_55,
                 'ts_wma_8' : ts_wma_8,
                 'ts_wma_21' : ts_wma_21,
                 'ts_wma_55' : ts_wma_55,
                 'ts_lag_3' : ts_lag_3,
                 'ts_lag_8' : ts_lag_8,
                 'ts_lag_17' : ts_lag_17,
                 # 20-32
                 'ts_delta_3' : ts_delta_3,
                 'ts_delta_8' : ts_delta_8,
                 'ts_delta_17' : ts_delta_17,
                 'ts_sum_3' : ts_sum_3,
                 'ts_sum_8' : ts_sum_8,
                 'ts_sum_17' : ts_sum_17,
                 'ts_prod_3' : ts_prod_3,
                 'ts_prod_8' : ts_prod_8,
                 'ts_prod_17' : ts_prod_17,
                 'ts_std_10' : ts_std_10,
                 'ts_std_20' : ts_std_20,
                 'ts_std_40' : ts_std_40,
                 # 33-44 
                 'ts_skew_10' : ts_skew_10,
                 'ts_skew_20' : ts_skew_20,
                 'ts_skew_40' : ts_skew_40,
                 'ts_kurt_10' : ts_kurt_10,
                 'ts_kurt_20' : ts_kurt_20,
                 'ts_kurt_40' : ts_kurt_40,
                 'ts_min_5' : ts_min_5,
                 'ts_min_10' : ts_min_10,
                 'ts_min_20' : ts_min_20,
                 'ts_max_5' : ts_max_5,
                 'ts_max_10' : ts_max_10,
                 'ts_max_20' : ts_max_20,
                 # 45-56
                 'ts_range_5' : ts_range_5,
                 'ts_range_10' : ts_range_10,
                 'ts_range_20' : ts_range_20,
                 'ts_argmin_5' : ts_argmin_5,
                 'ts_argmin_10' : ts_argmin_10,
                 'ts_argmin_20' : ts_argmin_20,
                 'ts_argmax_5' : ts_argmax_5,
                 'ts_argmax_10' : ts_argmax_10,
                 'ts_argmax_20' : ts_argmax_20,
                 'ts_argrange_5' : ts_argrange_5,
                 'ts_argrange_10' : ts_argrange_10,
                 'ts_argrange_20' : ts_argrange_20,
                 # 57-68
                 'ts_rank_5' : ts_rank_5,
                 'ts_rank_10' : ts_rank_10,
                 'ts_rank_20' : ts_rank_20,
                 'ts_mean_return_5' : ts_mean_return_5,
                 'ts_mean_return_10' : ts_mean_return_10,
                 'ts_mean_return_20' : ts_mean_return_20,
                 'ta_beta_5' : ta_beta_5,
                 'ta_beta_10' : ta_beta_10,
                 'ta_beta_20' : ta_beta_20,
                 'ta_lr_slope_5' : ta_lr_slope_5,
                 'ta_lr_slope_10' : ta_lr_slope_10,
                 'ta_lr_slope_20' : ta_lr_slope_20,
                 # 69-80
                 'ta_lr_intercept_5' : ta_lr_intercept_5,
                 'ta_lr_intercept_10' : ta_lr_intercept_10,
                 'ta_lr_intercept_20' : ta_lr_intercept_20,
                 'ta_lr_angle_5' : ta_lr_angle_5,
                 'ta_lr_angle_10' : ta_lr_angle_10,
                 'ta_lr_angle_20' : ta_lr_angle_20,
                 'ta_tsf_5' : ta_tsf_5,
                 'ta_tsf_10' : ta_tsf_10,
                 'ta_tsf_20' : ta_tsf_20,
                 'ta_ema_8' : ta_ema_8,
                 'ta_ema_21' : ta_ema_21,
                 'ta_ema_55' : ta_ema_55,
                 # 81-92
                 'ta_dema_8': ta_dema_8,
                 'ta_dema_21': ta_dema_21,
                 'ta_dema_55': ta_dema_55,
                 'ta_kama_8' : ta_kama_8,
                 'ta_kama_21' : ta_kama_21,
                 'ta_kama_55' : ta_kama_55,
                 'ta_tema_8' : ta_tema_8,
                 'ta_tema_21' : ta_tema_21,
                 'ta_tema_55' : ta_tema_55,
                 'ta_trima_8' : ta_trima_8,
                 'ta_trima_21' : ta_trima_21,
                 'ta_trima_55' : ta_trima_55,
                 # 93-105
                 'ta_rsi_6' : ta_rsi_6,
                 'ta_rsi_12' : ta_rsi_12,
                 'ta_rsi_24' : ta_rsi_24,
                 'ta_cmo_14' : ta_cmo_14,
                 'ta_cmo_25' : ta_cmo_25,
                 'ta_mom_12' : ta_mom_12,
                 'ta_mom_25' : ta_mom_25,
                #  'ta_roc_14' : ta_roc_14,
                #  'ta_roc_25' : ta_roc_25,
                 'ta_rocp_14' : ta_rocp_14,
                 'ta_rocp_25' : ta_rocp_25,
                 'ta_rocr_14' : ta_rocr_14,
                 'ta_rocr_25' : ta_rocr_25,
                 # 106-116
                 'ta_trix_8' : ta_trix_8,
                 'ta_trix_21' : ta_trix_21,
                 'ta_trix_55' : ta_trix_55,
                 'ta_adx_14' : ta_adx_14,
                 'ta_adx_25' : ta_adx_25,
                 'ta_adxr_14' : ta_adxr_14,
                 'ta_adxr_25' : ta_adxr_25,
                 'ta_aroonosc_14' : ta_aroonosc_14,
                 'ta_aroonosc_25' : ta_aroonosc_25,
                 'ta_cci_14' : ta_cci_14,
                 'ta_cci_25' : ta_cci_25,
                 # 117-120
                 'ta_dx_14' : ta_dx_14,
                 'ta_dx_25' : ta_dx_25,
                 'ta_mfi_14' : ta_mfi_14,
                 'ta_mfi_25' : ta_mfi_25,
                 'ta_minus_di_14' : ta_minus_di_14,
                 'ta_minus_di_25' : ta_minus_di_25,
                 'ta_minus_dm_14' : ta_minus_dm_14,
                 'ta_minus_dm_25' : ta_minus_dm_25,
                 'ta_willr_14' : ta_willr_14,
                 'ta_willr_25' : ta_willr_25,
                 'ta_atr_14' : ta_atr_14,
                 'ta_atr_25' : ta_atr_25,
                 'ta_natr_14' : ta_natr_14,
                 'ta_natr_25' : ta_natr_25,
                 }
