# here contains methods to with with numpy array
# THE NUMPY ARRAY INPUT NEEDS TO BE COPIED FIRS use 'np.copy()' to copy the input
# cause using the default way can cause bugs
import numpy as np
import tqdm

def numpy_concat(old: list, new: list):
    old = np.copy(old)
    new = np.copy(new)
    nlist = old
    for name in new:
        if not name in old:
            nlist = np.append(nlist, name)
    return nlist

def numpy_rollingApply(data: np.ndarray, window: int, func, progress: bool = True):
    '''
    does a rolling on a numpy array, as on 'pandas.series.rolling'
    '''
    data = np.copy(data)
    _selector = np.copy(data)
    window = window-1
    for i in tqdm.tqdm(range(window, len(data)), desc='rolling_apply', disable=1-progress, leave=False):
        data[i] = func(_selector[i-window:i])
    data[:window] = None
    return data

def numpy_expandingApply(data: np.ndarray, func, min_periods: int = 0, progress: bool = True):
    data = np.copy(data)
    _selector = np.copy(data)
    for i in tqdm.tqdm(range(min_periods, len(data)), desc='expanding_apply', disable=1-progress, leave=False):
        data[i] = func(_selector[:i])
    data[:min_periods] = None
    return data

def numpy_shift(data: np.ndarray, shift: int):
    data = np.copy(data)
    data = np.roll(data, shift)
    data[:shift] = None
    return data

def numpy_hHistogram(data, bins: int):
    """
    'h'histogram means human histogram, cause numpy return a value i dont know what is that on x axis
    so this correct this, y,x=np.histogram(...); x=x[:-1]
    :return : (x,y)
    """
    data = np.copy(data)
    y, x = np.histogram(a=data, bins=bins)
    x = x[:-1]
    assert len(x)==len(y)
    return (x, y)
