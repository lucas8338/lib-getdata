# this file contains transformations for numpy array
# transformations are math calculations to transform a data into another
# without any logical correlations with the another, this is the difference
# from normalization / standarlization
import os
import pickle

import numpy
import numpy as np
import pyod
import scipy
from .np import numpy_expandingApply

def transf_pctChange(data: np.ndarray):
    func = lambda x: (x[-2]-x[-1])/x[-1]
    return numpy.expanding_apply(data, func=func, min_periods=2)

def transf_logrp1(data: np.ndarray):
    data = np.copy(data)
    diff_data = np.diff(data)
    ndata = np.apply_along_axis(lambda x: np.log(1+x+abs(diff_data.min())), axis=0, arr=diff_data)
    return ndata

def transf_dVaR(data: np.ndarray, VaR_pct: float):
    data = np.copy(data)

    def ocurrence_of_pct(x, pct):
        # remove nans from array
        x = x[np.logical_not(np.isnan(x))]
        # remove zeros from array
        x = x[x!=0]
        # transform array in all positive
        x = np.abs(x)
        # variable to transform the values in percentage
        weights = np.ones(len(x))/len(x)
        # run the histogram
        y, x = np.histogram(x, weights=weights, bins=int(100/(pct*100))*2)
        # remove the last value of x because it is just a value exclusive for numpy is how numpy works
        x = x[:len(x)-1]
        # return the hightest value bigger than VaR_pct
        return np.max(x[np.where(y >= pct)])

    data = numpy.pct_change(data)

    ndata = numpy.expanding_apply(data, func=lambda x: ocurrence_of_pct(x, VaR_pct), min_periods=3)
    return ndata

def transf_smooth(data: np.ndarray, window: int, polyorder: int = 2):
    '''
    smoothe the timeseries using savgol filter a lagless filter
    it can be used to extract the trend or smoothing
    :param window: the window of the data it is similar to window in the moving average
    :param polyorder: it is the 'how much' times the smoothed line can 'bounce' (bounce is like a frequency), for example:
    if you have a windows=32 and polyorder=2 this mean inside a period of 32 data the line can bounce 2 times is like this.
    for a trend extraction 2 geraly is the best option that i have tested.
    :return:
    '''

    result = scipy.signal.savgol_filter(x=data, window_length=window, polyorder=polyorder)
    return result

def transf_rollingStd(data: np.ndarray, window: int):
    '''
    rolling standard deviation
    :param window:
    :return: np.ndarray
    '''

    return numpy.rolling_apply(data=data, window=window, func=lambda x: np.std(x))

def transf_rollingVar(data: np.ndarray, window: int):
    '''
    rolling variance
    :param window:
    :return: np.ndarray
    '''

    return numpy.rolling_apply(data, window=window, func=lambda x: np.var(x))

def transf_rollingMin(data: np.ndarray, window: int):
    return numpy.rolling_apply(data, window=window, func=lambda x: np.min(x))

def transf_rollingMax(data: np.ndarray, window: int):
    return numpy.rolling_apply(data, window=window, func=lambda x: np.max(x))

def transf_rollingKurtosis(data: np.ndarray, window: int):
    return numpy.rolling_apply(data, window=window, func=lambda x: scipy.stats.kurtosis(x))

def transf_rollingSkewness(data: np.ndarray, window: int):
    return numpy.rolling_apply(data, window=window, func=lambda x: scipy.stats.skew(x))

def transf_exTanhEstimator(data: np.ndarray):
    '''
    Normalization thecnique by tanh estimator
    reference:
        Impact of Data Normalization on Deep Neural Network for Time Series Forecasting, Samit Bhanja and Abhishek Das*, Member, IEEE
        formula from: https://github.com/calceML/PHM/blob/master/Chapter5/Example2-Normalization/tanhestimator.m
    :return:
    '''

    def formula(data, mean, std):
        '''
        :param data: the value to be normalized
        :param mean: the mean of all values
        :param std: the std (standard deviation) of all values
        :return: calculated by tanh estimator formula
        '''
        return 0.5*(np.tanh((0.01*(data-mean))/std)+1)

    result = numpy.expanding_apply(data, func=lambda x: formula(x[-1], mean=np.mean(x), std=np.std(x)), min_periods=1)
    return result

class transf_outlierDetectionModelEcod:
    def outlier_detect(data: 'np.ndarray', fitted_file: str):
        path = fitted_file
        X = data
        if not os.path.isfile(path):
            model = pyod.models.ecod.ECOD()
            fmodel = model.fit(X=X)
            with open(file=fitted_file, mode='wb') as file:
                pickle.dump(obj=fmodel, file=file)
        elif os.path.isfile(path):
            with open(file=fitted_file, mode='rb') as file:
                fmodel = pickle.load(file=file)
        predict_prob = fmodel.predict_proba(X=X)
        prob_be_outlier = []
        for value in predict_prob:
            prob_be_outlier.append(value[1])
        return prob_be_outlier

def transf_yangZhangEstimator(_open: np.ndarray, _high: np.ndarray, _low: np.ndarray, _close: np.ndarray):
    assert len(_open)==len(_high)==len(_low)==len(_close)

    def yangZhangEstimator_formula(_open, _high, _low, _close):
        n = len(_close)
        pt1 = 1/n*(np.log(_open[-1]/_close[-2]))**2
        pt2 = 1/n*1/2*(np.log(_high[-1]/_low[-1]))**2
        pt3 = 1/n*(2*np.log(2)-1)*(np.log(_close[-1]/_open[-1]))**2
        result = np.sqrt(pt1+pt2-pt3)
        return result

    data = np.array([_open, _high, _low, _close])
    data = np.reshape(data, (-1, 4))
    return \
        numpy_expandingApply(data,
                             func=lambda x: yangZhangEstimator_formula(x.transpose()[0], x.transpose()[1], x.transpose()[2],
                                                                       x.transpose()[3]), min_periods=3).transpose()[0]
