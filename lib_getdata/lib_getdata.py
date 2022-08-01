import datetime
import logging
import os

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection as sklms
import scipy.stats
import scipy.signal
import tqdm
from typing import Literal
import statsmodels.tsa.stattools
import joblib

def get_bars(currency: str = "EURUSD", timeframe=mt5.TIMEFRAME_M15, shift_from_actual_bar: int = 1, bars: int = 10000,
             start_date: 'int|datetime.datetime|None' = None, end_date: 'int|datetime.datetime|None' = None):
    '''
    get bars data from mt5 client by position relative to actual bar

    Params:
    -------
    *currency: str = the symbol to get it must to be exactly as in mt5
    *timeframe: mt5.timeframe = the timeframe to get the datas, need to be a mt5.timeframe
    *shift_from_actual_bar: int = the number of bars to get since the actual ('not closed') bar ('actual bar is 0')
    *bars: int = the number of bars to get
    *start_date: optional intTimestampSeconds | datetime | None = if setted shift_from_actual_bar and bars will be ignored and will get data using the date
    *end_date: optional intTimestampSeconds | datetime | None = as the previous
    '''
    if not mt5.initialize():
        logging.error('fail to mt5.initialize()')
    if not mt5.symbol_select(currency, True):
        logging.error(f"was not possible to add the currency: {currency} to the market watch")

    if start_date is not None or end_date is not None:
        logging.info("start_date or end_date is not None so using the function mt5.copy_rates_range(...)")
        to_timestamp_s = lambda t: int((t-datetime.datetime(1970, 1, 1)).total_seconds())
        if isinstance(start_date, datetime.datetime):
            logging.info('start_date is a datetime.datetime so converting it to timestamp seconds...')
            start_date = to_timestamp_s(start_date)
        if isinstance(end_date, datetime.datetime):
            logging.info('end_date is a datetime.datetime so converting it to timestamp seconds...')
            end_date = to_timestamp_s(end_date)
        for _ in range(3):
            data = mt5.copy_rates_range(currency, timeframe, start_date, end_date)
    elif start_date is None and end_date is None:
        for _ in range(3):
            data = mt5.copy_rates_from_pos(currency, timeframe, shift_from_actual_bar, bars)

    if shift_from_actual_bar+bars > 200000 and timeframe!=mt5.TIMEFRAME_M1:
        logging.warning(
                "You are getting a big number of data, check if the data from the past data is correct the most common is the spread dont be correct with repeated spread or large value for some brokers")

    if data is None:
        raise Exception(
                "The MT5 client is not configured to download this amount of bars to solve it do: in mt5 > tools > options > charts > max bars in chart. <- increase this value or set it to unlimited.")
    elif len(data)!=bars and (start_date or end_date) is None:
        raise Exception(
                "The number of bars returned by the client is lower than the requested (requested: "+str(bars)+" | "+"returned: "+str(
                        len(data))+") this can be happened because the broker don't have data enough this can be solved changing the broker or reducing the number of bars requested.")
    return data

def evalmodel(model, datax, datay, kfold_num: int = 10, verbose: int = False, metric: str = 'f1_weighted', n_jobs=-1):
    '''
    function to do cross validation and return the mean of sklearn models
    '''
    crossval = sklms.cross_validate(model, datax, datay, cv=kfold_num, verbose=verbose, n_jobs=n_jobs, scoring=metric)
    model_value = 0
    for i in range(0, list(crossval['test_score']).__len__(), 1):
        model_value += crossval['test_score'][i]
    result = (model_value/list(crossval['test_score']).__len__())
    return result

def dataframe_transform(data, columnstoexclude: 'list|None' = None,
                        renamecolumns: 'dict|None' = {'time':        'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                                                      'tick_volume': 'Volume', 'spread': 'Spread'}):
    '''
    transform mt5 bars response (turple) in a dataframe
    params:
    -------
    *data: the data to transform
    *columnstoexclude: list of columns to exclude the name needs to be BEFORE RENAMING
    *renamecolumns: rename the columns
    '''
    data = data
    data = pd.DataFrame(data)
    # com spread #0.5
    # sem spread #0.49
    if columnstoexclude is not None:
        data = data.drop(columns=list(columnstoexclude), errors='ignore')
    if renamecolumns is not None:
        data = data.rename(columns=renamecolumns)
    return data

def split_into_samples(df, number_of_samples='auto', bars_per_sample=64, auto_cut_data: bool = True):
    '''
    split dataframe into parts
    '''
    df = df
    maximum_number_of_samples = int(df.index.size//bars_per_sample)
    if number_of_samples=='auto':
        number_of_samples = maximum_number_of_samples
    elif number_of_samples > maximum_number_of_samples:
        raise Exception('the maximum number of samples is: '+str(maximum_number_of_samples))
    index_to_start = df.index.size%bars_per_sample
    if auto_cut_data==True:
        df = df.iloc[index_to_start:]
    samples = []
    for i in range(number_of_samples):
        samples.append(df.iloc[bars_per_sample*i:bars_per_sample*(i+1)])

    return samples

def label_simple(sampled_datax, close_column: str, distance_to_predict=32):
    '''
    simple labeling technique for classification 

    Return:
    -------
        return [datax,datay]
    '''
    close = close_column
    datax = sampled_datax
    result = []
    for i in range(0, datax.__len__()-1, 1):
        if datax[i+1][close].iloc[distance_to_predict-1] > datax[i][close].iloc[-1]:
            result.append([1])
        else:
            result.append([-1])
    return [datax[:-1], result]

def label_direction_number_bars(sampled_datax, distance_to_predict=32):
    '''
    Return:
    -------
        [datax,datay]
    '''
    datax = sampled_datax
    analisis = []
    analisis_to_result = 0
    result = []
    for i in range(0, datax.__len__()-1, 1):
        analisis.append(datax[i+1]['Close'].iloc[:distance_to_predict-1])
        for h in range(0, analisis[-1].index.size-1, 1):
            if analisis[-1].iloc[h+1] > analisis[-1].iloc[h]:
                analisis_to_result += 1
            else:
                analisis_to_result -= 1
        result.append([analisis_to_result])
        analisis_to_result = 0
    return [datax[:-1], result]

def label_interaction_simple_direction_number_bars(datay_simple, datay_direction_number_bars, threshold=1):
    if list(datay_simple).__len__()!=list(datay_direction_number_bars).__len__():
        raise Exception("there a difference between number of indexes among datay_simple and datay_direction_number_bars")
    datay_simple = datay_simple
    datay_advanced = datay_direction_number_bars
    result = []
    threshold = threshold
    for i in range(0, datay_simple.__len__() if datay_simple.__len__()==datay_advanced.__len__() else datay_advanced, 1):
        if datay_simple[i]==[1] and datay_advanced[i] >= [threshold]:
            result.append([1])
        elif datay_simple[i]==[-1] and datay_advanced[i] <= [-threshold]:
            result.append([-1])
        elif datay_simple[i]==[1] and datay_advanced[i] < [0]:
            result.append([99])
        elif datay_simple[i]==[-1] and datay_advanced[i] > [0]:
            result.append([99])
        else:
            result.append([0])
    return result

def list_diff(a, b):
    r = []

    for i in a:
        if i not in b:
            r.append(i)
    return r

def list_merge(oldlist: list, newlist: list):
    nlist = oldlist
    for name in newlist:
        if not name in oldlist:
            nlist.append(name)
    return nlist

def df_merge(old_df: pd.DataFrame, new_df: pd.DataFrame, update_old: Literal['auto', 'force', False] = 'auto',
             join_to_old_df: Literal['auto', 'always', False] = 'auto', sort_index: bool = True, rsuffix='_new_df') -> pd.DataFrame:
    '''
    merge 2 dataframes and the COMMON COLUMNS WILL BE UPDATED whether both dataframes are equals (there same columns)
    whether dataframes are differents REPEATED COLUMNS WILL BE ADDED A SUFFIX OR PREFIX
    '''
    old_df = pd.DataFrame(old_df)
    original_old_df_columns = old_df.columns

    if join_to_old_df=='auto':
        if not all(column in old_df.columns for column in new_df.columns):
            old_df = old_df.join(other=new_df, rsuffix=rsuffix)
    elif join_to_old_df=='always':
        old_df = old_df.join(other=new_df, rsuffix=rsuffix)

    if all(column in original_old_df_columns for column in old_df.columns) and len(original_old_df_columns)==len(
            old_df.columns) and update_old=='auto':
        old_df.update(new_df)
    elif update_old=='force':
        old_df.update(new_df)

    for index in new_df.index:
        if not index in old_df.index:
            old_df = old_df.append(new_df.loc[index])
    if sort_index is True:
        old_df = old_df.sort_index()
    return old_df

def drop_all_repeated_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    drop columns that have all values repeated, incuding columns with all nans
    '''
    nunique = df.nunique()
    cols_to_drop = nunique[nunique <= 1].index
    df = df.drop(cols_to_drop, axis=1)
    return df

def abs_return(df: pd.DataFrame):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.apply(lambda x: abs(x.diff()))

def df_sklearn_wrapper(sklfunc, df: pd.DataFrame):
    '''
    a function to apply sklearn.fit_transform in a dataframe and return a dataframe
    '''
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return pd.DataFrame(data=sklfunc.fit_transform(X=df), columns=df.columns, index=df.index)

def log_return(df: pd.DataFrame):
    '''
    transform data in log return
    '''
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df_sklearn_wrapper(sklfunc=sklearn.preprocessing.MinMaxScaler(feature_range=(1, 100)), df=df)
    return df.apply(func=lambda x: np.log(x/x.shift(1)))

def de_stationary(df: pd.DataFrame) -> pd.DataFrame:
    '''
    cumulative sum of the values to visualization, the visualization is near before log return
    '''
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.apply(func=lambda x: np.cumsum(x))

def class_to_dict(clas):
    tr = {key: value for key, value in clas.__dict__.items() if not key.startswith('__') and not callable(key)}
    return tr

class numpy:

    def rolling_apply(data: np.ndarray, window: int, func, progress: bool = True):
        '''
        does a rolling on a numpy array, as on 'pandas.series.rolling'
        '''
        data = np.copy(data)
        _selector = np.copy(data)
        window = window-1
        for i in tqdm.tqdm(range(window, len(data)), desc='rolling_apply', disable=1-progress,leave=False):
            data[i] = func(_selector[i-window:i])
        data[:window] = None
        return data

    def expanding_apply(data: np.ndarray, func, min_periods: int = 0, progress: bool = True):
        data = np.copy(data)
        _selector = np.copy(data)
        for i in tqdm.tqdm(range(min_periods, len(data)), desc='expanding_apply', disable=1-progress,leave=False):
            data[i] = func(_selector[:i])
        data[:min_periods] = None
        return data

    def shift(data:np.ndarray,shift:int):
        data=np.copy(data)
        data=np.roll(data,shift)
        data[:shift]=None
        return data

    def pct_change(data: np.ndarray):
        func = lambda x: (x[-2]-x[-1])/x[-1]
        return numpy.expanding_apply(data, func=func, min_periods=2)

    def hhistogram(data, bins: int):
        """
        'h'histogram means human histogram, cause numpy return a value i dont know what is that on x axis
        so this correct this, y,x=np.histogram(...); x=x[:-1]
        :return : (x,y)
        """
        y, x = np.histogram(a=data, bins=bins)
        x = x[:-1]
        assert len(x)==len(y)
        return (x, y)

    class feature_engineering:
        class feature_transformation:
            def logrp1(data: np.ndarray):
                data = np.copy(data)
                diff_data = np.diff(data)
                ndata = np.apply_along_axis(lambda x: np.log(1+x+abs(diff_data.min())), axis=0, arr=diff_data)
                return ndata

            def dVaR(data: np.ndarray, VaR_pct: float):
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

            def smooth(data: np.ndarray, window: int, polyorder: int = 2):
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

            def rolling_std(data: np.ndarray, window: int):
                '''
                rolling standard deviation
                :param window:
                :return: np.ndarray
                '''

                return numpy.rolling_apply(data=data, window=window, func=lambda x: np.std(x))

            def rolling_var(data: np.ndarray, window: int):
                '''
                rolling variance
                :param window:
                :return: np.ndarray
                '''

                return numpy.rolling_apply(data, window=window, func=lambda x: np.var(x))

            def rolling_min(data: np.ndarray, window: int):
                return numpy.rolling_apply(data, window=window, func=lambda x: np.min(x))

            def rolling_max(data: np.ndarray, window: int):
                return numpy.rolling_apply(data, window=window, func=lambda x: np.max(x))

            def rolling_kurtosis(data: np.ndarray, window: int):
                return numpy.rolling_apply(data, window=window, func=lambda x: scipy.stats.kurtosis(x))

            def rolling_skewness(data: np.ndarray, window: int):
                return numpy.rolling_apply(data, window=window, func=lambda x: scipy.stats.skew(x))

            def ex_tanh_estimator(data: np.ndarray):
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

        class outlier_detection:
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

        class volatility:
            def yangZhangEstimator(_open: np.ndarray, _high: np.ndarray, _low: np.ndarray, _close: np.ndarray):
                '''
                :param ohlc_df: the df param needs to be a dataframe with the mandatory column: "Open","High","Low","Close". is case-sensitive
                :param window: is the value for the pandas.rolling(window=thisparam)
                :param trading_periods: a especific parameter for yang-zhang estimator
                :return:
                '''
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
                    numpy.expanding_apply(data,
                                             func=lambda x: yangZhangEstimator_formula(x.transpose()[0], x.transpose()[1], x.transpose()[2],
                                                                                       x.transpose()[3]), min_periods=3).transpose()[0]

class pandas:
    class feature_engineering:
        class feature_selection:
            def grangercausalitytest(df, endog_column, maxlag) -> pd.DataFrame:
                df=df.copy()
                for column in df.columns:
                    assert df[column].nunique()>1,f"the column '{column}' is constant, granger raises an error whether value is constant"
                df_result = pd.DataFrame()
                for column in tqdm.tqdm(df.columns,desc='granger causality cross columns'):
                    test = statsmodels.tsa.stattools.grangercausalitytests(x=df[[endog_column, column]], maxlag=maxlag, verbose=False)
                    results = []
                    for key, value in test.items():
                        pvalue = value[0]['ssr_ftest'][1]
                        results.append(pvalue)
                    df_result[column] = pd.Series(results, index=list(range(1, maxlag+1)))
                df_result.index.name='lag'
                return df_result

        class scaler:
            class min_max:
                def __init__(self):
                    pass

                def fit(self,df):
                    df=df.copy()
                    self.model=pd.DataFrame(index=["min","max"])
                    for column in df:
                        min=df[column].min()
                        max=df[column].max()
                        self.model[column]=[min,max]
                        self.model[column].loc['min']=min
                        self.model[column].loc['max']=max
                    return self

                def load(self,model):
                    '''
                    model is a dataframe containig the min and max (as rows) for each column
                    '''
                    self.model=model
                    return self

                def transform(self,df):
                    df=df.copy()
                    for column in self.model.columns:
                        if not column in df.columns:
                            raise Exception(f"the column: '{column}', was present in the data when it was fitted but not is presend to transoform")

                    for column in self.model.columns:
                        min=self.model[column].loc['min']
                        max=self.model[column].loc['max']
                        df[column]=df[column].apply(lambda x: (x-min)/(max-min))
                    return df

    def timeseries_from_pandas(df,input_size,output_size,auto_resize=True,n_jobs=-1,progress=True):
        """
        this function split the dataframe into a x and y to fit machine learning like tensorflow models
        remember the data will be still list of dataframe is needed to convert them to numpy
        """
        df=df.copy()
        sequence_size = input_size+output_size
        if df.index.size%sequence_size!=0:
            if auto_resize is False:
                raise ValueError("the data is not divisible by {sequence_size} the data length needs to be divisible by that value"
                                 "you can set parameter 'auto_resize=True' to the algo remove the pasts data from the dataframe")
            else:
                df = df.iloc[df.index.size%sequence_size:]
        iters=range(df.index.size-sequence_size+1)
        x = []
        y = []
        assert df.index.size%sequence_size==0, f"the size of data is not divisible by the sequence_length (input_size+output_size): {sequence_size}"
        def process(i):
            visible = df.iloc[i:sequence_size+i]
            x.append(visible.iloc[:input_size])
            y.append(visible.iloc[input_size:])
        joblib.Parallel(n_jobs=n_jobs,backend='threading',verbose=progress)(joblib.delayed(process)(i) for i in iters)
        assert x[0].index[0]==df.index[0]
        assert y[-1].index[-1]==df.index[-1], "the last data of the y data is not the last data of the data"
        return (x,y)