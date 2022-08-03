# here contains functions and classes to work with pandas
# DO A DATAFRAME.COPY 'dataframe.copy()' to the argument to avoid bugs
import pandas as pd
import statsmodels.tsa.stattools
import tqdm

def pandas_grangerCausalityTest(df, endog_column, maxlag) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        assert df[column].nunique() > 1, f"the column '{column}' is constant, granger raises an error whether value is constant"
    df_result = pd.DataFrame()
    for column in tqdm.tqdm(df.columns, desc='granger causality cross columns'):
        test = statsmodels.tsa.stattools.grangercausalitytests(x=df[[endog_column, column]], maxlag=maxlag, verbose=False)
        results = []
        for key, value in test.items():
            pvalue = value[0]['ssr_ftest'][1]
            results.append(pvalue)
        df_result[column] = pd.Series(results, index=list(range(1, maxlag+1)))
    df_result.index.name = 'lag'
    return df_result

class pandas_minMaxScaler:
    def __init__(self):
        pass

    def fit(self, df):
        df = df.copy()
        self.model = pd.DataFrame(index=["min", "max"])
        for column in df.columns:
            min = df[column].min()
            max = df[column].max()
            self.model[column] = [min, max]
            self.model[column].loc['min'] = min
            self.model[column].loc['max'] = max
        return self

    def save(self):
        '''
        return the self.model dataframe to save
        this this file contains the min and max for each column
        '''
        return self.model

    def load(self, model):
        '''
        model is a dataframe containig the min and max (as rows) for each column
        '''
        self.model = model
        return self

    def transform(self, df):
        df = df.copy()
        for column in self.model.columns:
            if not column in df.columns:
                raise Exception(f"the column: '{column}', was present in the data when it was fitted but not is presend to transoform")

        for column in self.model.columns:
            min = self.model[column].loc['min']
            max = self.model[column].loc['max']
            df[column] = df[column].apply(lambda x: (x-min)/(max-min))
        return df

def pandas_timeseriesXYSplit(df, input_size, output_size, auto_resize=True, progress=True):
    """
    this function split the dataframe into a x and y to fit machine learning like tensorflow models
    remember the data x and y will be a list of dataframe is needed to transform them to fit
    """
    df = df.copy()
    sequence_size = input_size+output_size
    if df.index.size%sequence_size!=0:
        if auto_resize is False:
            raise ValueError("the data is not divisible by {sequence_size} the data length needs to be divisible by that value"
                             "you can set parameter 'auto_resize=True' to the algo remove the pasts data from the dataframe")
        else:
            df = df.iloc[df.index.size%sequence_size:]
    iters = range(df.index.size-sequence_size+1)
    x = []
    y = []
    assert df.index.size%sequence_size==0, f"the size of data is not divisible by the sequence_length (input_size+output_size): {sequence_size}"

    def process(i):
        visible = df.iloc[i:sequence_size+i]
        x.append(visible.iloc[:input_size])
        y.append(visible.iloc[input_size:])

    for i in tqdm.tqdm(iters, desc='timeseries_from_pandas', disable=1-progress):
        process(i)
    assert x[0].index[0]==df.index[0]
    assert y[-1].index[-1]==df.index[-1], "the last data of the y data is not the last data of the data"
    return (x, y)
