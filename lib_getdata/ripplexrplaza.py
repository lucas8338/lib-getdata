import concurrent.futures
import datetime
import json
import os
import time

import numpy as np
import pandas as pd
import requests
import requests_futures.sessions
from tqdm import tqdm

engine = 'fastparquet'

class ripplexrplaza():
    def __init__(self, file, date_start, jump_distance, n_requests_by_step, checkpoint_interval: datetime.timedelta, request_interval=0.3,
                 show_progress: bool = True):
        self.file = file
        self.jump_distance = jump_distance
        self.n_requests_by_step = n_requests_by_step
        self.checkpoint_interval = checkpoint_interval
        self.date_start = date_start
        self.request_interval = request_interval
        self.show_progress = show_progress

        self.date_now = datetime.datetime.utcnow()
        # bellow variables must be changed to work with others api's
        # self.time_column is the name of column containing the timestamp
        self.time_column = 'close_time'
        # self.block_column is the name of column containing the number of block
        self.block_column = 'ledger_index'

    def _get_already_data(self):
        if os.path.isfile(self.file):
            self.df = pd.read_parquet(path=self.file, engine=engine)
        elif not os.path.isfile(self.file):
            self.df = None
        return self

    def _check_marks(self):
        self.recently_block = self.df[self.block_column].max() if self.df is not None else np.nan
        self.recently_time = self.df.index.max().timestamp() if self.df is not None else None
        self.past_block = self.df[self.block_column].min() if self.df is not None else None
        self.past_time = self.df.index.min().timestamp() if self.df is not None else None
        return self

    def _check_already_ok(self):
        self.recently_time = np.nan if not self.recently_time else self.recently_time
        self.past_time = np.nan if not self.past_time else self.past_time
        # bellow the variable 'self.is_forward_ok' must check with a function returning
        # the latest number of block
        self.is_forward_ok = True if self.recently_block >= api_wrapper().get_latest().latest else False
        self.is_backward_ok = True if self.past_time <= self.date_start.timestamp() else False
        return self

    def _backward(self):
        block = None if self.df is None else self.past_block
        date_start = datetime.datetime.utcfromtimestamp(self.past_time) if self.df is not None else self.date_now
        date_start = date_start-self.checkpoint_interval

        data = api_wrapper().get(block=block,
                                 date_start=date_start, jump_distance=self.jump_distance,
                                 n_requests_by_step=self.n_requests_by_step, show_progress=self.show_progress,
                                 request_interval=self.request_interval)
        data=data.set_index(keys=self.time_column)
        data.index=pd.to_datetime(data.index,unit='s')
        # try to transform all columns into numeric
        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='ignore')
        data = self.df.append(other=data) if self.df is not None else data
        # line to fix maxsize error
        data = data.apply(lambda x: x.astype(np.float64,errors='ignore'))
        data = data[~data.index.duplicated()]
        data.to_parquet(path=self.file, engine=engine)
        return self

    def _forward(self):
        data = api_wrapper().get(block=None, date_start=datetime.datetime.utcfromtimestamp(self.recently_time),
                                 jump_distance=self.jump_distance,
                                 n_requests_by_step=self.n_requests_by_step, request_interval=self.request_interval,
                                 show_progress=self.show_progress)
        data = data.set_index(keys=self.time_column)
        data.index = pd.to_datetime(data.index, unit='s')
        data = self.df.append(other=data)
        # line to fix a error of maxsize int
        data=data.apply(lambda x:x.astype(np.float64,errors='ignore'))
        data = data[~data.index.duplicated()]
        data.to_parquet(path=self.file, engine=engine)
        return self

    def get(self):
        self._get_already_data()._check_marks()._check_already_ok()
        if self.df is not None:
            self._forward()
        while True:
            if not self.is_backward_ok:
                self._get_already_data()._check_marks()._check_already_ok()
                self._backward()
            else:
                break
        return self.df

class api_wrapper:
    def __init__(self):
        '''
        class to get data from ripple (xrp), it uses the xrpscan.com api
        '''
        # this variable needs to have the api url
        self.url = 'https://api.xrpscan.com/api/v1/ledger'
        # must contain the string of the timestamp
        self.time_column = 'close_time'
        # must contain the number of the block
        self.block_column = 'ledger_index'

    def get_latest(self):
        '''
        get the latest block
        :return: self or int (the number of the latest block) on '.latest'
        '''
        latest = requests.get(url=self.url)
        latest = json.loads(latest.text)['current_ledger']
        # self.latest must be a integer with the number of the latest block
        self.latest = latest
        return self

    def get_index(self, index, session=None):
        '''
        get a block by index
        :param index: the number of index
        :param session: optional a requests session can be passed here to get the data fastly
        :return: self or dict on '.index'
        '''
        session = session if session is not None else requests.session()
        data = session.get(url=f"{self.url}/{index}")
        data = json.loads(data.text)
        # self.index must be the data of the requested index this is a json that i can use it in a
        # pandas dataframe, be careful cause some apis return dict inside dict, to work with pandas well
        # the dict needs not to have subdicts
        self.index = data
        return self

    def get_blocks(self, date_start: datetime.datetime, jump_distance, n_request_by_step, block: 'None|int',
                   show_progress: bool, request_interval):
        '''
        function to get the blocks this will iterate to get all data
        :param date_start: is the date what i spect dataframe starts
        :param jump_distance: how jumps to do to get a date, this is necessary
        cause xrp blocks will each ~3 seconds, so there a lot of blocks, so i can jump to get
        blocks in a step of 200 in 200 blocks
        :param n_request_by_step: how much step is allowed to do in a request row
        think it as a difference between step an epochs in neural networks, this is like
        how much step in a epoch, a large amount of this will discart data at end
        :param block: the number of block to start, this function will get the datas from
        present to the past, so the block i can set when to start to get the data (preset),
        if none this will get the latest block on the network
        :param show_progress: true if print the progress will use tqdm
        :return: self or list[dicts] on '.blocks'
        '''
        self.date_start = date_start
        block = self.latest if block is None else block
        first_block = self.get_index(index=block).index
        responses = []
        # the value called by [self.time_column] must be the time of the 'block' variable
        with tqdm(total=self.get_index(block).index[self.time_column]-date_start.timestamp(), desc='Total Progress') as progress:
            while True:
                ex_futures = [f"{self.url}/{int(block-jump_distance*(1+n_request))}" for n_request in range(n_request_by_step)]
                with requests_futures.sessions.FuturesSession(max_workers=100) as future:
                    for item in tqdm(ex_futures, disable=1-show_progress, desc='api_requests'):
                        responses.append(future.get(item))
                        # the velow value will control how many requests by second so
                        # if the api provider is blocking this can be you are handling to many req
                        # per second, increase this sleed time
                        time.sleep(request_interval)

                concurrent.futures.wait([item for item in responses if isinstance(item, concurrent.futures.Future)], timeout=5)

                post_responses = []
                for item in responses:
                    result = item.result() if isinstance(item, concurrent.futures.Future) else item
                    stats = result.status_code if not isinstance(result, dict) else 200
                    if stats!=200:
                        raise Exception(f"Server returned an error: {stats}\n"
                                        f"result: {result}\n"
                                        f"result type: {type(result)}\n"
                                        f"item: {item}\n"
                                        f"item type: {type(item)}\n"
                                        f"url: {result.request.url}")
                    resultdict = json.loads(result.text) if not isinstance(result, dict) else result
                    post_responses.append(resultdict)
                responses = post_responses

                # this variable must return the lowest number of block in the responses
                min_index = min([item[self.block_column] for item in responses])
                # this variable must to contain the time of the index returned above
                min_index_date = self.get_index(index=min_index).index[self.time_column]
                if min_index_date <= date_start.timestamp():
                    break
                else:
                    block = min_index
                    progress.update((progress.total-(min_index_date-date_start.timestamp()))-progress.n)
        responses.insert(0, first_block)
        self.blocks = responses
        return self

    def blocks_dataframe(self):
        data = self.blocks
        df = pd.DataFrame(data)
        # here must be the name of the time column
        time_column = self.time_column
        df = df.sort_values(by=time_column)
        df = df.dropna(how='all')
        df = df.reset_index(drop=True)
        return df

    def get(self, date_start: datetime.datetime, jump_distance: int, n_requests_by_step: int, block: 'None|int',
            show_progress: bool, request_interval) -> pd.DataFrame:
        data = self.get_latest().get_blocks(date_start=date_start, jump_distance=jump_distance, n_request_by_step=n_requests_by_step,
                                            block=block, show_progress=show_progress,
                                            request_interval=request_interval).blocks_dataframe()
        return data
