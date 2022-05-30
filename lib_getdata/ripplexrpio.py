import concurrent.futures
import datetime
import json
import time

import pandas as pd
import requests
import requests_futures.sessions
from tqdm import tqdm

class ripplexrpio:
    def __init__(self):
        '''
        class to get data from ripple (xrp), it uses the xrpscan.com api
        '''
        # this variable needs to have the api url
        self.url = 'https://api.xrpscan.com/api/v1/ledger'

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

    def get_blocks(self, date_start: datetime.datetime, jump_distance, n_request_by_step, block: int = None, show_progress: bool = True):
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
        datas_blocks = []
        block = self.latest if block is None else block
        first_block = self.get_index(index=block).index
        responses = []
        # the value called by ['close_time'] must be the time of the 'block' variable
        with tqdm(total=self.get_index(block).index['close_time']-date_start.timestamp(),desc='Total Progress') as progress:
            while True:
                ex_futures = [f"{self.url}/{block-jump_distance*(1+n_request)}" for n_request in range(n_request_by_step)]
                with requests_futures.sessions.FuturesSession(max_workers=100) as future:
                    for item in tqdm(ex_futures, disable=1-show_progress, desc='api_requests'):
                        responses.append(future.get(item))
                        # the velow value will control how many requests by second so
                        # if the api provider is blocking this can be you are handling to many req
                        # per second, increase this sleed time
                        time.sleep(0.05)
                responses = [json.loads(item.result().text) if isinstance(item, concurrent.futures.Future) else item for item in responses]
                # this variable must return the lowest number of block in the responses
                min_index = min([item['ledger_index'] for item in responses])
                # this variable must to contain the time of the index returned above
                min_index_date=self.get_index(index=min_index).index['close_time']
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
        time_column = 'close_time'
        df = df.sort_values(by=time_column)
        df = df.where(df[time_column] >= self.date_start.timestamp())
        df = df.dropna(how='all')
        df = df.reset_index(drop=True)
        return df

    def get(self, date_start: datetime.datetime, jump_distance: int, n_requests_by_step: int, block: 'None|int' = None,
            show_progress: bool = True) -> pd.DataFrame:
        data = self.get_latest().get_blocks(date_start=date_start, jump_distance=jump_distance, n_request_by_step=n_requests_by_step,
                                            block=block, show_progress=show_progress).blocks_dataframe()
        return data
