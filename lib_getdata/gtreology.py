import logging
import os
from datetime import datetime

import lib_getdata as gd
import pandas as pd
import pytrends.request

import config

class gtreology():
    def __init__(self, pyt: pytrends.request.TrendReq, db_file: str, data_to_mount: dict, date_start: datetime, date_end: datetime = None,
                 sleep=0.2):
        '''
        Class to create and update a database of google trends data using the function 'pyt.get_historical_interest()'
        ps: pyt is the pytrends module.

        :param pyt: is the value returned of the class 'pytrends.request.TrendReq'
        :param db_file: is the location of db_file of google trends db
        :param data_to_mount: is a dict with shape dict={'name_of_db_file_with_word_for_gprop':['thing_to_download','thing_to_download',...],'name_of_other_db_file_with_word_for_gprop':['thing_to_download',...]} each key needs to be a db_file with their name.
        :param sleep: sleep param of 'pyt.get_historical_interest(sleep=)'
        '''
        self.data_to_mount = data_to_mount
        self.db_file = db_file
        self.pyt = pyt
        self.default_creation_starts = {'year_start': date_start.year, 'month_start': date_start.month,
                                        'day_start':  date_start.day, 'hour_start': date_start.hour}
        self.creation_ends = {'year_end': datetime.utcnow().year, 'month_end': datetime.utcnow().month,
                              'day_end':  datetime.utcnow().day, 'hour_end': datetime.utcnow().hour} if date_end==None else \
            {'year_end': date_end.year, 'month_end': date_end.month,
             'day_end':  date_end.day, 'hour_end': date_end.hour}
        self.sleep = sleep

    def list_remove_prefix_suffix(self, data: list, prefix: str, suffix: str):
        nword = []
        sword = ''
        for word in data:
            if str(word).startswith(prefix):
                sword = word[len(prefix):]
                if str(sword).endswith(suffix):
                    sword = sword[:-len(suffix)]
            nword.append(sword)
        return nword

    def update_db(self) -> None:
        for key in list(self.data_to_mount.keys()):
            if 'news' in str(key).lower():
                gprop = 'news'
                prefix = 'news_'
            elif 'images' in str(key).lower():
                gprop = 'images'
                prefix = 'images_'
            elif 'youtube' in str(key).lower():
                gprop = 'youtube'
                prefix = 'youtube_'
            elif 'shopping' in str(key).lower():
                gprop = 'froogle'
                prefix = 'shopping_'
            elif 'web' in str(key).lower():
                gprop = ''
                prefix = 'web_'
            else:
                etext = "cant find or not find the gprop word in the key, the key need to have: ['web','news','images','youtube','shopping']"
                logging.error(etext)
                raise Exception(etext)
            if os.path.isfile(self.db_file):
                db = pd.read_parquet(path=self.db_file, engine='fastparquet')
                db = pd.DataFrame(db).groupby(level=0).last()
            elif not os.path.isfile(self.db_file):
                db = pd.DataFrame()
            for item in self.data_to_mount[key]:
                suffix = f"_{item}"
                nword = self.list_remove_prefix_suffix(data=list(db.columns), prefix=prefix, suffix=suffix)
                if item in nword:
                    item = f"{prefix}{item}"
                    last_index_of_column = db[item].index[-1]
                    creation_starts = {'year_start':  last_index_of_column.year,
                                       'month_start': last_index_of_column.month,
                                       'day_start':   last_index_of_column.day, 'hour_start': last_index_of_column.hour}
                elif not item in nword:
                    creation_starts = self.default_creation_starts
                logging.info(f"downloading data to '{item}' in group: '{key}'.")
                ndata = self.pyt.get_historical_interest(keywords=[item], **creation_starts, **self.creation_ends,
                                                         gprop=gprop, sleep=self.sleep)
                ndata = pd.DataFrame(ndata).groupby(level=0).last()
                ndata = pd.DataFrame(ndata).add_prefix(prefix=prefix)
                if ndata is None:
                    e = Exception("download has returned 'None'")
                    logging.error(e)
                elif ndata.columns.size==0:
                    e = Exception(f"number of columns of returned data is: '{ndata.columns.size}'")
                    logging.error(
                            f"data has returned a data with '{ndata.columns.size}' columns")
                elif ndata.index.size==0:
                    e = Exception(f"number of rows of returned data is: '{ndata.index.size}'")
                    logging.error(
                            f"data has returned a data with '{ndata.index.size}' rows")
                db = gd.df_merge(old_df=db, new_df=ndata, update_old='auto', join_to_old_df='auto', sort_index=True,
                                 rsuffix=suffix)
                db = db.sort_index()
                db.to_parquet(path=self.db_file, index=True, engine='fastparquet')
                logging.info(f"update of data '{item}' in group: '{key}' has been completed.")
        logging.info('db update completed.')

