import datetime
import logging
import re
import time
from typing import Literal

import bs4

import pandas as pd
import requests

class econovents:
    def __init__(self, date_start: datetime.datetime, date_end: datetime.datetime, importance: Literal[1, 2, 3], countries: 'list[str]'):
        """
        class to get a economic calendar, this data is obtained from scrapping.
        :param date_start:
        :param date_end:
        :param importance: 1,2 and 3 are level of impacts, 3 will bring high impact events only and 1 will bring events of impact
        2 and 3
        :param countries: is a list of coutries by 3 letters 'ISO 3166'
        """
        self.date_start = date_start
        self.date_end = date_end
        self.importance = importance
        self.countries = countries

    def get(self) -> pd.DataFrame:
        """
        this method or this class 'econovents' is to fix a problem to the inner class 'economic_events', the problem
        is the site has a limit of data to return, so this class will verify if the returned data (time) range is currectly
        whether not it will does a second request with a time ahead and concatenate the old and the new dataframe and
        return the time was requested
        """
        data1: pd.DataFrame = self.economic_events(date_start=self.date_start, date_end=self.date_end, importance=self.importance,
                                                   countries=self.countries).get()
        data2 = None
        for i in range(5000):
            data1 = pd.concat([data1, data2], axis=0).drop_duplicates(ignore_index=True)
            last_time = data1['time'].max()
            is_ended_currectly = True if last_time.timestamp() >= self.date_end.timestamp() else False

            if is_ended_currectly is True:
                data1 = data1.reset_index(drop=True)
                return data1.where(data1['time'].between(left=self.date_start, right=self.date_end)).dropna(how='all')
            else:
                new_date_start = last_time-datetime.timedelta(days=1)
                new_date_end = last_time+datetime.timedelta(days=365)
                # will sleep by 1 second to not flooding the site
                time.sleep(1)
                data2 = self.economic_events(date_start=new_date_start, date_end=new_date_end, importance=self.importance,
                                             countries=self.countries).get()

    class economic_events:
        def __init__(self, date_start: datetime.datetime, date_end: datetime.datetime, importance: Literal['1', '2', '3'], countries: list):
            self.date_start = date_start
            self.date_end = date_end
            self.importance = importance
            self.countries = countries

            self.rows = []

        def _get_data(self):
            # the site return data based on the cookies setting
            cookies = {'cal-custom-range':    f"{str(self.date_start)}|{str(self.date_end)}", 'cal-timezone-offset': '0',
                       'calendar-importance': str(self.importance),
                       'calendar-countries':  str(self.countries).replace('[', '').replace(']', '').replace(' ', '').replace("'", ""), }
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36'
            req = requests.get(url='https://tradingeconomics.com/calendar', headers={'User-Agent': user_agent}, cookies=cookies,
                               verify=False)
            self.data_raw = req.content
            return self

        class date_format:
            def to(s):
                return '/'.join(s.strip().split(' ')[1:])

            def verify(s):
                date_format_verify = re.compile("[A-z]*/[\d]{2}/[0-9]{4}(?![^\r])(?![^\n])")
                result = date_format_verify.match(s)
                return True if result is not None else False

        def _preprocess(self):
            soup = bs4.BeautifulSoup(self.data_raw, features="lxml")
            table_all = soup.findAll(name='table', attrs={'id': 'calendar'})[0]

            total_size = len(table_all.findAllNext(name='thead', attrs={'class': 'table-header'}))

            self.ts = table_all.findAllNext()
            return self

        def _reset(self, reset_date: bool):
            self.date = None if reset_date is True else self.date
            self.time = None
            self.country = None
            self.event = None
            self.importance = None
            self.actual = None
            self.previous = None
            self.revised = False
            self.consensus = None
            self.forecast = None

        def _scrap(self):
            '''
            this is the scrapper itself, the site format is a bit unique, them uses a format where the data day is over
            the data: " table thead row row row ... thead row ... thead ... thead /table"
            the table is the table with the datas and the thead is a element containing the "date" (month, day, year)
            and the time (hours) is in the rows, for this motive the scrapper will looking in each element
            building the data, an element with class "td-alert" is aproximaly the last element of each row,
            so it is used as separator of rows, and when a new "date" is found it mean the day ends and the
            nexts times will belongs to a new day the date is get from element by regex.
            '''
            self._reset(reset_date=True)
            ts = self.ts
            for i in range(len(ts)):
                item = ts[i]
                tv_isdate = self.date_format.to(item.text)
                if self.date_format.verify(tv_isdate):
                    self.date = tv_isdate
                if 'class' in item.attrs:
                    if 'calendar-date' in str(item.attrs['class']):
                        self.time = item.text.strip()
                        if 'calendar-date-1' in str(item.attrs['class']):
                            self.importance = 1
                        elif 'calendar-date-2' in str(item.attrs['class']):
                            self.importance = 2
                        elif 'calendar-date-3' in str(item.attrs['class']):
                            self.importance = 3
                if 'class' in item.attrs:
                    if 'calendar-iso' in str(item.attrs['class']):
                        self.country = str(item.attrs['title']).strip()
                if 'data-event' in item.attrs:
                    self.event = str(item.attrs['data-event']).strip()
                if 'id' in item.attrs:
                    if item.attrs['id']=='actual':
                        self.actual = item.text.strip()
                if 'id' in item.attrs:
                    if item.attrs['id']=='previous':
                        self.previous = item.text.strip()
                if 'id' in item.attrs:
                    if item.attrs['id']=='revised':
                        if len(item.text.strip()) > 0:
                            self.revised = True
                if 'id' in item.attrs:
                    if item.attrs['id']=='consensus':
                        self.consensus = item.text.strip()
                if 'id' in item.attrs:
                    if item.attrs['id']=='forecast':
                        self.forecast = item.text.strip()
                if 'class' in item.attrs:
                    if 'td-alert' in str(item.attrs['class']):
                        self._make_row()
                        self._reset(reset_date=False)

        def _make_row(self):
            row = {'date':     self.date, 'time': self.time, 'country': self.country, 'event': self.event, 'importance': self.importance,
                   'actual':   self.actual, 'previous': self.previous, 'revised': self.revised, 'consensus': self.consensus,
                   'forecast': self.forecast}
            self.rows.append(row)
            return self

        def _dfformat(self):
            df = pd.DataFrame(self.rows)
            df['time'] = df['time'].fillna('00:00 AM')
            df['date'] = df['date']+' '+df['time']
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop(columns=['time'])
            df = df.rename(columns={'date': 'time'})
            # this will change the nan for pd.NA, this one is a type of missing but for example this will not be
            # handled by a interpolation for example, read more: shorturl.at/oqLYZ
            df = df.fillna(value=pd.NA)
            return df

        def get(self):
            self._get_data()._preprocess()._scrap()
            return self._dfformat()

class uniquify:
    def __init__(self, df):
        '''
        will transform the data in a suitable format to machine learning, every country and every data will have the
        self column, this is done using pd.pivot
        '''
        self.df = df

    def _run(self):
        data = self.df

        # the regex bellow will match numbers like: "usd '10.99'","usd'10.99'","'10.99'usd","'-10'usd" and others
        reg = "-{0,1}[0-9]{1,}\.{0,1}[0-9]{0,}"
        data = data.reset_index(drop=True)
        data = data.dropna(how='all', axis=1).dropna(how='all', axis=0)
        data = data.drop_duplicates().T.drop_duplicates().T
        data = data.pivot_table(index='time', columns=['country', 'event'], aggfunc='first')
        data = data.swifter.apply(
                lambda x: x.apply(lambda y: float(max(re.findall(reg, str(y)), key=len)) if re.search(reg, str(y)) is not None else pd.NA))
        newcolumns = ['_'.join(column).replace(' ', '_').replace('.', 'dot').replace('/', 'and').replace('-', '').replace('&', 'and') for
                      column in data.columns]
        data.columns = data.columns.get_level_values(0)
        data.columns = newcolumns
        self.tdf = data

    def get(self):
        logging.warning('''this function can take minutes depending the size of the datas, so is recommended to save the data
        for visualization or posterior use''')
        logging.warning('is recommended too to use a reduction dimensionality thecnique to reduce the size of datas')
        self._run()
        return self.tdf
