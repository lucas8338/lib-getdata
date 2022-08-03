import holidays

import pandas as pd

class maldays():
    def __init__(self, countries: list, date_start, date_end, freq):
        self.countries = countries
        self.date_start = date_start
        self.date_end = date_end
        self.freq = freq

    def _get(self):
        dfs = []
        for country in self.countries:
            data = holidays.country_holidays(country=country, years=[self.date_start.year, self.date_end.year+1])
            data_df = pd.Series(data.values(), index=data.keys(), name=str(country))
            dfs.append(data_df)
        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.asfreq(freq='1D').fillna(value='noEvent').ffill()
        df = df.asfreq(freq=self.freq).ffill()
        df = df.loc[self.date_start:self.date_end]
        self.df = df
        return self

    def get(self):
        self._get()
        return self.df
