import datetime
import logging

import MetaTrader5 as mt5
import pandas as pd

def mt5_getBars(currency: str, timeframe, shift_from_actual_bar: int = 1, bars: int = 10000,
                start_date: 'int|datetime.datetime|None' = None, end_date: 'int|datetime.datetime|None' = None):
    '''
    get bars data from mt5 client by position relative to actual bar or by date range


    :param currency : str = the symbol to get it must to be exactly as in mt5
    :param timeframe: mt5.timeframe = the timeframe to get the datas, need to be a mt5.timeframe
    :param shift_from_actual_bar: int = the number of bars to get since the actual ('not closed') bar ('actual bar is 0')
    :param bars: int = the number of bars to get
    :param start_date: optional intTimestampSeconds | datetime | None = if setted shift_from_actual_bar and bars will be ignored and will get data using the date
    :param end_date: optional intTimestampSeconds | datetime | None = as the previous

    :return: a pandas dataframe with time ohlcv and real_volume
    '''
    if mt5.terminal_info() is None:
        raise Exception(f"the mt5 can not been initialized, try to call 'mt5.initialize()'\n"
                        "before calling this function")
    if not mt5.symbol_select(currency, True):
        raise Exception(f"was not possible to add the currency: {currency} to the market watch")

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

    data = pd.DataFrame(data)
    data = data.rename(columns={'time':   'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume',
                                'spread': 'Spread', 'real_volume': 'Real_volume'})
    data = data.set_index(keys='Time')
    return data
