import pandas as pd
import yfinance as yf


def get_stock_df(stock_list, start_date, end_date):
    return yf.download(stock_list, start=start_date, end=end_date)['Adj Close']

def get_legacy_stock_df():
    return pd.read_pickle('nifty_stock_df.pickle')

def get_index_df(index_dict, start_date, end_date):
    index_df = pd.DataFrame()
    for index, index_ticker in index_dict.items():
        index_df[index] = yf.download(index_ticker, start=start_date, end=end_date)['Adj Close']
    return index_df