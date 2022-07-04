import pandas as pd
import yfinance as yf
import numpy as np


def get_stock_df(stock_list, start_date, end_date):
    return yf.download(stock_list, start=start_date, end=end_date)['Adj Close']

def get_legacy_stock_df():
    return pd.read_pickle('nifty_stock_df.pickle')

def get_index_df(index_dict, start_date, end_date):
    index_df = pd.DataFrame()
    for index, index_ticker in index_dict.items():
        index_df[index] = yf.download(index_ticker, start=start_date, end=end_date)['Adj Close']
    return index_df

def get_legacy_index_df():
    return pd.read_pickle('index_stock_df.pickle')

def get_portfolio_weights_df():
    return pd.read_pickle('portfolio_weights_df.pickle')
    
def get_fund_comparison(data, tickers, return_type='mean', risk_free_rate=0, trading_days=248):
    daily_returns = data.pct_change()
    cumu_return = (np.prod((daily_returns + 1)) - 1).to_numpy()
    if (return_type == 'mean'):
        mean_returns = daily_returns
        ann_mean_returns = mean_returns.mean() * trading_days
        
        std = mean_returns.std() * np.sqrt(trading_days)
        downside_deviation = mean_returns[mean_returns<0].dropna().std()*np.sqrt(trading_days)
    
        sharpe_ratio = (ann_mean_returns - risk_free_rate) / std
        sortino_ratio = (ann_mean_returns - risk_free_rate) / downside_deviation
        keys = ['Cumulative returns', 'Annualized mean returns', 'Annualized volatility', \
                'Annualized downside volatility', 'Sharpe ratio', 'Sortino ratio']
        values = [cumu_return, ann_mean_returns, std, \
                  downside_deviation, sharpe_ratio, sortino_ratio]
        fund_comparison_df = pd.DataFrame(data=values, index=keys, columns=tickers)
    
    if (return_type == 'log'):
        log_returns = daily_returns.apply(lambda x: np.log(1+x))
        ann_log_returns = log_returns.mean() * trading_days
        
        std = log_returns.std() * np.sqrt(trading_days)
        downside_deviation = log_returns[log_returns<0].dropna().std()*np.sqrt(trading_days)
        
        sharpe_ratio = (ann_log_returns - risk_free_rate) / std
        sortino_ratio = (ann_log_returns - risk_free_rate) / downside_deviation
        keys = ['Cumulative returns', 'Annualized mean returns', 'Annualized volatility', \
                'Annualized downside volatility', 'Sharpe ratio', 'Sortino ratio']
        values = [cumu_return, ann_log_returns, std, \
                  downside_deviation, sharpe_ratio, sortino_ratio]
        fund_comparison_df = pd.DataFrame(data=values, index=keys, columns=tickers)
    
    return fund_comparison_df