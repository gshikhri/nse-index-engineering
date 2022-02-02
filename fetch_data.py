# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:27:53 2021.

@author: gshik
"""


import numpy as np

def get_portfolio_growth(stock_df, weights, start_date, end_date):
    """
    Calculate the growth of a portfolio over a given time period.
    
    Takes in the stock list in the portfolio, their weights in the portfolio
    the start and end date. It calculates the portfolio value on the start_date
    and the end_date and 
    returns the portfolio_growth = portfolio_end_val/ portfolio_start_val - 1
    
    Parameters
    ----------
    stock_df : pandas.core.frame.DataFrame
        Dataframe containing stock information, obtained from Yahoo.
    weights : numpy.ndarray or list of floats
        Weightage of each stock in the final portfolio..
    start_date : datetime.datetime
        start date for the data range for which stock data needs to be fetched.
    end_date : datetime.datetime
        end date for the data range for which stock data needs to be fetched.

    Returns
    -------
    float
        The growth in the portfolio.
    """
    p_start_value = np.sum((weights * stock_df.loc[stock_df.index == start_date]).T)
    p_end_value = np.sum((weights * stock_df.loc[stock_df.index == end_date]).T)
    return (p_end_value[0]/ p_start_value[0]) - 1

def get_portfolio_CAGR(stock_df, weights, start_date, end_date):
    
    p_start_value = np.sum((weights * stock_df.loc[stock_df.index == start_date]).T)
    p_end_value = np.sum((weights * stock_df.loc[stock_df.index == end_date]).T)

    duration = end_date.year - start_date.year
    
    if (duration<1):
        return ((p_end_value[0]/ p_start_value[0])) - 1 
    else:
        return np.power((p_end_value[0]/ p_start_value[0]), (1/duration)) - 1


def get_data(stock_df, start_date, end_date):
    """
    Get mean returns and covariance matrix for the selected stocks.

    Parameters
    ----------
    stock_df : pandas.core.frame.DataFrame
        Dataframe containing stock information, obtained from Yahoo.
    start_date : datetime.datetime
        start date for the data range for which stock data needs to be fetched.
    end_date : datetime.datetime
        end date for the data range for which stock data needs to be fetched.

    Returns
    -------
    mean_returns : pandas.core.series.Series
        Arithematic mean of the daily stock returns between the given datetime 
        range.
    cov_matrix : pandas.core.frame.DataFrame
        Covariance matrix containing the covariance between the stocks in the 
        stock_list.
    """
    mask = (stock_df.index > start_date) & (stock_df.index <= end_date)
    filtered_df = stock_df.loc[mask]
    stock_returns = filtered_df.pct_change()
    mean_returns = stock_returns.mean()
    cov_matrix = stock_returns.cov()  
    return mean_returns, cov_matrix

