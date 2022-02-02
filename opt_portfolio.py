# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:20:35 2021.

@author: gshik
"""
import numpy as np
import scipy.optimize as sc
import pandas as pd


def portfolio_performance(weights, mean_returns, cov_matrix, market_days=248):
    """
    Return the annulaized portfolio performance and volatility (std deviation).

    Parameters
    ----------
    weights : numpy.ndarray or list of floats
        Weightage of each stock in the final portfolio.
    mean_returns : pandas.core.series.Series
        Arithematic mean of the daily stock returns between the given datetime 
        range.
    cov_matrix : pandas.core.frame.DataFrame
        Covariance matrix containing the covariance between the stocks in the 
        stock_list.
    market_days : int, optional
        Number of days in an year the stocks are traded. The default is 248

    Returns
    -------
    p_returns : float64
        Annualized returns of the portfolio with the given stocks.
    p_std : float64
        Annualized standard deviation or the volatility of the portfolio.

    """
    p_returns = np.sum(weights*mean_returns)*market_days
    p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(market_days)
    return p_returns, p_std
                            
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Calculate the negative Sharpe Ratio.

    Parameters
    ----------
    weights : numpy.ndarray or list of floats
        Weightage of each stock in the final portfolio.
    mean_returns : pandas.core.series.Series
        Arithematic mean of the daily stock returns between the given datetime 
        range.
    cov_matrix : pandas.core.frame.DataFrame
        Covariance matrix containing the covariance between the stocks in the 
        stock_list.
    risk_free_rate : float, optional
        Rate at which one can borrow with zero volatility. The default is 0.0.

    Returns
    -------
    float
        Return the negative Sharpe ratio.

    """
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)  
    return -(p_returns-risk_free_rate)/p_std

# def negative_sortino_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
#     """
#     Calculate the negative Sortino Ratio.

#     Parameters
#     ----------
#     weights : numpy.ndarray or list of floats
#         Weightage of each stock in the final portfolio.
#     mean_returns : pandas.core.series.Series
#         Arithematic mean of the daily stock returns between the given datetime 
#         range.
#     cov_matrix : pandas.core.frame.DataFrame
#         Covariance matrix containing the covariance between the stocks in the 
#         stock_list.
#     risk_free_rate : float, optional
#         Rate at which one can borrow with zero volatility. The default is 0.0.

#     Returns
#     -------
#     float
#         Return the negative Sharpe ratio.

#     """
#     p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)  
#     p_std_downside = 
#     return -(p_returns-risk_free_rate)/p_std_downside

def maximize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0, constraint_set=(0, 1)):
    """
    Get the weights for stocks that will get maximum Sharpe ratio.

    Parameters
    ----------
    mean_returns : pandas.core.series.Series
        Arithematic mean of the daily stock returns between the given datetime 
        range.
    cov_matrix : pandas.core.frame.DataFrame
        Covariance matrix containing the covariance between the stocks in the 
        stock_list.
    risk_free_rate : float, optional
        Rate at which one can borrow with zero volatility. The default is 0.0.
    constraintSet : tuple, optional
        Constraint on the percentage allocation of each stock in the portfolio. 
        The default is (0, 1).

    Returns
    -------
    results : numpy.ndarray
        Returns the weights for given stocks such that the portfolio has 
        the maximum sharp ratio.
        
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for asset in range(num_assets))
    results = sc.minimize(negative_sharpe_ratio, np.full(num_assets, 1/num_assets), 
                          args=args, method='SLSQP', bounds=bounds, 
                          constraints=constraints)
    return results

def portfolio_return(weights, mean_returns, cov_matrix):
    """
    Get the annualized returns of a portfolio.

    Parameters
    ----------
    weights : numpy.ndarray or list of floats
        Weightage of each stock in the final portfolio.
    mean_returns : pandas.core.series.Series
        Arithematic mean of the daily stock returns between the given datetime 
        range.
    cov_matrix : pandas.core.frame.DataFrame
        Covariance matrix containing the covariance between the stocks in the 
        stock_list.

    Returns
    -------
    float64
        Annualized returns of the portfolio with the given stocks.

    """
    return portfolio_performance(weights, mean_returns, cov_matrix)[0]

def portfolio_variance(weights, mean_returns, cov_matrix):
    """
    Get the annualized volatility or standard deviation of a portfolio.

    Parameters
    ----------
    weights : numpy.ndarray or list of floats
        Weightage of each stock in the final portfolio.
    mean_returns : pandas.core.series.Series
        Arithematic mean of the daily stock returns between the given datetime 
        range.
    cov_matrix : pandas.core.frame.DataFrame
        Covariance matrix containing the covariance between the stocks in the 
        stock_list.

    Returns
    -------
    float64
        Annualized returns of the portfolio with the given stocks.

    """
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def minimize_variance(mean_returns, cov_matrix, constraint_set=(0, 1)):
    """
    Minimize the variance of a given portfolio by choosing specific weights.

    Parameters
    ----------
    mean_returns : pandas.core.series.Series
        Arithematic mean of the daily stock returns between the given datetime 
        range.
    cov_matrix : pandas.core.frame.DataFrame
        Covariance matrix containing the covariance between the stocks in the 
        stock_list.
    constraintSet : tuple, optional
        Constraint on the percentage allocation of each stock in the portfolio. 
        The default is (0, 1).

    Returns
    -------
    result : numpy.ndarray
        Returns the weights for the given stocks such that the portfolio has 
        the maximum sharp ratio.

    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for asset in range(num_assets))
    result = sc.minimize(portfolio_variance, np.full(num_assets, 1/num_assets), 
                         args=args, method='SLSQP', bounds=bounds, 
                         constraints=constraints)
    return result

def efficientOpt(mean_returns, cov_matrix, return_target, constraint_set=(0, 1)):
    """
    For each return target we want to optimize the portfolio for minimum variance.

    Parameters
    ----------
    mean_returns : pandas.core.series.Series
        Arithematic mean of the daily stock returns between the given datetime 
        range.
    cov_matrix : pandas.core.frame.DataFrame
        Covariance matrix containing the covariance between the stocks in the 
        stock_list.
    return_target : TYPE
        DESCRIPTION.
    constraintSet : tuple, optional
        Constraint on the percentage allocation of each stock in the portfolio. 
        The default is (0, 1).

    Returns
    -------
    result : numpy.ndarray
        Returns the weights for the given stocks such that the portfolio has 
        the maximum sharp ratio.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun':lambda x: portfolio_return(x, mean_returns, cov_matrix) - return_target}, 
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for asset in range(num_assets))
    effOpt = sc.minimize(portfolio_variance, np.full(num_assets, 1/num_assets), 
                         args=args, method='SLSQP', bounds=bounds, 
                         constraints=constraints)
    return effOpt


def calculated_returns(mean_returns, cov_matrix, risk_free_rate=0, constraint_set=(0, 1)):
    """
    We want the max SR and the min Vol portfolios and the efficient frontier.

    Parameters
    ----------
    mmean_returns : pandas.core.series.Series
        Arithematic mean of the daily stock returns between the given datetime 
        range.
    cov_matrix : pandas.core.frame.DataFrame
        Covariance matrix containing the covariance between the stocks in the 
        stock_list.
    risk_free_rate : float, optional
        Rate at which one can borrow with zero volatility. The default is 0.0.
    constraintSet : tuple, optional
        Constraint on the percentage allocation of each stock in the portfolio. 
        The default is (0, 1).

    Returns
    -------
    max_sharpe_ratio_returns, max_sharpe_ratio_std, max_sharpe_ratio_allocation, min_vol_returns, 
    min_vol_std, min_vol_allocation, efficient_list, target_returns.

    """
    # Max SR portfolio
    max_sharpe_ratio_results = maximize_sharpe_ratio(mean_returns, cov_matrix)
    max_sharpe_ratio_returns, max_sharpe_ratio_std = portfolio_performance(max_sharpe_ratio_results['x'], 
                                                      mean_returns, 
                                                      cov_matrix)
    max_sharpe_ratio_allocation = pd.DataFrame(max_sharpe_ratio_results['x'], index=mean_returns.index, 
                                    columns=['allocation'])
    max_sharpe_ratio_allocation['allocation'] = [round(i*100, 2) for i in max_sharpe_ratio_allocation['allocation']]
    
    # Min vol portfolio
    min_vol_results = minimize_variance(mean_returns, cov_matrix)
    min_vol_returns, min_vol_std = portfolio_performance(min_vol_results['x'], 
                                                         mean_returns, 
                                                         cov_matrix)
    
    min_vol_allocation = pd.DataFrame(min_vol_results['x'], index=mean_returns.index,
                                      columns=['allocation'])
    min_vol_allocation['allocation'] = [round(i*100, 2) for i in min_vol_allocation['allocation']]
    
    target_returns = np.linspace(min_vol_returns, max_sharpe_ratio_returns, 20)
    efficient_list = []
    for target in target_returns:
        efficient_list.append(efficientOpt(mean_returns, cov_matrix, target)['fun'])
    
    max_sharpe_ratio_returns, max_sharpe_ratio_std = round(max_sharpe_ratio_returns*100, 2), round(max_sharpe_ratio_std*100, 2)
    min_vol_returns, min_vol_std = round(min_vol_returns*100, 2), round(min_vol_std*100, 2)
    
    return max_sharpe_ratio_returns, max_sharpe_ratio_std, max_sharpe_ratio_allocation, min_vol_returns, \
        min_vol_std, min_vol_allocation, efficient_list, target_returns