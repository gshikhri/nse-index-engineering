# nse-index-engineering: 

Repository for the dashboard app deployed (here)[http://nse-index-engineering.herokuapp.com]

- [Project overview](#project-overview)
- [Methodology](#Methodology)
- [Issues and questions](#issues-and-questions)

## Project overview
This is an implementation of the Modern Portfolio Theory (MPT.)[https://www.investopedia.com/terms/m/modernportfoliotheory.asp] 
Here, I have applied the MPT on the stocks listed at the National Stock Exchange in India (NSE). 
I have engineered two portfolios, a high risk-high reward strategy portfolio labelled below as Maximum Sharpe Ratio Portfolio (Blue) and another that gives good returns with very low risk labelled below as Minimum Volatility Portfolio (Red) and another portfolio that works on.

Both the portfolios vastly overperform the benchmark as seen here (Benchmark Nifty 50 in green). An interesting observation from this portfolio is the minimal drawdown during the onset of Covid induced market correction for the minimum volatility portfolio (Highlighted in Red).

## Methodology
The main idea is to designate weights to individual stocks such that the resulting portfolio maximizes the returns while maximizing the (Sharpe Ratio)[https://en.wikipedia.org/wiki/Sharpe_ratio] (Risk Adjusted Returns).
So given the mean returns and covariance betweeen the individual stocks, I define the negative sharpe ratio as: 

```python
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
```
This is then maximized using (scipy.minimize)[https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html]
```
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
```

## Issues and questions
In case you need help or have suggestions or you want to report an issue, please do so in a reproducible example at the corresponding [GitHub page](https://github.com/gshikhri/find-my-constellation/issues).
