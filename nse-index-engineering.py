from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import os
import opt_portfolio
import fetch_data
import datetime as dt
from get_data import get_stock_df, get_legacy_stock_df, get_index_df
import numpy as np
import plotly.express as px


app_description = """
Calculate the efficient frontier curve for given list of securities
"""
app_title = "Markowitz Efficient Frontier Calculation"

app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB, dbc.icons.FONT_AWESOME], title=app_title)
server = app.server

index_dict = {
    'nifty50': '^NSEI', 
    'sensex30': '^BSESN'
}

start_date = dt.datetime(2013, 5, 27)
end_date = dt.datetime(2022, 2, 1)

with open(os.path.join('Resources', 'ind_nifty50list.csv'), 'r') as read_file:
    stock_list = pd.read_csv(read_file, sep=',')['Symbol'].values
stock_list = [stock+'.NS' for stock in stock_list]

# stock_df = get_stock_df(stock_list=stock_list, start_date=start_date, end_date=end_date)
# stock_df.to_pickle('nifty_stock_df.pickle')

stock_df = get_legacy_stock_df()

index_df = get_index_df(index_dict, start_date, end_date)

#choosing the training interval
train_start = dt.datetime(2013, 5, 27)
train_end = dt.datetime(2018, 9, 17)
train_mean_returns, train_cov_matrix = fetch_data.get_data(stock_df, train_start, train_end)

max_SR_returns, max_SR_std, max_SR_allocation, min_vol_returns, \
        min_vol_std, min_vol_allocation, efficient_list, target_returns \
            = opt_portfolio.calculated_returns(train_mean_returns, train_cov_matrix)

max_sr_weights = max_SR_allocation['allocation']/100
min_vol_weights = min_vol_allocation['allocation']/100

test_end = dt.datetime(2021, 9, 17)
test_start = dt.datetime(2019, 9, 18)

test_mean_returns, test_cov_matrix = fetch_data.get_data(stock_df, test_start, test_end)

min_volatility_growth = fetch_data.get_portfolio_growth(stock_df, min_vol_weights, test_start, test_end)
max_SR_growth = fetch_data.get_portfolio_growth(stock_df, max_sr_weights, test_start, test_end)

min_volatility_CAGR = fetch_data.get_portfolio_CAGR(stock_df, min_vol_weights, test_start, test_end)
max_SR_CAGR = fetch_data.get_portfolio_CAGR(stock_df, max_sr_weights, test_start, test_end)

min_volatility_perf = opt_portfolio.portfolio_performance(min_vol_weights, test_mean_returns, test_cov_matrix)
max_SR_perf = opt_portfolio.portfolio_performance(max_sr_weights, test_mean_returns, test_cov_matrix)

max_SR_performance = (stock_df.dropna() * max_sr_weights).sum(axis=1)
min_vol_performance = (stock_df.dropna() * min_vol_weights).sum(axis=1)

summary_df = pd.DataFrame(columns=['Min. Vol. Portfolio', 'Max SR Portfolio'], \
    index=['Cumulative Growth','CAGR', 'Annualized Returns', 'Annualized Volatility'])

summary_df['Min. Vol. Portfolio'] = ['{:.2f}%'.format(min_volatility_growth*100), \
    '{:.2f}%'.format(min_volatility_CAGR*100), \
        '{:.2f}%'.format(min_volatility_perf[0]*100), '{:.2f}%'.format(min_volatility_perf[1]*100)]

summary_df['Max SR Portfolio'] = ['{:.2f}%'.format(max_SR_growth*100), \
    '{:.2f}%'.format(max_SR_CAGR*100), \
        '{:.2f}%'.format(max_SR_perf[0]*100), '{:.2f}%'.format(max_SR_perf[1]*100)]

summary_df_copy = summary_df.copy().reset_index()
summary_df_copy = summary_df_copy.rename(columns={'index':'Metric'})

#generating random portfolio weights to show they are suboptimal
num_dummy_weights = 500
dummy_performance_list = []

for i in range(num_dummy_weights):
    dummy_weights = np.random.rand(*max_sr_weights.shape)
    dummy_weights = dummy_weights / np.sum(dummy_weights)
    dummy_weights = pd.Series(data=dummy_weights, index=max_sr_weights.index)
    dummy_performance_list.append((opt_portfolio.portfolio_performance(dummy_weights, \
        train_mean_returns, train_cov_matrix)))

"""
==========================================================================
Markdown Text
"""

data_text = dcc.Markdown(
    """
    >The modern portfolio theory (MPT) by Harry Markowitz is a practical method 
    for selecting investments in order to maximize their overall returns 
    within an acceptable level of risk. Here I've tried to apply the Modern Portfolio Theroy 
    on the stocks listed at the National Stock Exchange in India.
    
    >I have engineered two portfolios, a high risk high reward strategy portfolio labelled below 
    as Maximum Sharpe Ratio Portfolio (Blue) and another that gives good returns with very low risk 
    labelled below as Minimum Volatility Portfolio (Red) and another portfolio that works on . 
    Finally the Green line shows the performance of the market index (Nifty 50). 
    You might notice below that the portfolios have vastly overperformed the benchmark index. 
    Especially the low risk portfolio becuase it did not go through the same drawdown at the onset
    of the pandemic in March 2020. 
    """
)
frontier_text = dcc.Markdown(
    """
    >How did I do this? Here is a teaser: I analyzed the historically available asset returns, 
    variances and covariances and from that I optimized the weights for each asset such that 
    I maximize my return while I minimize my risk. I teach about this in a cohort based [course](https://webinars.shikhargupta.com) that I teach for free.
    """
)

footer = html.Div(
    dcc.Markdown(
        """
        [Data source](https://finance.yahoo.com). The following tool allows the user to produce an efficient frontier base on historical prices of assets listed on National Stock Exchange (India). 
        It is intended to be a visualisation dashboard only, representative of the past movement in the product. 
        """
    ),
    className="p-2 mt-5 bg-primary text-white small",
)

"""
===========================================================================
Main Layout
"""

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.Br(),
                    html.H3(
                        "Engineering efficient portfolios from Indian stocks",
                        className="text-center"
                    )
                ]
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Hr(),
                        html.Div(id='output_container', children=[]),
                        html.H6(data_text, className="my-2")
                    ], width={"size": 10, "offset": 1}
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    width=3
                ),
                dbc.Col(
                    [
                        html.H4(
                            "Compare the performance against a benchmark portfolio",
                            className="text-center"
                        ),
                        dcc.Dropdown(id="select_index",
                            options=[
                                {"label": "Nifty 50 (National Stock Exchange)", "value": "nifty50"}, 
                                {"label": "Sensex 30 (Bombay Stock Exchange)", "value": "sensex30"}],
                            multi=True,
                            value='nifty50', 
                            placeholder="Choose a benchmark to compare the portfolio performance"
                        )
                    ]
                ),
                dbc.Col(
                    width=3
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="plot_portfolio_performance", figure={}), 
                    width={"size": 10, "offset": 1}
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Hr(),
                        html.H6(frontier_text, className="my-2"),
                    ], 
                    width={"size": 10, "offset": 1}
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="plot_efficient_frontier", figure={}),
                ),
                dbc.Col(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Dropdown(id="select_portfolio",
                                    options=[
                                        {"label": "View the Maximum Sharpe Ratio Portfolio", "value": "Max_SR"}, 
                                        {"label": "View the Minimum Volatility Portfolio", "value": "Min_Vol"}],
                                    multi=False,
                                    value="Max_SR"
                                    ),
                                    dcc.Graph(id="plot_portfolio_allotment", figure={})
                                ]
                            )
                        ]
                    )
                )
            ]
        ),
        dbc.Row(
            dbc.Col(
                html.H4(
                    "Performance of the engineered portfolios in a nut-shell",
                    className="text-center"
                ),
            )
        ), 
        dbc.Row(
            [
                dbc.Col(
                    width=3
                ),
                dbc.Col(
                    [
                        dash_table.DataTable(
                            id='table',
                            columns=[{"name": i, "id": i} for i in summary_df_copy.columns],
                            data=summary_df_copy.to_dict('records')
                        )
                    ]
                ),
                dbc.Col(
                    width=3
                )
            ]
        ),
        dbc.Row(
            dbc.Col(
                [
                    html.Br(),
                ]
            )
        )
    ],
    fluid=True,
)

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'), 
    Output(component_id='plot_efficient_frontier', component_property='figure'), 
    Output(component_id='plot_portfolio_allotment', component_property='figure'), 
    Output(component_id='plot_portfolio_performance', component_property='figure')],
    [Input(component_id='select_portfolio', component_property='value'), 
    Input(component_id='select_index', component_property='value')]
)

def update_graph(select_portfolio, select_index):
    container = " "
    #Max SR
    MaxSharpeRatio = go.Scatter(
        name='Maximium Sharpe Ratio Portfolio',
        mode='markers',
        x=[max_SR_std],
        y=[max_SR_returns],
        marker=dict(color='red',size=14,line=dict(width=3, color='black'))
    )

    #Min Vol
    MinVol = go.Scatter(
        name='Mininium Volatility Portfolio',
        mode='markers',
        x=[min_vol_std],
        y=[min_vol_returns],
        marker=dict(color='green', size=14, line=dict(width=3, color='black'))
    )

    #Efficient Frontier
    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std*100, 2) for ef_std in efficient_list],
        y=[round(target*100, 2) for target in target_returns],
        line=dict(color='black', width=4, dash='dashdot')
    )

    #Dummy Portfolios
    DummyPortfolio = go.Scatter(
        name='Sub-optimal portfolios',
        mode='markers',
        x=[round(i[1]*100, 2) for i in dummy_performance_list],
        y=[round(i[0]*100, 2) for i in dummy_performance_list], 
        marker=dict(color='mediumorchid', size=8, line=dict(width=0.75, color='black'))
    )

    data = [MaxSharpeRatio, MinVol, EF_curve, DummyPortfolio]
    layout = go.Layout(
        title = "Modern Portfolio Theory on NSE stocks<br>",
        yaxis = dict(title='Annualised Return (%)'),
        xaxis = dict(title='Annualised Volatility (%)'),
        showlegend = True,
        legend = dict(x = 0, y = 0.95, traceorder='normal', bgcolor='#E2E2E2', bordercolor='black', borderwidth=1),
        width=800,
        height=700)

    ef_fig = go.Figure(data=data, layout=layout)
    ef_fig.update_layout(title={'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
    
    if (select_portfolio == 'Max_SR'):    
        filtered_max_SR_df = max_SR_allocation[max_SR_allocation['allocation'] >= 1]
        alloc_fig = px.pie(filtered_max_SR_df, values='allocation', \
            names=filtered_max_SR_df.index, title='Portfolio with high risk - high reward', \
                width=500, height=700)
    
    if (select_portfolio == 'Min_Vol'):    
        filtered_min_vol_df = min_vol_allocation[min_vol_allocation['allocation'] >= 1]
        alloc_fig = px.pie(filtered_min_vol_df, values='allocation', \
            names=filtered_min_vol_df.index, title='Portfolio with minimum risk - good reward',\
                width=500, height=700)

    alloc_fig.update_layout(title={'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})

    portfolio_performance = pd.DataFrame(columns=['Max SR Portfolio', 'Min Vol Portfolio']) 
    portfolio_performance['Max SR Portfolio'] = max_SR_performance
    portfolio_performance['Min Vol Portfolio'] = min_vol_performance

    ind_df = index_df.copy()
    ind_df = ind_df[select_index]

    portfolio_performance = portfolio_performance.join(ind_df)
    portfolio_performance = portfolio_performance.divide(portfolio_performance.iloc[0]) * 100

    sub_title = "<sup>All investments are normalized to an initial 100 Rs investment 4.5 years ago (to capture all market cycles)</sup><br>"
    nav_fig = px.line(portfolio_performance, title='Portfolio performance<br>' + sub_title)
    nav_fig.update_layout(title={'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})


    return container, ef_fig, alloc_fig, nav_fig


if __name__ == "__main__":
    app.run_server(debug=True)
