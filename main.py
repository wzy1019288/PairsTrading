

# %%
# Importing Libraries
import os
from sklearn.linear_model import LinearRegression as lm 
import pandas as pd 
import numpy as np 
from itertools import combinations
import statsmodels.api as sm
import datetime as date
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.stats import t,ttest_ind,norm,iqr
from sklearn.metrics import r2_score
import pyfolio as pf
from numpy.linalg import inv

# Ignoring Warnings produced
import warnings
warnings.filterwarnings('ignore')

# Importing functions
from stats_func import (
    regression,
    dynamic_regression,
    test_stationarity,
    test_significance,
    cointegration_test,
    robustness
)

from backtest import (
    trading_algorithm
)


# %% settings

name1 = 'ACC.NS'
name2 = 'BOSCHLTD.NS'

train_start = '2016-05-30'
train_end = '2017-05-30'
adf_ci = 0.95   # adf test confidence interval
sig_test_ci = 0.95  # significance test confidence interval

robust_start = '2014-05-30'
robust_end = '2016-05-30'
rob_ci=0.95     # robustness confidence interval

test_start='2017-05-30'
test_end='2018-05-30'

if_use_kmf_when_cal_res=True    # whether to use kalman filter when cal residuals
if_use_kmf_when_cal_OU=True     # whether to use kalman filter when cal OU-process

if_train=True
init_params = {
    # Delta value for Kalman Filter for computing Cointegration Weight. Default is 0.0001
    'delta_resid': 0.0001,
    # Delta value for Kalman Filter for fitting to OU process. Default is 0.0001
    'delta_ou': 0.0001,
    # Number of standard deviations from the mean at which trade should be initiated
    'entry_point': 0.7,
    # The permissible slippage observed on residual spread: [Enter -999 for no slippage consideration]
    'slippage': 0.1,
    # The permissible stoploss observed on residual spread: [Enter -999 for no stoploss consideration]
    'stoploss': 0.3,
    # The commission on executing a short trade: [Enter 0 for no commission consideration]
    'comm_short': 0.0002,
    # The commission on executing a long trade: [Enter 0 for no commission consideration]
    'comm_long': 0.0002,
    # Risk free rate
    'rfr': 0.0685,
    # Maximum number of days a trade can last post the trade initiation: [Enter -999 for no maximum trade duration consideration]
    'max_trade_exit': 45
}
if_backtest=True

if_optimize=True
# Optimization attribute: Total Trades, Complete Trades, Incomplete Trades, Profit Trades, Loss Trades, Total Profit, Average Trade Duration,
# Average Profit, Win Ratio, Average Profit on Profitable Trades, Standard Deviation of Returns, Value at Risk, Expected Shortfall, Sharpe Ratio, 
# Sortino Ratio, Cumulative Return, Market Alpha, Market Beta, HML Beta, SMB Beta, WML Beta, Momentum Beta,Fama French Four Factor Alpha
optimize_params={
    'delta_resid':     {'lower_bound': 1e-10,  'upper_bound': 1e-1,  'val': 50, 'optimization_criteria': 'Sharpe Ratio', 'direction': 'max', 'if_opt': True},
    'delta_ou':        {'lower_bound': 1e-10,  'upper_bound': 1e-1,  'val': 50, 'optimization_criteria': 'Sharpe Ratio', 'direction': 'max', 'if_opt': True},  # 同时优化两个delta时，仅根据delta_resid的范围优化，delta_ou的参数无效
    'entry_point':     {'lower_bound': 0.3,    'upper_bound': 1.5,   'val': 20, 'optimization_criteria': 'Sharpe Ratio', 'direction': 'max', 'if_opt': True},
    'slippage':        {'lower_bound': 0.05,   'upper_bound': 0.3,   'val': 20, 'optimization_criteria': 'Sharpe Ratio', 'direction': 'max', 'if_opt': True},
    'max_trade_exit':  {'lower_bound': 20,     'upper_bound': 50,    'val': 30, 'optimization_criteria': 'Sharpe Ratio', 'direction': 'max', 'if_opt': True},
    'stoploss':        {'lower_bound': 0.15,   'upper_bound': 0.45,  'val': 20, 'optimization_criteria': 'Sharpe Ratio', 'direction': 'max', 'if_opt': True},
}

if_test=True
if_test_use_opt_params=True
####################################################################


# %%
def search_success_pairs():
    data_csv_names = [i for i in os.listdir('./data') if i not in ('ind_nifty100list.csv')]
    data_csv_names = [i for i in data_csv_names if pd.read_csv(os.path.join('./data', i))['Date'].min() < '2014-05-30']

    pairs = list(combinations(data_csv_names, 2))

    success_pairs = []
    for pair in pairs:

        flag = trading_algorithm(pair[0][:-4], pair[1][:-4], if_train=False)
        if flag != False:
            success_pairs.append(pair)

    len(success_pairs)
    pd.DataFrame(success_pairs).to_csv('success_pairs.csv', index=False)


# %%

# init_params['delta_resid'] = 1.e-10
# init_params['delta_ou'] = 0.002223
# init_params['entry_point'] = 0.5
# init_params['slippage'] = 0.26


trading_algorithm(name1, name2, 
                    train_start, train_end, adf_ci, sig_test_ci, 
                    robust_start, robust_end, rob_ci, 
                    test_start, test_end, 
                    if_use_kmf_when_cal_res, if_use_kmf_when_cal_OU, 
                    if_train, init_params, if_backtest, 
                    if_optimize, optimize_params,
                    if_test, if_test_use_opt_params
                    )
    
# %%
