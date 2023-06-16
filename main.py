

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



####################################################################
# settings
# -------------------------
name1 = 'ACC.NS'
name2 = 'BOSCHLTD.NS'
train_start = '2016-05-30'
train_end = '2017-05-30'
adf_ci = 0.95
sig_test_ci = 0.95
robust_start = '2014-05-30'
robust_end = '2016-05-30'
rob_ci=0.95
if_train=True
if_set_trading_params=True

####################################################################



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

