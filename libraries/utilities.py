def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import math
import timeit
import collections
import statsmodels
import sklearn
from math import *
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.linear_model import *
from sklearn.pipeline import *
from sklearn.grid_search import *
from sklearn.learning_curve import *
from sklearn.model_selection import *
from sklearn.manifold import *
from sklearn.metrics import *
from scipy.cluster.hierarchy import *
from matplotlib.collections import *
from sklearn.decomposition import *
from sklearn.cross_decomposition import *


#####################################################
def get_data(currencies=[]):
    pn = pd.read_pickle('C:/Users/tfc_m/AnacondaProjects/Data/autoencoder/data/data.pkl')

    if currencies == []:
        currencies = list(pn.axes[0])

    df = pd.DataFrame()
    df_returns = pd.DataFrame()

    for currency in currencies:
        columns = list(pn[currency].columns)
        for column in columns:
            df[currency + '_' + column] = pn[currency][column]
            df_returns[currency + '_' + column] = np.log(pn[currency][column]) - np.log(pn[currency][column].shift(1))

    df['timestamp'] = pn[currency].index.to_pydatetime()
    df_returns['timestamp'] = pn[currency].index.to_pydatetime()
    df.set_index(np.arange(df.shape[0]), inplace=True)
    df_returns.set_index(np.arange(df_returns.shape[0]), inplace=True)
    df_returns.dropna(inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_returns['timestamp'] = pd.to_datetime(df_returns['timestamp'])
    df.set_index('timestamp', inplace=True)
    df_returns.set_index('timestamp', inplace=True)

    return df, df_returns


#########################################################
def get_columns(df, include_timestamp=True, n_first=-1):
    columns = list(df.columns)

    if ~include_timestamp:
        try:
            columns.remove('timestamp')
        except:
            pass

    if n_first > -1:
        columns = columns[:n_first]

    return columns


#########################################################
def get_errors(df_train, df_train_decoded, df_test, df_test_decoded, columns):
    rmse_train = {column:math.sqrt(mean_squared_error(df_train[column], df_train_decoded[column])) for column in columns}
    rmse_test = {column:math.sqrt(mean_squared_error(df_test[column], df_test_decoded[column])) for column in columns}

    results_train = pd.DataFrame.from_dict(rmse_train, orient='index').rename(columns={0:'RMSE'})
    results_test = pd.DataFrame.from_dict(rmse_test, orient='index').rename(columns={0:'RMSE'})
    results = pd.merge(results_train, results_test, left_index=True, right_index=True, suffixes=('_train', '_test'))
    results.sort_index(inplace=True)

    return results.copy()


#########################################################
def plot_decoded_data(df_train, df_train_decoded, df_test, df_test_decoded, columns, code_dim, column_to_plot=''):

    rmse_train = {column:math.sqrt(mean_squared_error(df_train[column], df_train_decoded[column])) for column in columns}
    rmse_test = {column:math.sqrt(mean_squared_error(df_test[column], df_test_decoded[column])) for column in columns}

    fig = plt.figure(5, figsize=(10, 10))
    ax1 = fig.add_subplot(211)
    original, = ax1.plot(df_train.index, np.cumsum(df_train[column_to_plot]), color='red', label='original')
    decoded, = ax1.plot(df_train.index, np.cumsum(df_train_decoded[column_to_plot]), color='green', label='decoded')
    error, = ax1.plot(df_train.index, np.cumsum(df_train[column_to_plot]) - np.cumsum(df_train_decoded[column_to_plot]), color='blue', label='error')
    ax1.set_title('Train set for %s (code dimension: %d, RMSE: %.3f)' % (column_to_plot, code_dim, rmse_train[column_to_plot]))
    ax1.legend(handles=[original, decoded, error])

    ax2 = fig.add_subplot(212)
    original, = ax2.plot(df_test.index, np.cumsum(df_test[column_to_plot]), color='red', label='original')
    decoded, = ax2.plot(df_test.index, np.cumsum(df_test_decoded[column_to_plot]), color='green', label='decoded')
    error, = ax2.plot(df_test.index, np.cumsum(df_test[column_to_plot]) - np.cumsum(df_test_decoded[column_to_plot]), color='blue', label='error')
    ax2.set_title('Test set for %s (code dimension: %d, RMSE: %.3f)' % (column_to_plot, code_dim, rmse_test[column_to_plot]))
    ax2.legend(handles=[original, decoded, error])

    plt.show()