import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import math
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
pd.set_option('display.max_columns', 32)
np.set_printoptions(precision=5, suppress=True)


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
def autoencoding_pca(df_train, df_test, columns, n_components):
    pca = sklearn.decomposition.PCA(n_components=len(columns), whiten=False)
    pca.fit(df_train)
    components = pca.components_.T
    components[:, n_components:] = components[:, n_components:] * 0

    df_train_encoded = np.dot(df_train - pca.mean_, components)
    df_test_encoded = np.dot(df_test - pca.mean_, components)

    df_train_decoded = df_train.copy()
    df_test_decoded = df_test.copy()
    df_train_decoded[columns] = np.dot(df_train_encoded, components.T) + pca.mean_
    df_test_decoded[columns] = np.dot(df_test_encoded, components.T) + pca.mean_

    results = get_errors(df_train, df_train_decoded, df_test, df_test_decoded, columns)

    return results, df_train_decoded, df_test_decoded


#########################################################
def autoencoding_pca_many(df_train, df_test, columns):
    results = {}
    n_columns = np.arange(1, len(columns) + 1)
    for n in n_columns:
        results[str(n)], _, _ = autoencoding_pca(df_train, df_test, columns, n)

    results_per_currency = {}
    for currency in columns:
        results_per_currency[currency] = {'train': list(), 'test': list()}
        for n in n_columns:
            result_for_n = results[str(n)]
            result_for_n = result_for_n[result_for_n.index.values == currency].values[0]
            results_per_currency[currency]['train'].append(result_for_n[0])
            results_per_currency[currency]['test'].append(result_for_n[1])

    return results_per_currency


#########################################################
def autoencoding_ica(df_train, df_test, columns, n_components):
    ica = sklearn.decomposition.FastICA(n_components=len(columns), whiten=False)
    ica.fit(df_train)
    df_train_encoded = ica.transform(df_train)
    df_test_encoded = ica.transform(df_test)

    df_train_encoded[:, n_components:] = df_train_encoded[:, n_components:] * 0
    df_test_encoded[:, n_components:] = df_test_encoded[:, n_components:] * 0

    df_train_decoded = df_train.copy()
    df_test_decoded = df_test.copy()
    df_train_decoded[columns] = ica.inverse_transform(df_train_encoded)
    df_test_decoded[columns] = ica.inverse_transform(df_test_encoded)

    results = get_errors(df_train, df_train_decoded, df_test, df_test_decoded, columns)

    return results, df_train_decoded, df_test_decoded


#########################################################
def autoencoding_ica_many(df_train, df_test, columns):
    results = {}
    n_columns = np.arange(1, len(columns) + 1)

    ica = sklearn.decomposition.FastICA(n_components=len(columns), whiten=False)
    ica.fit(df_train)
    df_train_encoded = ica.transform(df_train)
    df_test_encoded = ica.transform(df_test)

    for n in n_columns:
        df_train_encoded_filter = df_train_encoded.copy()
        df_test_encoded_filter = df_test_encoded.copy()
        df_train_encoded_filter[:, n:] = df_train_encoded_filter[:, n:] * 0
        df_test_encoded_filter[:, n:] = df_test_encoded_filter[:, n:] * 0

        df_train_decoded = df_train.copy()
        df_test_decoded = df_test.copy()
        df_train_decoded[columns] = ica.inverse_transform(df_train_encoded_filter)
        df_test_decoded[columns] = ica.inverse_transform(df_test_encoded_filter)

        results[str(n)] = get_errors(df_train, df_train_decoded, df_test, df_test_decoded, columns)

    results_per_currency = {}
    for currency in columns:
        results_per_currency[currency] = {'train': list(), 'test': list()}
        for n in n_columns:
            result_for_n = results[str(n)]
            result_for_n = result_for_n[result_for_n.index.values == currency].values[0]
            results_per_currency[currency]['train'].append(result_for_n[0])
            results_per_currency[currency]['test'].append(result_for_n[1])

    return results_per_currency
