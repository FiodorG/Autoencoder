from libraries import utilities, visualization
from autoencoder.libraries import utilities as utilities_autoencoder
from autoencoder.libraries import visualization_pca, utilities_pca
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
currency = 'GBPUSD_close'
n_components = 12
df, df_returns = utilities_autoencoder.get_data()
columns = utilities_autoencoder.get_columns(df_returns, include_timestamp=False)

scaler = sklearn.preprocessing.StandardScaler()
df_returns[columns] = scaler.fit_transform(df_returns[columns])
df_train = df_returns.iloc[:300000, ]
df_test = df_returns.iloc[300000:, ]


### one off PCA
pca = sklearn.decomposition.PCA(n_components=len(columns), whiten=False)
pca.fit(df_train)
components = pca.components_.T
#np.dot(components.T, components)
components[:, n_components:] = components[:, n_components:] * 0

df_train_encoded = np.dot(df_train - pca.mean_, components)
df_test_encoded = np.dot(df_test - pca.mean_, components)

df_train_decoded = df_train.copy()
df_test_decoded = df_test.copy()
df_train_decoded[columns] = np.dot(df_train_encoded, components.T) + pca.mean_
df_test_decoded[columns] = np.dot(df_test_encoded, components.T) + pca.mean_

result = utilities_autoencoder.get_errors(df_train, df_train_decoded, df_test, df_test_decoded, columns)

visualization_pca.plot_components_image(components)
visualization_pca.plot_component(components, columns, component_to_plot=0)
visualization_pca.plot_first_pcs(df_train, df_train_encoded, plot_3d=False)
utilities_autoencoder.plot_decoded_data(df_train, df_train_decoded, df_test, df_test_decoded, columns, n_components, currency)


### PCA as function of n_components
results_per_currency = utilities_pca.autoencoding_pca_many(df_train, df_test, columns)
visualization_pca.plot_all_columns(results_per_currency, columns)


### one off ICA
ica = sklearn.decomposition.FastICA(n_components=len(columns), whiten=False, random_state=42)
ica.fit(df_train)

df_train_encoded = ica.transform(df_train)
df_test_encoded = ica.transform(df_test)

visualization_pca.plot_first_pcs(df_train, df_train_encoded, plot_3d=False)

df_train_encoded[:, n_components:] = df_train_encoded[:, n_components:] * 0
df_test_encoded[:, n_components:] = df_test_encoded[:, n_components:] * 0

df_train_decoded = df_train.copy()
df_test_decoded = df_test.copy()
df_train_decoded[columns] = ica.inverse_transform(df_train_encoded)
df_test_decoded[columns] = ica.inverse_transform(df_test_encoded)

result = utilities_autoencoder.get_errors(df_train, df_train_decoded, df_test, df_test_decoded, columns)

utilities_autoencoder.plot_decoded_data(df_train, df_train_decoded, df_test, df_test_decoded, columns, n_components, currency)


### ICA as function of n_components
results_per_currency = utilities_pca.autoencoding_ica_many(df_train, df_test, columns)
visualization_pca.plot_all_columns(results_per_currency, columns)
