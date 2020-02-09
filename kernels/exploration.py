import os
import sys
module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from libraries import utilities, visualization
from autoencoder.libraries import utilities as utilities_autoencoder
import matplotlib.pyplot as plt
import collections
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, is_valid_linkage
import scipy
import statsmodels.tsa.api as smt


currency = 'GBPUSD'
df, df_returns = utilities_autoencoder.get_data()
columns = utilities_autoencoder.get_columns(df_returns, include_timestamp=False)


### Time stats
timestamps = df[['timestamp']]
plt.plot(df.index, timestamps.timestamp)
print('First Date: %s' % timestamps.timestamp.values[0])
print('Last Date:  %s' % timestamps.timestamp.values[-1])
print('# of timestamps: %s' % len(timestamps))

timestamps_diff = timestamps.diff().dropna()
timestamps_diff.timestamp = timestamps_diff.timestamp / np.timedelta64(1, 's')
print(dict(collections.Counter(timestamps_diff.timestamp)))
plt.plot(timestamps_diff.index, timestamps_diff.timestamp)

print('Smallest diff: %s at %s' % (min(timestamps_diff.timestamp), np.argmin(timestamps_diff.timestamp)))
print('Largest diff:  %s at %s' % (max(timestamps_diff.timestamp), np.argmax(timestamps_diff.timestamp)))


### Plotting the data
visualization.plot_all_columns(df, columns=columns, drop='timestamp', title='HLOC spots for %s' % currency)
visualization.plot_all_columns(df_returns, columns=columns, drop='timestamp', title='HLOC returns for %s' % currency)

plt.figure(1)
plt.plot(df.timestamp[800:1000,], df[currency + '_low'][800:1000,], color='red')
plt.plot(df.timestamp[800:1000,], df[currency + '_high'][800:1000,], color='green')
plt.plot(df.timestamp[800:1000,], df[currency + '_open'][800:1000,], color='orange')
plt.plot(df.timestamp[800:1000,], df[currency + '_close'][800:1000,], color='blue')
plt.title('HLOC spots for %s, 200 timestamps of 4Jan16' % currency)
plt.show()

plt.figure(2)
plt.plot(df_returns.timestamp[800:900,], df_returns[currency + '_low'][800:900,], color='red')
plt.plot(df_returns.timestamp[800:900,], df_returns[currency + '_high'][800:900,], color='green')
plt.plot(df_returns.timestamp[800:900,], df_returns[currency + '_open'][800:900,], color='orange')
plt.plot(df_returns.timestamp[800:900,], df_returns[currency + '_close'][800:900,], color='blue')
plt.title('HLOC returns for %s, 200 timestamps of 4Jan16' % currency)
plt.show()

plt.plot(df_returns.timestamp[800:900,], df_returns[currency + '_low'][800:900,], color='red')
plt.plot(df_returns.timestamp[800:900,], df_returns[currency + '_high'][800:900,], color='green')
plt.plot(df_returns.timestamp[800:900,], df_returns[currency + '_open'][800:900,], color='orange')
plt.plot(df_returns.timestamp[800:900,], df_returns[currency + '_close'][800:900,], color='blue')
plt.title('HLOC returns for %s, 200 timestamps of 4Jan16' % currency)

fig = plt.figure()
ax1 = fig.add_subplot(411)
ax1.hist(df_returns[currency + '_low'].clip(-0.001, 0.001), bins=100, color='red')
ax1.set_title('Histogram of %s_low returns' % currency)
ax2 = fig.add_subplot(412)
ax2.hist(df_returns[currency + '_high'].clip(-0.001, 0.001), bins=100, color='green')
ax2.set_title('Histogram of %s_high returns' % currency)
ax3 = fig.add_subplot(413)
ax3.hist(df_returns[currency + '_open'].clip(-0.001, 0.001), bins=100, color='orange')
ax3.set_title('Histogram of %s_open returns' % currency)
ax4 = fig.add_subplot(414)
ax4.hist(df_returns[currency + '_close'].clip(-0.001, 0.001), bins=100, color='blue')
ax4.set_title('Histogram of %s_close returns' % currency)
fig.show()


### Describe dataset
visualization.describe_df(df)

df_returns.set_index('timestamp')[[currency + '_close']].rolling(24*60).mean().plot(title='24h rolling mean')
df_returns.set_index('timestamp')[[currency + '_close']].rolling(24*60).std().plot(title='24h rolling std')
df_returns.set_index('timestamp')[[currency + '_close']].rolling(24*60).skew().plot(title='24h rolling skew')
df_returns.set_index('timestamp')[[currency + '_close']].rolling(24*60).kurt().plot(title='24h rolling kurt')


### Autocorrelation
visualization.plot_autocorrelation(df_returns.head(10000), currency + '_close')
visualization.plot_autocorrelation(df_returns.head(10000), currency + '_close', squared=True)
visualization.plot_autocorrelation(df_returns.head(10000), currency + '_high')
visualization.plot_autocorrelation(df_returns.head(10000), currency + '_high', squared=True)


### Classification datasets
visualization.correlation_matrix(df, method='kendall', last_row=False)
visualization.correlation_matrix(df_returns, method='kendall', last_row=False)


### PCA
visualization.pca(df.drop('timestamp', axis=1))
visualization.pca(df_returns.drop('timestamp', axis=1))


### KDE-sklearn
x_grid = np.linspace(-0.001, 0.001, 100)
kde_low = visualization.kde_sklearn(df_returns[currency + '_low'], bandwidth=0.00005, x_grid=x_grid)
kde_high = visualization.kde_sklearn(df_returns[currency + '_high'], bandwidth=0.00005, x_grid=x_grid)
kde_open = visualization.kde_sklearn(df_returns[currency + '_open'], bandwidth=0.00005, x_grid=x_grid)
kde_close = visualization.kde_sklearn(df_returns[currency + '_close'], bandwidth=0.00005, x_grid=x_grid)
fig = plt.figure()
ax = fig.add_subplot(111)
low_plot, = ax.plot(x_grid, kde_low, color='red', alpha=0.5, lw=3, label='low')
high_plot, = ax.plot(x_grid, kde_high, color='green', alpha=0.5, lw=3, label='high')
open_plot, = ax.plot(x_grid, kde_open, color='orange', alpha=0.5, lw=3, label='open')
close_plot, = ax.plot(x_grid, kde_close, color='blue', alpha=0.5, lw=3, label='close')
ax.legend(handles=[low_plot, high_plot, open_plot, close_plot])
ax.set_title('Kernel density estimation of %s returns' % currency)
plt.show()


### KDE-statsmodel
visualization.kde_statsmodel(df_returns[currency + '_low'].clip(-0.001, 0.001))


### Stationarity
visualization.stationarity_test(df[currency + '_close'])
visualization.stationarity_test(df_returns[currency + '_close'])
pvalues_spot = visualization.stationarity_test_many(df.head(10000).drop('timestamp', axis=1), return_value='pvalue')
pvalues_returns = visualization.stationarity_test_many(df_returns.head(10000).drop('timestamp', axis=1), return_value='pvalue')


### DTW
dtw_matrix = utilities.DTW_distance_matrix(df.head(1000).drop('timestamp', axis=1), 1)
dtw_matrix = dtw_matrix.T + dtw_matrix
linkage_matrix = linkage(ssd.squareform(dtw_matrix), method='weighted', metric='euclidean')
print(is_valid_linkage(linkage_matrix))

plt.figure(1)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
scipy.cluster.hierarchy.dendrogram(linkage_matrix, labels=columns, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.show()

dtw_matrix = utilities.DTW_distance_matrix(df_returns.head(1000).drop('timestamp', axis=1), 1)
dtw_matrix = dtw_matrix.T + dtw_matrix
linkage_matrix = linkage(ssd.squareform(dtw_matrix), method='weighted', metric='euclidean')
print(is_valid_linkage(linkage_matrix))

plt.figure(1)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
scipy.cluster.hierarchy.dendrogram(linkage_matrix, labels=columns, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.show()


### Seasonality
df.reset_index(inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
smt.seasonal_decompose(df.GBPUSD_close, freq=60).plot()