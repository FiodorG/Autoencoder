import matplotlib.pyplot as plt
import numpy as np
import sklearn
import math
import pylab
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D


#########################################################
def plot_component(components, columns, component_to_plot):
    f = pylab.figure(6, figsize=(8, 10))
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.barh(range(len(columns)), components[:, component_to_plot], align='center')
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)
    f.show()
    plt.show()


#########################################################
def plot_components_image(components):
    fig = plt.figure(1, figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(components, cmap='bwr', interpolation='none')
    ax.set_title('Loadings matrix')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Currencies')
    plt.show()


#########################################################
def plot_first_pcs(df_train, df_train_encoded, plot_3d=False):
    range_ = [np.min(df_train_encoded[:, :3]), np.max(df_train_encoded[:, :3])]
    fig = plt.figure(3, figsize=(12, 12))
    ax1 = fig.add_subplot(311)
    ax1.plot(df_train.index, df_train_encoded[:, 0], color='red')
    ax1.set_title('PC1')
    ax1.set_ylim(range_)
    ax2 = fig.add_subplot(312)
    ax2.plot(df_train.index, df_train_encoded[:, 1], color='blue')
    ax2.set_title('PC2')
    ax2.set_ylim(range_)
    ax3 = fig.add_subplot(313)
    ax3.plot(df_train.index, df_train_encoded[:, 2], color='green')
    ax3.set_title('PC3')
    ax3.set_ylim(range_)
    plt.show()

    if plot_3d:
        fig = plt.figure(4)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_train_encoded[:, 0], df_train_encoded[:, 1], df_train_encoded[:, 2])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show()


#########################################################
def plot_all_columns(results_per_currency, columns=[]):
    dx = 4
    dy = 8
    fig = plt.figure(figsize=(dy*2, dx*6))
    fig_num = 1
    for currency in columns:
        ax1 = fig.add_subplot(dy, dx, fig_num)
        train, = ax1.plot(np.arange(1, len(columns) + 1), results_per_currency[currency]['train'], color='red', label='train')
        test, = ax1.plot(np.arange(1, len(columns) + 1), results_per_currency[currency]['test'], color='blue', label='test')
        ax1.set_title('%s' % currency)
        ax1.set_xlim([1, 32])

        if fig_num >= 29:
            ax1.set_xlabel('Code dimension')

        ax1.set_ylabel('RMSE')
        ax1.legend(handles=[train, test])
        for key, spine in ax1.spines.items():
            spine.set_visible(False)
        ax1.tick_params(axis='both', which='major', labelsize=6)
        fig_num += 1

    plt.show()
