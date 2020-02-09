# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 23:08:12 2017

@author: Fiodor
"""

##### Fast Fourrier transform

def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = 10                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

x = np.array([669, 592, 664, 1005, 699, 401, 646, 472, 598, 681, 1126, 1260, 562, 491, 714, 530, 521, 687, 776, 802, 499, 536, 871, 801, 965, 768, 381, 497, 458, 699, 549, 427, 358, 219, 635, 756, 775, 969, 598, 630, 649, 722, 835, 812, 724, 966, 778, 584, 697, 737, 777, 1059, 1218, 848, 713, 884, 879, 1056, 1273, 1848, 780, 1206, 1404, 1444, 1412, 1493, 1576, 1178, 836, 1087, 1101, 1082, 775, 698, 620, 651, 731, 906, 958, 1039, 1105, 620, 576, 707, 888, 1052, 1072, 1357, 768, 986, 816, 889, 973, 983, 1351, 1266, 1053, 1879, 2085, 2419, 1880, 2045, 2212, 1491, 1378, 1524, 1231, 1577, 2459, 1848, 1506, 1589, 1386, 1111, 1180, 1075, 1595, 1309, 2092, 1846, 2321, 2036, 3587, 1637, 1416, 1432, 1110, 1135, 1233, 1439, 894, 628, 967, 1176, 1069, 1193, 1771, 1199, 888, 1155, 1254, 1403, 1502, 1692, 1187, 1110, 1382, 1808, 2039, 1810, 1819, 1408, 803, 1568, 1227, 1270, 1268, 1535, 873, 1006, 1328, 1733, 1352, 1906, 2029, 1734, 1314, 1810, 1540, 1958, 1420, 1530, 1126, 721, 771, 874, 997, 1186, 1415, 973, 1146, 1147, 1079, 3854, 3407, 2257, 1200, 734, 1051, 1030, 1370, 2422, 1531, 1062, 530, 1030, 1061, 1249, 2080, 2251, 1190, 756, 1161, 1053, 1063, 932, 1604, 1130, 744, 930, 948, 1107, 1161, 1194, 1366, 1155, 785, 602, 903, 1142, 1410, 1256, 742, 985, 1037, 1067, 1196, 1412, 1127, 779, 911, 989, 946, 888, 1349, 1124, 761, 994, 1068, 971, 1157, 1558, 1223, 782, 2790, 1835, 1444, 1098, 1399, 1255, 950, 1110, 1345, 1224, 1092, 1446, 1210, 1122, 1259, 1181, 1035, 1325, 1481, 1278, 769, 911, 876, 877, 950, 1383, 980, 705, 888, 877, 638, 1065, 1142, 1090, 1316, 1270, 1048, 1256, 1009, 1175, 1176, 870, 856, 860])
x = df['GBPUSD_close'].values
n_predict = 10000
extrapolation = fourierExtrapolation(x, n_predict)
plt.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label = 'extrapolation')
plt.plot(np.arange(0, x.size), x, 'b', label = 'x', linewidth = 3)
plt.legend()
plt.show()

#########################################################
def fourier_series(df_train, df_test, frequencies_to_keep=1, verbose=False, column_to_plot=''):

    columns = utilities_autoencoder.get_columns(df_train, include_timestamp=False)
    df_train_decoded = df_train.copy()
    df_test_decoded = df_test.copy()

    df_train_transform = scipy.fftpack.rfft(df_train[columns].values)
    train_frequencies = scipy.fftpack.fftfreq(n_train, 60)

    plt.plot(df_train_transform, color='red')
    plt.plot(df_train[columns], color='blue')

################################
    cut_f_signal = df_train_transform.copy()
    cut_f_signal[(np.abs(train_frequencies)>0.0025)] = 0
    restored_sig = scipy.fftpack.irfft(cut_f_signal)

    plt.plot(df_train_transform)
    plt.plot(cut_f_signal)
    
################################
#    plt.plot(df_train_transform)
#    plt.plot(train_frequencies)
#
#    if frequencies_to_keep>0:
#        df_train_transform[frequencies_to_keep:-frequencies_to_keep] = 0
#
#    restored_sig = scipy.fftpack.irfft(df_train_transform)

################################
    n_train = len(df_train)
    n_test = len(df_test)
    
    t = np.arange(0, n_train + n_test)
    restored_sig = np.zeros(len(t))
    
    indexes = list(range(n_train))
    indexes.sort(key = lambda i: np.absolute(train_frequencies[i]))

    for i in indexes[:1 + frequencies_to_keep * 2]:
        ampli = np.absolute(df_train_transform[i]) / n
        phase = np.angle(df_train_transform[i])
        restored_sig += ampli * np.cos(2 * np.pi * train_frequencies[i] * t + phase)


################################
    df_train_decoded[columns] = restored_sig[:300000]
    df_test_decoded[columns] = restored_sig[300000:, np.newaxis]
    
    rmse_train = math.sqrt(mean_squared_error(df_train[columns], df_train_decoded[columns]))
    rmse_test = math.sqrt(mean_squared_error(df_test[columns], df_test_decoded[columns]))

    if verbose:
        plot_decoded_data(
            df_train,
            df_train_decoded,
            df_test,
            df_test_decoded,
            rmse_train,
            rmse_test,
            frequencies_to_keep,
            column_to_plot
        )

    return df_train_decoded


def fft_inverse(Yhat, x):
    """Based on http://stackoverflow.com/a/4452499/190597 (mtrw)"""
    Yhat = np.asarray(Yhat)
    x = np.asarray(x).reshape(-1, 1)
    N = len(Yhat)
    k = np.arange(N)
    total = Yhat * np.exp(1j * x * k * 2 * np.pi / N)
    return np.real(total.sum(axis=1))/N

mydata = [8.3, 8.3, 8.3, 8.3, 7.2, 7.8, 7.8, 8.3, 9.4, 10.6, 10.0, 10.6, 11.1, 12.8,
         12.8, 12.8, 11.7, 10.6, 10.6, 10.0, 10.0, 8.9, 8.9, 8.3, 7.2, 6.7, 6.7, 6.7,
         7.2, 8.3, 7.2, 10.6, 11.1, 11.7, 12.8, 13.3, 15.0, 15.6, 13.3, 15.0, 13.3,
         11.7, 11.1, 10.0, 10.6, 9.4, 8.9, 8.3, 8.9, 6.7, 6.7, 6.0, 6.1, 8.3, 8.3,
         10.6, 11.1, 11.1, 11.7, 12.2, 13.3, 14.4, 16.7, 14.4, 13.3, 12.2, 11.7,
         11.1, 10.0, 8.3, 7.8, 7.2, 8.0, 6.7, 7.2, 7.2, 7.8, 10.0, 12.2, 12.8,
         12.8, 13.9, 15.0, 16.7, 16.7, 16.7, 15.6, 13.9, 12.8, 12.2, 10.6, 9.0,
         8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 10.0, 10.6, 11.1, 12.0, 11.7,
         11.1, 13.0, 13.3, 13.0, 11.1, 10.6, 10.6, 10.0, 10.0, 10.0, 9.4, 9.4,
         8.9, 8.3, 9.0, 8.9, 9.4, 9.0, 9.4, 10.6, 11.7, 11.1, 11.7, 12.8, 12.8,
         12.8, 13.0, 11.7, 10.6, 10.0, 10.0, 8.9, 9.4, 7.8, 7.8, 8.3, 7.8, 8.9,
         8.9, 8.9, 9.4, 10.0, 10.0, 10.6, 11.0, 11.1, 11.1, 12.2, 10.6, 10.0, 8.9,
         8.9, 9.0, 8.9, 8.3, 8.9, 8.9, 9.4, 9.4, 9.4, 8.9, 8.9, 8.9, 9.4, 10.0,
         11.1, 11.7, 11.7, 11.7, 11.7, 12.0, 11.7, 11.7, 12.0, 11.7, 11.0, 10.6,
         9.4, 10.0, 8.3, 8.0, 7.2, 5.6, 6.1, 5.6, 6.1, 6.7, 8.0, 10.0, 10.6, 11.1,
         13.3, 12.8, 12.8, 12.2, 11.1, 10.0, 10.0, 10.0, 10.0, 9.4, 8.3] 

Yhat = fftp.fft(mydata)
fig, ax = plt.subplots(nrows=2, sharex=True)
xs = np.arange(len(mydata))
ax[0].plot(xs, mydata)

new_xs = np.linspace(xs.min(), xs.max(), len(mydata)*1.5)
new_ys = fft_inverse(Yhat, new_xs)
ax[1].plot(new_xs, new_ys)

#def wavelet_series(x,y):
    

#Dummy test
res = fourier_series(df_train, frequencies_to_keep=100000, verbose=True, column_to_plot='GBPUSD_close')
y_t2 = fourier_series(x,y,wn,True,n)


plt.plot(np.cumsum(y), color='red')
plt.plot(np.cumsum(y_t2), color='green')
plt.plot(np.cumsum(y)-np.cumsum(y_t2), color='blue')
