# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 00:30:29 2020

@author: Bex.0
"""

!pip install pmdarima
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
# Import
data = pd.read_csv('./averageMonthlyTemperatureNZ.csv', parse_dates=['date'], index_col='date',usecols=['date','AverageTemperature'])

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(data[:], label='Original Series')
axes[0].plot(data[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)


# Seasinal Dei
axes[1].plot(data[:], label='Original Series')
axes[1].plot(data[:].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Antidiabetic Drug Sales in Australia', fontsize=16)
plt.show()

import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(data, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=24,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()