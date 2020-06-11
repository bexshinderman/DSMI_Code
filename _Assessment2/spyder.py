# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:40:22 2020

@author: Bex.0
"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

dfM=pd.read_csv('./averageMonthlyTemperatureNZ.csv',index_col='date',parse_dates=True)
dfM
dfY=dfM.resample('Y').mean()
dfY

plt.rcParams.update({'figure.figsize': (4, 2), 'figure.dpi': 120})
dfY.plot(title="Average yearly temperature in New Zealand",linewidth=1)

plt.rcParams.update({'figure.figsize': (4, 2), 'figure.dpi': 120})
dfM.iloc[-120:].plot(title="Average monthly temperature in New Zealand (1993-2013)")
plt.xticks(rotation=45)

#1
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
series = read_csv('./averageMonthlyTemperatureNZ.csv', header=0, index_col=0, squeeze=True)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
print('The p value is less than 0.05 the data is stationary')

#2
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df = read_csv('./averageMonthlyTemperatureNZ.csv', usecols=['date'],parse_dates=['date'], index_col='date')
X = series.values
# Calculate ACF and PACF upto 50 lags
acf_50 = acf(X, nlags=50)
pacf_50 = pacf(X, nlags=50)

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(X.tolist(), lags=50, ax=axes[0])
plot_pacf(X.tolist(), lags=50, ax=axes[1])

#arima

import numpy as np, pandas as pd
import matplotlib.pyplot as plt


# Import data
df = pd.read_csv('./averageMonthlyTemperatureNZ.csv', names=['value'], header=0)
print(df)
df.plot()
from statsmodels.tsa.arima_model import ARIMA

p=1 #AR Term
d=0 #Number of differencing operations
q=2 #MA term
model = ARIMA(df.value, order=(p,d,q)) # 1,1,2 ARIMA Model
model_fit = model.fit(Idisp=0)
print(model_fit.summary())


# ANSWER GOES HERE
#!pip install pmdarima
#from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm
#import pandas as pd
from pandas import read_csv
#Data = read_csv('./averageMonthlyTemperatureNZ.csv', header=0, index_col=0, squeeze=True)
Data = read_csv('./averageMonthlyTemperatureNZ.csv', usecols=['date', 'AverageTemperature'],parse_dates=['date'], index_col='date')

df = pd.DataFrame(Data)
print(df)
print(df.dtypes)
print(df.date)
#df['




from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Import data
df = pd.read_csv('./averageMonthlyTemperatureNZ.csv', names=['value'], header=0)
model = pm.auto_arima(df.value, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True,
                      method='lbfgs',
                     )

print(model.summary())
model.plot_diagnostics(figsize=(7,5))
plt.show()

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
# Forecast
n_periods = 24
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df.value), len(df.value)+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df.value)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast")
plt.show()