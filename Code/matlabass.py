# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:53:26 2020

@author: Bex.0
"""

import pandas as pd

data = pd.read_csv('crime.csv', index_col=0)
print(data.head())

import numpy as np
from sklearn.utils import shuffle
feature_cols = ['Education','Police','Income','Inequality']
target = ['Crime']
X = np.array(data[feature_cols])
y = np.array(data[target])
X, y = shuffle(X, y, random_state=1)

import matplotlib.pyplot as plt
X = data['Education']
y = data['Crime']


kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)

# Plot
plt.hist(X, **kwargs, color='g', label='Ideal')


plt.gca().set(title='Probability Histogram of Diamond Depths', ylabel='Probability')
plt.xlim(50,75)
plt.legend();