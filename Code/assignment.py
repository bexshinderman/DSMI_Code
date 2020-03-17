# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:03:02 2020

@author: shind
"""

import pandas as pd

data = pd.read_csv('crime.csv', index_col=0)
print(data.head())
print(data.keys())
print(data.Education)
print(data[:1]) # prints first entry of data

import matplotlib
import matplotlib.pyplot as plt



import numpy as np
from sklearn.utils import shuffle
feature_cols = ['Education','Police','Income','Inequality']
target = ['Crime']
X = np.array(data[feature_cols])
y = np.array(data[target])
X, y = shuffle(X, y, random_state=1)

education = X[:,0]
plt.scatter(education, y)


