# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:51:04 2020

@author: Bex.0
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

data = pd.read_csv('churn.csv', index_col=0)
#print(data.head())
feature_col  = data.keys()
feature_cols = feature_col[0:16]
target = feature_col[17]
print("X ",feature_cols)
print("y ", target)
X = np.array(data[feature_cols])
y = np.array(data[target])
X, y = shuffle(X, y, random_state=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33)