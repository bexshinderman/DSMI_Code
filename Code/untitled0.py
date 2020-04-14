# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:32:38 2020

@author: Bex.0
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

data = pd.read_csv('churn.csv', index_col=0)
#print(data.head())
feature_col  = data.keys()
#feature_cols exclude account length as it is an id and would skew the data, and the last column is our target
feature_cols = feature_col[1:16]
#target is 
target = feature_col[17]
print("X ",feature_cols)
print("Target", target)

X = np.array(data[feature_cols])
y = np.array(data[target])
print("y ", target)
print("***", np.array(data[feature_cols[1]]) )
print("***", np.array(data[target]) )
X, y = shuffle(X, y, random_state=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33)
print(X_train)
