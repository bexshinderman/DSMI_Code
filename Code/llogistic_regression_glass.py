# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:35:18 2020

@author: Bex.0
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
feature_cols = ['ri','na','mg','al','si','k','ca','ba','fe', 'household']
data = pd.read_csv('glass.csv', names=feature_cols)
print(data.head())
print(data.shape)
#columns = data.columns
#print("collumns" ,data.columns)
#place predictors ri to household  colummns except for household our feature

print("*************")
X = data.values[:,0:8]
print("X:",X)
y = data.values[:,9]
#y = data['household']
print("Y:",y)
X, y = shuffle(X, y, random_state=1)
X_train = X[:75]
y_train = y[:75]
X_test = X[25:]
y_test = y[25:]


from sklearn.linear_model import LogisticRegression
logRegModel = LogisticRegression()
logRegModel.fit(X_train, y_train)