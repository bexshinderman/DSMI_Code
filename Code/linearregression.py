# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:47:49 2020

@author: shind
"""

from sklearn import preprocessing
import numpy as np

np.set_printoptions(suppress=True) #line to prettify the output of numpy arrays so they are not ridiculously long
X = np.array([[ 1000, -1,  0.02],
              [ 1500,  2,  0.07],
              [ 1290,  1.5, -0.03]])
X_scaled = preprocessing.scale(X)
print(X_scaled, "x scaled")

X_train = X #Let's assume we already split the data into training set and test set
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print("X_train_scaled[:10]",X_train_scaled[:10])#Print the first 10 rows of the data
print("scaler.mean_",scaler.mean_ ) #Mean of each column
print( "scaler.scale_",scaler.scale_) #Per feature relative scaling of the data

X_test = np.array([[ 1100, -2,  0.03],
              [ 1200,  0.3,  -0.04],
              [ 1050,  1.4, -0.01]])
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled) 