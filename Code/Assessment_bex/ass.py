# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:41:27 2020

@author: Bex.0
"""

import pandas as pd

data = pd.read_csv('crime.csv', index_col=0)
print(data.head())

import numpy as np
from sklearn.utils import shuffle
feature_cols = ['Education','Police','Income','Inequality']
target = ['Crime']
X = np.array(data[feature_cols[0]])
y = np.array(data[target])
X, y = shuffle(X, y, random_state=1)
print('X',X)
print(X.shape) # Education
print(y.shape) # Crime Target


#  Question One create scatterplot
fig = plt.scatter(data['Education'], data['Crime'])

# add plot labels
plt.xlabel('Education')
plt.ylabel('Crime')
plt.show()

fig = plt.scatter(data['Police'], data['Crime'])

# add plot labels
plt.xlabel('Police')
plt.ylabel('Crime')
plt.show()

fig = plt.scatter(data['Police'], data['Crime'])

# add plot labels
plt.xlabel('Income')
plt.ylabel('Crime')
plt.show()

fig = plt.scatter(data['Inequality'], data['Crime'])

# add plot labels
plt.xlabel('Income')
plt.ylabel('Crime')
plt.show()
