# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:07:03 2020

@author: Bex.0
"""

import pandas as pd

data = pd.read_csv('crime.csv', index_col=0)
print(data.head())
import numpy as np
from numpy import cov
from sklearn.utils import shuffle
feature_cols = ['Education','Police','Income','Inequality']
target = ['Crime']
X = np.array(data[feature_cols])
y = np.array(data[target])
X, y = shuffle(X, y, random_state=1)

import matplotlib.pyplot as plt
import seaborn as sns
#  Question One create scatterplot
fig = plt.scatter(data['Education'], data['Crime'])

# add plot labels
plt.xlabel('Education')
plt.ylabel('Crime')
plt.show()

fig = plt.scatter(data['Police'], data['Crime'], c=['red'])

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
plt.xlabel('Inequality')
plt.ylabel('Crime')
plt.show()




feature_col = data.keys()
print(feature_col[1])
print (np.array(data['Education']))
X = np.array(data['Education'])
y = np.array(data['Crime'])
fig = plt.scatter(X, y)

# add plot labels
plt.xlabel('Education')
plt.ylabel('Crime')
plt.show()



#this ?
a = np.array(data[data.columns[0]])
b = np.array(data[data.columns[4]])
X = a + b
print(X)
print("covariance: ", np.cov(a,b))
print("correlation: ", np.corrcoef(a,b.T))

#or this?
print (feature_col[4])

#education vs crime
a = np.array(data[data.columns[0]])
b = np.array(data[data.columns[4]])
print("***********")
print("A", a)
print("***********")
print("B", b)

print("covariance matrix: ", np.cov(a,b))
corr = np.corrcoef(a, b)
print("correlation matrix: ", corr)
print("pearsons correlation is : ",corr[0,1])

	
sd_a = np.std(a)
sd_b = np.std(b)
print(sd_b)

#education vs crime 2
from scipy.stats import pearsonr
a = np.array(data[data.columns[0]])
b = np.array(data[data.columns[4]])
X = np.append(a,b)

print("***********")
print("A", a)
print("***********")
print("B", b)
print("***********")
print("X", X)
print("covariance: ", np.cov(a,b))
print("correlation: ", np.corrcoef(a,b))
corr = pearsonr(a, b)
print("pearsons correlation", corr)



#police vs crime
a = np.array(data[data.columns[1]])
b = np.array(data[data.columns[4]])

print("covariance matrix: ", np.cov(a,b))
corr = np.corrcoef(a, b)
print("correlation matrix: ", corr)
print("pearsons correlation is : ",corr[0,1])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)

from sklearn import linear_model
model = linear_model.LinearRegression()

model.fit(X_train, y_train);
print(model.intercept_)
print(model.coef_) 

print("Test model score ",model.score(X_test, y_test))
print("Training model score ",model.score(X_train, y_train))

print("City2 has the highest crime rate")
city1 = np.array([[10, 5, 6000, 16]])
crime = model.predict(city1)
print("y predicted for city 1", crime)

city2 = np.array([[8, 11, 4500, 25]])
crime = model.predict(city2)
print("y predicted for city 2", crime)

city3 = np.array([[6, 8, 3780, 17]])
crime = model.predict(city3)
print("y predicted for city 3", crime)

city4 = np.array([[12, 6, 5634, 22]])
crime = model.predict(city4)
print("y predicted for city 4", crime)

from sklearn import linear_model
from numpy.linalg import inv
model = linear_model.LinearRegression(fit_intercept = False)

model.fit(X, y);
print(model.coef_)

#y = y.reshape((-1,1))
print(X.shape)
coeffs = inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
print(coeffs)

print("************run seperately*************************")
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
print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(data)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)
print('*****************************')
print(data)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_train, y_train)
print(score)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
score = neigh.score(X_train, y_train)
print(score)




