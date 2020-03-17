# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:58:17 2020

@author: shind
"""

import numpy as np
data = np.loadtxt("artifical_lin.txt")
print(data)
print("*********************")
X = data[:, :-1] # select all the rows [:, in the data object and all the columns except the last one ,:-1
y = data[:, -1] # select all the rows in the last column of the data object
print(X[:10, :])
print(y[:10])


#shuffle

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=1)
print(X.shape)  # 
print(y.shape) #

''' train_set_size = int(X.shape[0] / 2 )
print("train set size = half of x data", train_set_size)
X_train = X[:train_set_size, :]  # selects first train_set_size rows (examples) for train set
X_test = X[train_set_size:, :]   # selects from row train_set_size until the last one for test set
print(X_train.shape)
print(X_test.shape)

#split Y (cliniciancs prediction)  into ttraining data 
y_train = y[:train_set_size]
y_test = y[train_set_size:]    # selects from row 250 until the last one for test set
print(y_train.shape)
print(y_test.shape) '''

#alternately, one may use the following to achieve the same outcome:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)

#Remember, you do need to include the previous line if you're working in the Spyder interpreter
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(X_train[:, 0], X_train[:, 1], y_train[:],c='r')
ax.view_init(29, -15) #specifying the position in space from which we look at the data (i.e. perspective)

plt.xlabel(r'Voice feature 1 ($x_1$)',size=13)
plt.ylabel(r'Voice feature 2 ($x_2$)',size=13)
ax.set_zlabel('Clinician score of the patient symptoms')
plt.show()

print("**************************************")
#create linear regression model
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
print(model.intercept_)# theta0
print(model.coef_) #theta1 and theta2
print("model score - closest to 1 is perfect",model.score(X_test, y_test))

# The mean square error
print("Mean squared error on training set: ", np.mean((model.predict(X_train) - y_train) ** 2))
print("Mean squared error on test set: ", np.mean((model.predict(X_test) - y_test) ** 2))
print("**************************************")
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter3D(X_train[:, 0], X_train[:, 1], y_train[:],c='r')    # plots 3d points, 500 is number of points which are visualized

# here we create plane which we want to plot, using the train data and predictions (you don't need to understand it)
range_x = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), num=10)
range_y = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), num=10)
xx, yy = np.meshgrid(range_x, range_y)
zz = np.vstack([xx.ravel(), yy.ravel()]).T
pred = model.predict(zz)
pred = pred.reshape(10, 10)
plt.xlabel('Voice feature 1')
plt.ylabel('Voice feature 2')
ax.set_zlabel('Clinician score of the patient symptoms')
ax.plot_surface(xx, yy, pred, alpha=.3,color='r')  # plots the plane
ax.view_init(6,-20)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(X_test[:, 0], X_test[:, 1], y_test[:],c='b')    # plots 3d points 500 is number of points which are visualized

# here we create plane which we want to plot, using the train data and predictions (you don't need to understand the code)
range_x = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), num=10)
range_y = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), num=10)
xx, yy = np.meshgrid(range_x, range_y)
zz = np.vstack([xx.ravel(), yy.ravel()]).T
pred = model.predict(zz)
pred = pred.reshape(10, 10)
plt.xlabel('Voice feature 1')
plt.ylabel('Voice feature 2')
ax.set_zlabel('Clinician score of the patient symptoms')
ax.plot_surface(xx, yy, pred, alpha=.3,color='y')  # plots the plane
ax.view_init(20,-20)
plt.show()
print("**************************************")
#To use our model created above we can simply plug numbers in, say our x and y values are the following, we can get a representation of how severe the parkinsons is, 0 being not and 3 being severe. x and y are voice factors

x_predicted=model.predict([[0.4,0.9]]) 
print("prediction =", x_predicted)
print("**************************************")
#polynomial
import numpy as np
import matplotlib.pyplot as plt
X = np.array([0,2,4,6,7])
y = np.array([1,3,5,3,1])

plt.scatter(X,y)
print("**************************************")

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_transformed  = poly.fit_transform(X.reshape(-1,1))
print(X_transformed)

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_transformed,y)

X_test = np.array([1,3,5,8]) #New data points that our model didn't see during training
X_test_tranformed = poly.transform(X_test.reshape(-1,1))
y_predicted = model.predict(X_test_tranformed)
print("y prediced", y_predicted)

from sklearn import preprocessing

import numpy as np
np.set_printoptions(suppress=True) #line to prettify the output of numpy arrays so they are not ridiculously long

X = np.array([[ 1000, -1,  0.02],

              [ 1500,  2,  0.07],

              [ 1290,  1.5, -0.03]])

X_scaled = preprocessing.scale(X)

print("x  scaled", X_scaled  )
