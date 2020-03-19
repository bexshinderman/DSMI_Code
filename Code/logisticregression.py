# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:46:53 2020

@author: shind
"""

 
import math

#Change these parameters and observe their impact on the sigmoid model
theta0=100
theta1=10

def sigmoid(x,t0,t1):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-(t0+t1*item))))
    return a

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10., 10., 0.2)
sig = sigmoid(x,theta0,theta1) #anything above 0.5 a 1, below a 0
plt.plot(x,sig)
plt.show()

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.7,random_state=1)#generate dataset consisting of two Gaussian clusters
print("X shape", X.shape)
print("y", y)
print(X)

import pylab as plt
plt.prism()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

from sklearn.linear_model import LogisticRegression
logRegModel = LogisticRegression()

#split data set
X_train = X[:50] 
y_train = y[:50]
X_test = X[50:]
y_test = y[50:]

plt.prism()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c='black', marker='^')

#fit model
logRegModel.fit(X_train, y_train)

print("intercept (theta 0)",logRegModel.intercept_) #theta_0
print("coefficiant (theta 1 and 2)", logRegModel.coef_) #theta_1 and #theta_2

import numpy as np
def plot_decision_boundary(clf, X):
    w = clf.coef_.ravel()
    a = -w[0] / w[1]
    xx = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]))
    yy = a * xx - clf.intercept_ / w[1]
    plt.plot(xx, yy)
    plt.xticks(())
    plt.yticks(())
y_pred_train = logRegModel.predict(X_train)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train)
plot_decision_boundary(logRegModel, X)

print("Accuracy on training set:", logRegModel.score(X_train, y_train))
print('*******************************************************************************')
import numpy as np
import scipy.io
mat = scipy.io.loadmat("./mnist")
X_digits = mat['data'].T
y_digits = mat['label'][0].T
print(mat['label'][0])
print(X_digits.shape)
print("Unique entries of y_digits:", np.unique(y_digits)) #The classes in y

import pylab as plt
print("Class of first element in our data set: ", y_digits[0])
plt.rc("image", cmap="binary")
print("Data shape of first row of X: ", X_digits[0].shape)
print("First row of X: " +str(list(X_digits[0])))
print("Transforming the first row of X into a 2 dimensional representation:")
plt.matshow(X_digits[0].reshape(28, 28)) # we reshape the 784 elements row into a 28x28 matrix
ax = plt.gca()
ax.grid(False)

# split between ones and zero classes
zeros = X_digits[y_digits==0]  # select all the rows of X where y (target value) is zero (i.e. the zero digits)
ones = X_digits[y_digits==1]   # select all the rows of X where y is one (i.e. the one digits)
print("zeros.shape: ", zeros.shape) # print the number of instances of class 0
print("ones.shape: ", ones.shape) # print the number of instances of class 1

plt.rc("image", cmap="binary")
plt.matshow(zeros[0].reshape(28, 28)) 
ax = plt.gca()
ax.grid(False)
plt.matshow(ones[0].reshape(28, 28)) 
ax = plt.gca()
ax.grid(False)

#generate new dataset
X_new = np.vstack([zeros, ones])  # this "stacks" zeros and ones vertically
print("X_new.shape: ", X_new.shape)
y_new = np.hstack([np.repeat(0, zeros.shape[0]), np.repeat(1, ones.shape[0])])
print("y_new.shape: ", y_new.shape)

#shuffe new data into traing and test 
from sklearn.utils import shuffle
X_new, y_new = shuffle(X_new, y_new)
X_mnist_train = X_new[:5000]
y_mnist_train = y_new[:5000]
X_mnist_test = X_new[5000:]
y_mnist_test = y_new[5000:]

#learn logistic regression model
from sklearn.linear_model import LogisticRegression
logRegModel = LogisticRegression(solver='lbfgs')
logRegModel.fit(X_mnist_train, y_mnist_train)

#visualise classes
plt.matshow(logRegModel.coef_.reshape(28, 28))
plt.colorbar()
ax = plt.gca()
ax.grid(False)

print("Accuracy training set 2:", logRegModel.score(X_mnist_train, y_mnist_train))
print("Accuracy test set 2:", logRegModel.score(X_mnist_test, y_mnist_test))

