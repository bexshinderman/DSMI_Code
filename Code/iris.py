# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:03:39 2020

@author: shind
"""
import seaborn as sns
import pandas as pd
from IPython.core.display import Image, display
display(Image(filename='images/iris_setosa.jpg'))
print("Iris Setosa\n")

display(Image(filename='images/iris_versicolor.jpg'))
print("Iris Versicolor\n")

display(Image(filename='images/iris_virginica.jpg'))
print("Iris Virginica")

from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()
n_samples, n_features = iris.data.shape
print (n_samples, n_features) # this is the number of samples and features in the X feature matrix
print(iris.data[0])
print("the last instance is",iris.data[-1])
print(iris.target.shape)
# first instance and corresponding y target
print(".data",iris.data[0])
print("target",iris.target[0]) #this represents type of iris
print(iris.target_names)
print(iris.target)

import numpy as np
import matplotlib.pyplot as plt

#plots petal and sepal length by type
def plot_iris_projection(x_index, y_index):
    # this formatter will label the colorbar with the correct target names
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plt.scatter(iris.data[:, x_index], iris.data[:, y_index],
                c=iris.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])

#to better represent the data start both axis at 0. this shows the larger the 
x_index=1
y_index=1
plot_iris_projection(x_index,y_index)

#how to close graph code?

n_samples, n_features = iris.data.shape
print("this is the number  of samples",iris.data.shape[0])
print("this is the number  of features",iris.data.shape[1])
print("spits everything",iris.data)

from sklearn import datasets
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.keys())
n_samples, n_features = digits.data.shape
print(n_samples, n_features)
#or
print(digits.data.shape[0], digits.data.shape[1])
print(digits.data[0])
print(digits.target)
print(digits.data.shape) #data is a flattened  version 
print(digits.images.shape) # 8*8 = 64, the image is the matriz representation of an image

# set up the figure
import pylab as plt
fig = plt.figure(figsize=(3, 3))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
import numpy as np
print(np.zeros(5))
print(np.zeros((3,4)))