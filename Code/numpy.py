# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:21:17 2020

@author: shind
"""

from sklearn.datasets import load_iris
iris = load_iris()
keys = iris.keys()
print(keys)

n_samples, n_features = iris.data.shape
print(n_samples)
print(n_features)
print(iris.data)
print(iris.data.shape)
print(iris.target.shape)
print(iris.target)
print(iris.target_names)

import numpy as np
array=np.zeros(5)
print(array)
print(np.zeros((3,4)))


import matplotlib.image as mpimg
from matplotlib import pyplot as plt

imgArray=mpimg.imread('./images/ap.png')
plt.imshow(imgArray, interpolation='nearest')
plt.show()

print(imgArray)

print(np.ones((3,4)))

print(np.full((3,4), 8))

import matplotlib.pyplot as plt
plt.hist(np.random.rand(100000), normed=True, bins=100, histtype="step", color="blue", label="rand: Uniform")
plt.hist(np.random.randn(100000), normed=True, bins=100, histtype="step", color="red", label="randn: Normal")
plt.axis([-2.5, 2.5, 0, 1.1])
plt.legend(loc = "upper left")
plt.title("Random distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

c = np.arange(1, 5)
print(c.dtype, c)
#to specify data typee

print(c.itemsize)


d = np.arange(1, 5, dtype=np.complex64)
print(d.dtype, d)

g = np.arange(1,24)
print(g)
print("Rank:", g.ndim)

a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print(g.prod())

print(a)
print(f"square",np.square(a))

for func in (np.abs, np.sqrt, np.exp, np.log, np.sign, np.ceil, np.modf, np.isnan, np.cos):
    print("\n", func.__name__)
    print(func(a))
    
    
for i in a.flat:
    print("Item:", i)