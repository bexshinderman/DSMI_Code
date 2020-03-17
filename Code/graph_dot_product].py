# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 01:35:16 2020

@author: shind
"""

import numpy as np
v = np.array([1,2,3])
print(v)
size = v.size
print(size)


import matplotlib.pyplot as plt
u = np.array([2, 5])
v = np.array([3, 1])

def plot_vector2d(vector2d, origin=[0, 0], **options):
    return plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1],
              head_width=0.2, head_length=0.3, length_includes_head=True,
              **options)
    
plot_vector2d(u, color="r")
plot_vector2d(v, color="b")
plt.axis([0, 9, 0, 6])
plt.grid()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

a = np.array([1, 2, 30])
b = np.array([4, 1, 3])

x, y, z = zip(a,b)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0,0,0,x, y, z) #0,0,0 tail of vectors, x,y,z tip of vectors
ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([0, 30])
plt.show()

def vector_norm(vector):
    squares = [element**2 for element in vector]
    return sum(squares)**0.5

u = np.array([2, 5])

print("||u|| =",end=' ')
print(vector_norm(u))

#or

import numpy.linalg as LA
print(LA.norm(u))

print(u+v)

def dot_product(v1, v2):
    return sum(v1i * v2i for v1i, v2i in zip(v1, v2))
print("dot product",  dot_product(u, v))
dot_product(u, v)