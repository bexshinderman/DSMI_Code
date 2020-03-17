# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:44:07 2020

@author: shind
"""

import matplotlib
matplotlib.use("TKAgg")  # use this instead in your program if you want to use Tk as your graphics backend. (it opens a separate  window for each figure)

import matplotlib.pyplot as plt
plt.plot([1, 2, 4, 9, 5, 3])
import numpy as np
x = np.linspace(-2, 2, 500)
y = x**2

plt.plot(x, y)
plt.title("Square function")
plt.xlabel("x")
plt.ylabel("y = x**2")
plt.grid(True)
plt.show()