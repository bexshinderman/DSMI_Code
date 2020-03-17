# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 00:13:45 2020

@author: shind
"""

print(__doc__)

# Modified from Mathieu Blondel and Jake Vanderplas
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)


# generate points used to plot
x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

plt.figure(figsize=(15, 10))
# plt.figure(figsize=(40, 20), dpi=100)
plt.plot(x_plot, f(x_plot), label="ground truth: some unknown rule governing the phenomena you are studying",color='k')

plt.scatter(x, y, label="training points: your X", color='k')


for degree in [1, 3, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    if degree == 1:
        plt.plot(x_plot, y_plot, label=r'Polynomial of degree %d: $y=\theta_0+\theta_1x$' % degree)
    elif degree == 3:
        plt.plot(x_plot, y_plot, label=r'Polynomial of degree %d: $y=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3$' % degree)
    elif degree == 5:
        plt.plot(x_plot, y_plot, label=r'Polynomial of degree %d: $y=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4+\theta_5x^5$' % degree)

plt.legend(loc='lower left',fontsize=14)
plt.show();