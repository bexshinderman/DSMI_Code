# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 00:30:07 2020

@author: shind
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# read dataset
df = pd.read_csv('./data/winequality-red.csv',sep=';')
print(df.keys())
print(df.chlorides[1])


# create histogram
bin_edges = np.arange(0, df['residual sugar'].max() + 1, 1)
fig = plt.hist(df['residual sugar'], bins=bin_edges)

# add plot labels
plt.xlabel('count')
plt.ylabel('residual sugar')
plt.show()

# create scatterplot
fig = plt.scatter(df['pH'], df['residual sugar'])

# add plot labels
plt.xlabel('pH')
plt.ylabel('residual sugar')
plt.show()

plt.boxplot(df['alcohol'])

plt.ylim([8, 16])
plt.ylabel('alcohol')

fig = plt.gca()
fig.axes.get_xaxis().set_ticks([])
plt.show()

#gen random num
print("random num",np.random.uniform(0, 10))

#create array of random nums 100 nums of 1-10
observations = np.random.uniform(0, 10, 100)
print(observations)
fig = plt.hist(observations, bins=bin_edges)