# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:37:10 2020

@author: Bex.0
"""

import tensorflow as tf
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#The data must be preprocessed before training the network. We can scale these values to a range of 0 to 1 before feeding them to the neural
#network model. To do that we simply divide by the maximum number in the data
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig=plt.figure(figsize=(7,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]],fontsize=7)
fig.subplots_adjust(hspace=0.5)    
plt.show()