# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:46:53 2020

@author: shind
"""

import numpy as np
import scipy.io
mat = scipy.io.loadmat("/mnist")
X_digits = mat['data'].T
y_digits = mat['label'][0].T