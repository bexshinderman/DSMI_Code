# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:01:26 2020

@author: shind
"""
'''Questions:
   - Why is mdev not a featurename? what feature name can I check house prices so I can check accuracy of my model?
    - coeefficiant of determination & mean squared which method is right
    '''
from sklearn.datasets import load_boston
boston = load_boston()
print (boston.keys())
print (boston.data.shape)
print (boston.feature_names)

#SHUFFLE
X = boston.data
print(X)
from sklearn.utils import shuffle

X = boston.data 
y = boston.target

print("shape of x", X.shape)   
print("shape of y", y.shape)  

#shuffle
X, y = shuffle(X, y, random_state=1)


#split into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5) #50% of the data in training set and 50% of the data in test set

#fit the model to the data
from sklearn import linear_model
model = linear_model.LinearRegression()

fit = model.fit(X_train,y_train)

print("coeffecient 0 to 13", model.coef_)
print("intercept", model.intercept_)

#effectiveness of model
print("score",model.score(X_test, y_test))



from sklearn.metrics import mean_squared_error
error = mean_squared_error(X_train, X_test)
print("error x", error)
error = mean_squared_error(y_train, y_test)
print("error y", error)

import numpy as np
#mean squared error on train and test ,coeefient determination of the sets 
print("Training error: ", np.mean((model.predict(X_train) - y_train) ** 2))
print("Test     error: ", np.mean((model.predict(X_test) - y_test) ** 2))

#coefficient of determination
''' from sklearn.metrics import r2_score
r2_score_X = r2_score(X_train, X_test)
print("r2 score for x" ,r2_score_X)

r2_score_y = r2_score(y_train, y_test)
print("r2 score for y" ,r2_score_y) '''


new_house = (X_test[0]+X_test[1])/2
prediction = model.predict(new_house.reshape(1, -2))
print("prediction", prediction)

