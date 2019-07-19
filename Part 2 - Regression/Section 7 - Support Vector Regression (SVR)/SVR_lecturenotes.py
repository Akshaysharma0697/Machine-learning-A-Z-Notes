# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 19:47:56 2019

@author: Akshay
"""
#second non linear regression model

#polynomial regression template

#Polynomial Regression

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 ##importing datasets
dataset = pd.read_csv('Position_Salaries.csv') #variable dataset, now use pandas to import the dataset
X = dataset.iloc[:,1:2].values #taking all the lines by the first : , now after this [:-1] means taking all columns except the last one i.e purchased column
Y =dataset.iloc[:,2].values #dependent variable vector {taking all the lines of the column 3rd}

##splitting the dataset into the training set and Test set#as we need to have the most of the info to predict the best results so we cant lose any data
"""from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y ,test_size=0.2, random_state= 0
"""
# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()#objects
sc_Y = StandardScaler()#objects that are going to scale x and y
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
X = sc_X.fit_transform(X)#fitting and transforming these to the scale these x and y 
Y = sc_Y.fit_transform(Y)


#fitting the SVR to the dataset
#create our regressor
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,Y)


#predicting the reults with the polynomial regression
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))#we need to transform it as X AND Y were transformed to  fit by feature scaling the  SVR and this is not transformed 
#so we use the sc_x object to , transform it , now making it in array using np library and method array
#we also need to use the inverse transform to get the original scale salary
#if we execute the above line we get the scaled salary
#so we need to inverse sc_y to get the original scale prediction

#visualising the SVR results
plt.scatter(X , Y , color='red')
plt.plot(X, regressor.predict(X),color='blue')#here the lin_rag_2 is still the object of the linear regression class so we need to add something to make predictions of the poly regression class.
#so we add this poly_rag.fit and not x_poly as we need to make it general for any new matrix of features x 
plt.title("Truth or Bluff(SVR)")
plt.xlabel("position label")
plt.ylabel("Salary")
plt.show()
#there is something missing as we obtain a wrong output . We forgot to add feature scaling
#so after adding the feature scaling we see the change

















