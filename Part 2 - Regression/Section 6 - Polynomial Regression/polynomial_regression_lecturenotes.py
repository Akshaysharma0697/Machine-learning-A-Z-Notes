# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:25:08 2019

@author: Akshay
"""
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
#fitting the linear regression to the dataset
#we are creating the linear regression model as a reference so that we can compare the linear with the polynomial regression
from sklearn.linear_model import LinearRegression
lin_rag = LinearRegression()#object of the linearregression class
lin_rag.fit(X,Y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_rag= PolynomialFeatures(degree =  4) # this polyrag is going to transform our matrix of features x into a new matrix of features x_poly containing x . x^2 , x^3 and so on
X_poly = poly_rag.fit_transform(X)#fitting and transforming the x_poly
lin_rag_2= LinearRegression()#including the fit into our multiple linear regression model
lin_rag_2.fit(X_poly,Y)#fitting x_poly and y to this new object

#visualising the linear regression results
plt.scatter(X , Y , color='red')
plt.plot(X, lin_rag.predict(X),color='blue')#vector containing the y coordinate of the prediction points
#therefore using lin_rag object with the predict method
plt.title("Truth or Bluff(Linear regression)")
plt.xlabel("position label")
plt.ylabel("Salary")
plt.show()
#the red points on graph are the real obseravtion points
#the linear regression model we will always have a straight line in 2d 

#visualising the polynomial regression results
X_grid = np.arange(min(X), max(X),0.1)#we can have a better plot by having all the levels with a step size 0.1
#min(x) is the lower bound of curve and max(x) is the upper bound of the curve with step size 0.1
#this creates all the imaginary points in the curve from 1 to 10 with step size = 0.1 so , 90 points are there , thus to get a smoother curve and better curve
#this gives us a vector but we need a matrix so we add this reshape method to it.
X_grid = X_grid.reshape(len(X_grid),1)#reshapes x_grid to the matrix of 1 column and 90 lines 
plt.scatter(X , Y , color='red')
plt.plot(X_grid, lin_rag_2.predict(poly_rag.fit_transform(X_grid)),color='blue')#here the lin_rag_2 is still the object of the linear regression class so we need to add something to make predictions of the poly regression class.
#so we add this poly_rag.fit and not x_poly as we need to make it general for any new matrix of features x 
plt.title("Truth or Bluff(Polynomial regression)")
plt.xlabel("position label")
plt.ylabel("Salary")
plt.show()
#compare both the models and make observations

#predicting the new result with linear regression
lin_rag.predict(6.5)#this level 6.5 is here so as to predict only this level salary and not the whole x matrix salary


#predicting the new results with the polynomial regression model
lin_rag_2.predict(poly_rag.fit_transform(6.5))
#so this future employee said it's previous salary was 160k and the linear regression model predicted said the salary was 330k and the polynomial regression said it was 158k 
#the verdict is truth , so the employee is honest and thus a valuable employee in the team



















