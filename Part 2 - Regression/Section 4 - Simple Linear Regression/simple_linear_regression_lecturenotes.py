# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 18:31:03 2019

@author: Akshay
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 ##importing datasets
dataset = pd.read_csv('Salary_Data.csv') #variable dataset, now use pandas to import the dataset
X = dataset.iloc[:,:-1].values #taking all the lines by the first : , now after this [:-1] means taking all columns except the last one i.e purchased column
Y =dataset.iloc[:,1].values #dependent variable vector {taking all the lines of the column 3rd}

##splitting the dataset into the training set and Test set
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y ,test_size=1/3, random_state= 0)

#****feature scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)""" 

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()#object of the linearregression class
regressor.fit(X_train,Y_train)

#predicting the test set results
Y_pred = regressor.predict(X_test) # creating a vector for storing the values predicted by our Ml model on X_test set i.e predicted salaries

#visualising the training set results
plt.scatter(X_train,Y_train,color='red')#plotting the observation points of the predicted values(points) by our model
plt.plot(X_train,regressor.predict(X_train),color='blue')#plotting our regression line i.e x coordinate and the y coordinate i.e prediction of number of experiences of the x_train set
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()#to plot the ending of the graph and thus we are ready to plot
 
 
#visualising the test set results
plt.scatter(X_test,Y_test,color='red')#plotting the observation points of the predicted values by our model
plt.plot(X_train,regressor.predict(X_train),color='blue')#we dont change here the x_train by x_Test as we will build some new points here 
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()#to plot the ending of the graph and thus we are ready to plot