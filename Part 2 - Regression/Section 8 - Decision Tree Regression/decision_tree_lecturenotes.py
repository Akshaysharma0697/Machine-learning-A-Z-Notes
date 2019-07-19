# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:07:06 2019

@author: Akshay
"""
##CART - 1.CLASSIFICATION TREES 2. REGRESSION TREES
#non continuous regression model
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
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting the regression model to the dataset
#create our regressor
from sklearn.tree import  DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)


#predicting the reults with the polynomial regression
#Y_pred = regressor.predict(6.5)


#for higher resolution and smoother curve
#visualising the Decision  regression results
X_grid=np.arange(min(X), max(X),0.01)
X_grid =X_grid.reshape((len(X_grid),1))
plt.scatter(X , Y , color='red')
plt.plot(X_grid, regressor.predict(X_grid),color='blue')#here the lin_rag_2 is still the object of the linear regression class so we need to add something to make predictions of the poly regression class.
#so we add this poly_rag.fit and not x_poly as we need to make it general for any new matrix of features x 
plt.title("Truth or Bluff(Decision Tree regression)")
plt.xlabel("position label")
plt.ylabel("Salary")
plt.show()
#not the real graph without the feature scaling


















