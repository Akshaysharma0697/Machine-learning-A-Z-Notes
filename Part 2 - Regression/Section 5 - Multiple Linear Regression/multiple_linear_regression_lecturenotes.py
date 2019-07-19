# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:15:54 2019

@author: Akshay
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 ##importing datasets
dataset = pd.read_csv('50_Startups.csv') #variable dataset, now use pandas to import the dataset
X = dataset.iloc[:,:-1].values #taking all the lines by the first : , now after this [:-1] means taking all columns except the last one i.e purchased column
Y =dataset.iloc[:,4].values #dependent variable vector {taking all the lines of the column 3rd}


#<<encoding categorial data >> in our dataset the 2 categorial variables are country and purchased as they contain categories so we need to encode these into values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
labelencoder_X = LabelEncoder() #object of class
X[:,3]=labelencoder_X.fit_transform(X[:,3]) #we use fit transform method on the column country and this returns the country column encoded
#this creates a problem for the ML as it thinks that one country is of higher priority than other as based on their values
#so for solving this we create dummy variable columns for each of the categories 
#so for 3 categories we have 3 columns and each hold 0 or 1 value
#so we use onehotencoder class for this solution
onehotencoder = OneHotEncoder(categorical_features = [3])
x= onehotencoder.fit_transform(X).toarray() # same feature used to modify X[0]
 
#avoiding the dummy variable trap i.e removing one dummy variable 
X = X[:,1:]

##splitting the dataset into the training set and Test set
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y ,test_size=0.2, random_state= 0)

#fitting multiple linear regression to the training sets
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()#object of class linear regression
regressor.fit(X_train,Y_train)

#predicitng the test set reults // not plotting graph as graph for 5 variables is hard
Y_pred = regressor.predict(X_test)#vector of prediction now using predcit method to predict the observations of the test set
#the values of x test predicted by our model 

#Building the optimal model using Backward Elimination:-
import statsmodels.formula.api as sm #in this library it does not include the constant b0 in the multiple regression eq
#so we need to add a column of 1's to include b0x0 in the matrix of features
#******#X = np.append(arr= X, values =np.ones((50,1)).astype(int), axis=1 )
#to prevent the datatype error we need to convert this values into int
#now we need this column to be at the start of X as this will do add the column in last 
#so we change the values and arr 
X = np.append(arr= np.ones((50,1)).astype(int) , values =X, axis=1 )#50 lines and 1 colu()mn appended using np.ones we create a column of 50 ones(1's) 
#now the matrix of featues x is added to this 1 column of 50 1's . so this will appear as the matrix of features x at the start

#now we start the backward elimination method:-#taking all the independent variables at first and then removing them one by one which are not statistically significant
X_opt = X[: ,[0,1,2,3,4,5]] #new optical matrix of features
####step1:- select a significance level to stay in the model  (e.g SL=0.05)
####step2:- to fit the full model with all possible predictors(not the regressor as it is a new library)
#creating a new regressor of the new class stats model lib
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()#fitting the full model i.e step2 done

####step3:- consider the predictor with the highest p value . if p>SL goto step4 otherwise Optimal results are achieved
#now we use the summary function for step3 to calculate the p values of the independent variables
#the p values is a prbability , the lower the p value the more significant the independent varibale will be wrt the dependent variale
regressor_OLS.summary()
####step4: remove the predictor as p>0.05
#i.e here column 2 needs to be removed
#so changing the index of 2 column i.eremoving 2 in X_opt = X[: , [0,1,2,3,4,5]]
X_opt = X[: , [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
####step5:-fit the model without this variable* after this we go back to step 3
###repeat step 3,4,5 until the highest p value is not more than the significant value 
X_opt = X[: , [0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,3,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,3]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
#only one independent variable r&d column 
#therefore a very powerful predictor







