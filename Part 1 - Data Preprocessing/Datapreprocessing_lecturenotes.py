# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:57:10 2019

@author: Akshay
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 ##importing datasets
dataset = pd.read_csv('Data.csv') #variable dataset, now use pandas to import the dataset
X = dataset.iloc[:,:-1].values #taking all the lines by the first : , now after this [:-1] means taking all columns except the last one i.e purchased column
Y =dataset.iloc[:,3].values #dependent variable vector {taking all the lines of the column 3rd}


#Imp*******taking care of missing data*******

from sklearn.preprocessing import Imputer #taking library use for our work i.e sklearn importing imputer(preprocessing library)
imputer = Imputer(missing_values= 'NaN' , strategy='mean', axis =0) #object of class Imputer
imputer = imputer.fit(X[:,1:3]) #fitting this imputer object to our matrix of features X . Now we select those columns that has missing data in them i.e 1:3(not 2)
X[:,1:3]= imputer.transform(X[:,1:3]) # replacing the missing data of x by mean of all values by using **Transform method*

#<<encoding categorial data >> in our dataset the 2 categorial variables are country and purchased as they contain categories so we need to encode these into values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
labelencoder_X = LabelEncoder() #object of class
X[:,0]=labelencoder_X.fit_transform(X[:,0]) #we use fit transform method on the column country and this returns the country column encoded
#this creates a problem for the ML as it thinks that one country is of higher priority than other as based on their values
#so for solving this we create dummy variable columns for each of the categories 
#so for 3 categories we have 3 columns and each hold 0 or 1 value
#so we use onehotencoder class for this solution

onehotencoder = OneHotEncoder(categorical_features = [0])
x= onehotencoder.fit_transform(X).toarray() # same feature used to modify X[0]
#now for the dependent variable we use labelencoder method only as the ML will know there in no relation btn the 2 as it is dependent variable
labelencoder_Y = LabelEncoder() 
Y=labelencoder_Y.fit_transform(Y) 

##splitting the dataset into the training set and Test set
from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y ,test_size=0.2, random_state= 0)

#****feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test) 

