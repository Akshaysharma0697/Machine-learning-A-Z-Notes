# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 17:41:21 2019

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

##splitting the dataset into the training set and Test set
from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y ,test_size=0.2, random_state= 0)

#****feature scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)""" 










