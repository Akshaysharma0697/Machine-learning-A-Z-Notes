# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:42:43 2019

@author: Akshay
"""
#apriori (ARL) :- based on people who bought x also bought y 
 #step1:-set a minimum support and confidence
 #step2:- take all the subsets in transactions having higher support than minimum support
 #step3:- take all the rules of these subsets having higher confidence than minimum confidence
 #step4:- sort the rules by decreasing lift
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)#python thought that the first line contain the title so we make header none; 
#we need to import the dataset in a specific way ; apriori  is aspecting a list of list
transactions =[]#a list
for i in range(0,7501):#used for going through all the transactions
    transactions.append((str(dataset.values[i,j]))for j in range(0,20))#for adding j we use this other for loop i.e all the columns which are 20
    #we set all the products in string format  as it is aspecting it in strings ; if we do not use str then they will simply have the names but by using it the names will be in quotes
    
#training the apriori on the dataset 
from apyori.py import apriori#using the library from our folder directly
##it takes transactions as input and gives rules as output
rules =apriori(transactions ,min_support=0.003  ,min_confidence = 0.2 , min_lift = 3 ,  min_length= 2)#confidence should be low so that no two objects get attached together

#visualising the results:-
results =list(rules)