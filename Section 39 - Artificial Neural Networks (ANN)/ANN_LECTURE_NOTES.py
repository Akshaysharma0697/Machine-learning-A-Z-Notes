# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:06:33 2019

@author: Akshay
"""
#NEURON:-
"""
.The main purpose of deep learning is to mimic the human brain
how to recreate a neuron in a machine:-
dendrites - receiver for the neuron
axon - the transmitter for the neuron
signals travel down from the axon and then travels to the next neuron
synapse:-the connecting part of one axon to the next neuron
{X1,X2,X3 }(input values in yellow color)----synapse----->neuron------->output signal(in red color)
in input layer ; x1,x2,x3 are independent variables for a single feature  
output value can be
            - continuos(price)
            - binary(will exist yes or no)
            - categorical(multiple output values)
.each input lines will have weights associated to them .
*********what happenns inside the neuron-
1. phi(i=1 to m -E WiXi) Here E is the summation of all wixi values 


*************Activation function:-***********
the values that is transferred from the neuron to the output signal
a.threshhold function:- 
  - if the value is <0 then it passes 0
  - it it is >0 then it passes a 1
b.sigmoid function:-phi(x)= 1/ 1+e^-x
    - it is a smooth curve
c.rectifier function:- 
  - phi(x)= max(x,0)
d.hyperbolic function(tanh):-
  phi(x)= (1-e^-2x)/1+e^2x
  
*****How do neural network works?******
e.g taking a property valuation of a house
{x1,x2,x3,x4}={area(feet2), bedrooms, distance to city, age} #input values
having {w1,w2,w3,w4}
output layer---> price= w1*x1+w2*x2...
Hiddden layer gives us that extra power of accuracy:-
 HL1(neuron1):- is looking for places that are close to the city and having a large area so only looking for the properties 
       like area and distance to city ..
 HL2:- area , bedroom, age of property 
 and so on
 together they can predict the value of property
 
 ***** How do NN learn?? ******
-backpropagation is needed 
-stochastic GRADIENT DESCENT:-
 it does not require a cos function to be a convex curve with a gobal minimum
 normal gradient descent(batch):- in this all the rows are taken and plug them into pur neural network
                           and then calculate our cos function using the formula given on the right 
stochastic gradient descent :- we take a single row , we run our neural network then we adjust the weights , then we take the next one and so ono
                              -it helps us avoid the local minimums rather than the global mimnimum's
mini batch gradiet descent :- wee take a bunch of rows and then we update the weights afterwards and so on ///
 
forward propagation:- in gradient descent we proceed through hidden layers and then to get y hats and compare those to actual values then we calculate the errors which are then 
backpropagated throught the network in the opposite direction which allows us to adjust weights and thus training the network

***********backpropagation***************                       
process of adjusting all the weights at the same time  

******** Training the ANN with stochastic gradient descent********
step1:- randomly initialise the weights to small numbers close to 0 ( but not 0)
step2:- input the first observation of your dataset in the input layer , each feature in one input node
step3:- forward propagation , from left to right , the neurons are activated in a way that the impact of each neurons activation
        is limited by the weights . propagate the activations until getting the predicted result y
step4:- compare the predicted result to the actual result , measure the genereated error
step5:- backpropagation step , the error is backpropagated
step6:-do update weights after each observation or in batch
step7:- when the whole training set passed through ann that makes an epoch. redo more epochs

#business problem description :-
-10k customers  with rownumber , surnames, creditstores, etc
-they are seeing a fall in the number of customers are leaving the bank at a high rate and they want to understand what the problem is
-so for 6 months it keeps a data whether the customer if they left or stayed in the bank
******* so our main motive is to find who of the customers remaining are at the highest risk of leaving the bank in the next months
******* this is also applicable in if should the person be approved for a loan in the bank or for credit or not ...
  
-theano lib:-
-it is numerical computation lib based on numpy syntax , can run on both cpu and gpu(graphic processor) 
-gpu is more powerful than cpu as it has many more cores and able to handel more computations

-tensorflow lib:-same as theano , developed by google
-keras lib:- combination of both tensorflow and theano
"""
#PART1:- DATA  PREPROCESSING(1- For person leaving the bank, 0 for person staying)
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

#we have categorical variables in this so we need to encode them 
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()#this is for the 3 country names
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()#this is for male or female
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])#creating these dummy variables
X = onehotencoder.fit_transform(X).toarray()
X= X[:,1:]#removing one dummy variable so that we dnt fall in the dummy variable trap


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling#this is necessary in this 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""PART2 - NOW LETS MAKE A ANN****"""

""" importing the keras library and packages"""
import keras
from keras.models import Sequential#required to initialise our neural network
from keras.layers import Dense# required to build the layers of our ANN



"""#INITIALISING THE ANN"""
classifier = Sequential()#this is the future artifical neural network that we are going to build,we will fit the variables afterwards one by one

"""******Adding the input and the first hidden layer"""
#using the 7 steps mentioned above in lecture notes
#we will choose the rectifier function for te hidden layer and sigmoid function for the output layer
#what is the optimal number of nodes in the hidden layer :- choose the average of the input layer nodes and the avg of output nodes layer
# or using parameter tuning :- kfoldcrossvalidation in which we experiment different parameter models and then test them one by one , we will do this in part 10
#no. of nodes in input layer =11 and output layer =1 so avg =11+1/2=6 nodes
classifier.add(Dense(output_dim=6,init='uniform', activation='relu',input_dim=11))#relu stands for rectifier function
#input_dim is extremely imp as so far it is just initialised , we have to add hidden layer as it doesnt know which node it is accepting as input. we will only need to specify it one time 
#as next time it will already know what to expect from using this first hidden layer , so 11 independent variables for this 


"""adding the second hidden layer to our model"""
classifier.add(Dense(output_dim=6,init='uniform', activation='relu'))
#no use of input_dim here as it now knows what to expect as input nodes but now it knows it needs to expect 11 hidden layer nodes as from the first hidden layer

"""adding the output layer"""
classifier.add(Dense(output_dim=1 ,init='uniform', activation='sigmoid'))#here we want to have probabilities of people leaving the bank
#as we need only one node because our dependent variable is a categorical variable with binary outcome i.e 0 if he stays and 1 if he leaves and when we have a binary outcome there is only one layer
#if we are dealing wiht dependent variable with 3 categories then we chnge ouput_dim=3 and the next thing activation =softmax 


"""COMPILING our atrificial  neural network"""
classifier.compile(optimizer = 'adam' , loss ='binary_crossentropy',metrics = ['accuracy'])#the optimizer is simply the algo u want to use to find the optimal set of weights for our nn
#this algo is nothing else  than stockastic gradient descent algo , adam is a part of this algo
#loss is also a function of stockastic gradient descent algo
#making an accuracy matirx i.e a list of accuracy matrix



"""Fitting the ANN to the training set"""
classifier.fit(X_train,Y_train,batch_size=10, nb_epoch=100)
#addind two additional arguments

"""Making the predicitons(predicted probabilities and evaluating the model"""
# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred=(Y_pred > 0.5)
#if y_pred >0.5 ---> True else false

""" Making the Confusion Matrix for checking our model's accuracy"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

"""so, correct predicitons / total predictions =>(1548+131)/2000 = 83.95% accuracy of our model"""






























            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            