# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 01:41:43 2019

@author: Akshay
"""
#formula for logistic regressioon = ln(p/1-p)=b0 + b1*x
 
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#here we need to find the coorelation btwn the ade and estimated salary with the purchased one
 ##importing datasets
dataset = pd.read_csv('Social_Network_Ads.csv') #variable dataset, now use pandas to import the dataset
X = dataset.iloc[:,[2,3]].values #taking all the lines by the first : , now after this [:-1] means taking all columns except the last one i.e purchased column
Y =dataset.iloc[:,4].values #dependent variable vector {taking all the lines of the column 3rd}

##splitting the dataset into the training set and Test set
from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y ,test_size=0.25, random_state= 0)

#****feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test) 

#fitting the logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

#predicting the test set results
Y_pred = classifier.predict(X_test)

#Making the confusion matrix i.e it will contain the correct and the incorrect predictions our model made on the dataset
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test ,Y_pred )#the true values are y_test and the predicted values y_pred
#so 65+24 are correct predictions and the 8+3 are incorrect predictions
#to evaluate the model performance

#***visualising the training set results
#the goal is to classify the right users into the right category by using a classifier
#so instead of a straight line we need a curved line to get all the points in the right category
#we took all the pixel points of the framewaork and applied our classifier on them
#for red =0 and for green= 1
from matplotlib.colors import ListedColormap
X_set,Y_set = X_train , Y_train
X1 , X2= np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                     np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))#contour is the line btn the red and the green region
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set(Y_set==j,0),X_set(Y_set==j,1)),
    c= ListedColormap((('red','green'))(i),label=j)#with this loop we plot all the points in the red and the green region
plt.title('Logistic regression (Training set)')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()
                  
#visualising the results for the test set
from matplotlib.colors import ListedColormap
X_set,Y_set = X_test , Y_test
X1 , X2= np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                     np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set(Y_set==j,0),X_set(Y_set==j,1)),
    c= ListedColormap((('red','green'))(i),label=j)
plt.title('Logistic regression (Test set)')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()
                     


















