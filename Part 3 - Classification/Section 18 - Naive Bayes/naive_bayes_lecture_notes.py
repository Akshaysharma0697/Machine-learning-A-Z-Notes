# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 19:09:39 2019

@author: Akshay
"""
#BAYES THEOREM :-P(A/B)=[P(B/A)*P(A)]/P(B)
#Naive bayes classifier :-
#given to select a new person eihter walks or drives to work :-p(walks/x)=[p(x/walks)*p(walks)]/p(x);here x is the features
#step1
#here we calculate the things in this manner :- 1.prior probability i.e p(walks ) 2.marginal likelihood i.e p(x),3.likelihood i.e p(x/walks)
#and in the last 4.posterior probability i.e p(walks /x)
#step2 :- we calculate now p(drives/x)=[p(x/drives)*p(drives)]/p(x)
#first step:-p(wALKS)=no. of walkers /total people
#to calculte p(x):- we need to select a radius around the new input observation ;therefore p(x)=no. of similar observations/total no. of observations
#to calculate p(x/walks):-again we need to select a radius around the new input observation ;p(x/walks)=among people who walk/total number of walkers
#do same for p(drives/x)
#after this we get p(walks/x)>p(drives/x) ; so the new entered point will be the person who walks to job(red)

#1. why is it called "naive"?ans. 1. bayes theorem requires some independence assumptions or naive assumption(i.e age and salary have to be independent)
#2. p(x):-it is the likelihood that a randomly selected point from the dataset will exhibit the features to the data that we are about to add...remains the same for both sides
#3. what happens when we have More than 2 classes (i.e walks or drives or more):-in case of two if the first one has >50% probability then it is the major one, in case of more than 2 classes calculate all then compare


# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()   