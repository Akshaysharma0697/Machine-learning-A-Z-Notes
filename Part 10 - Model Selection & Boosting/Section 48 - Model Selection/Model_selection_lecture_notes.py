# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:18:13 2019
"""
-to improve the model performance can be done bY model selection that can be done bY choosing the correct parameters

#K-fold cross validation
-this fixes the variance problem bY splitting the training set into 10 folds i.e k=10 and we train on 9 folds and 1 fold to test it 
-we will use kernel svm model for this test

# Kernel SVM

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

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

"""APPLYING K FOLD CROSS VALIDATION"""
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier , X = X_train , y= Y_train , cv = 10)#cv is the no.. of folds You want to split it into
#accuracY is a vector that will contain the 10 accuracY that will be computed through the 10 combinations created through k fold cross validations
accuracy.mean()#gives us the mean of all the percentages of all 10 accuracies
accuracy.std()#gives us the standard deviation i.e btwn 84% and 90% 

"""Grid search"""
what is the optimal value of these parameters?
how do i knw which model to choose to solve my problem ?
step1:- we need to look at the dependent variable - if there is no dependent variable then --->clustering
-if the dependent variable is continuos outcome ----> regression
-if   "              "           is categorical outcome----> classification 
step2:-is my problem a linear or a non linear problem ???
       this will be solved by grid search
-grid search can be applied after , as it uses the classifier
-first we evaluate the model performance and  then we reduce it

#applying the grid search to choose the best model and the best parameter
from sklearn.model_selection import GridSearchCV
parameters = [{ 'C':[1,10,100,1000],'kernel':['linear']} , #this is for linear
               {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.001,0.0001]}]#this is for non linear
 #will tell you the optimal value for the penalty parameter c ,by this we should know that if we want to use a linear or non linear model

grid_search= GridSearchCV(estimator= classifier , param_grid = parameters ,scoring= 'accuracy', cv = 10 , n_jobs=-1)
grid_search = grid_search.fit(X_train,Y_train)
best_accuracy =grid_search.best_score_
best_parameters= grid_search.best_params_
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.arraY([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.Ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.Ylabel('Estimated SalarY')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.arraY([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.Ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.Ylabel('Estimated SalarY')
plt.legend()
plt.show()