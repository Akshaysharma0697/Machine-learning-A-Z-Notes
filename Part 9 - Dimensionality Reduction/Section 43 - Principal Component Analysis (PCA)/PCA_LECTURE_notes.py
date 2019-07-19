# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:17:47 2019

@author: Akshay
"""

#dimensionality reduction:-
1 feature selection:-  backward elimination , forward sekection, bidirectional elimination, score comparison
2 feature extraction :- PCA, LDA,kernel PCA

PCA:- from the m independent variables of your dataset, PCA extracts p<=m new independent variables that explain
the most the variance of the dataset , regardless of the dependent variable .
the fact that the Dependent variable is not considered makes PCA an unsupervised model.

-independent variable was used because :- we needed graphical visualisation of our results and since each iv
corresponded to one dimension in the plot so it was easy with 2 independent variables.
-we can reduce this number of independent variable using PCA

"""PCA"""
#business owner found 3 types of wines :- each correspoding to one segment of customers
#our model will give customer segment that each new wine should be recommended to 
#as there are 13 independent variables so we need to reduce the number of variables or reduce dimnesionality in this
# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
Y = dataset.iloc[:,  13].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""applying PCA"""
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)#number of extracted features that will explain the most the variance
X_train= pca.fit_transform(X_train)
X_test= pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_#this will give us the list of all the given components and we can have the perc of variance explained by each of them
#here the exolained vairance vector shows us that the decreasing order of the variances , so we take the first 2 pricipal components as they will explain 57%variance
#the top 2 are the two new independent variables
#so we replace the None by 2


#fitting the logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix for finding the accuracy of our model we made
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()