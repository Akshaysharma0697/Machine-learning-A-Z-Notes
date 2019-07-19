# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:24:08 2019

@author: Akshay
"""
# we have different clusters of data points so kmeans identifies the number of gropups and separates them
#step1:- choose the number of clusters k
#step2:- select at random k points , the centroids(not necessarily from dataset)
#step3:- assign each data point to the closest centroid ; that forms the k clusters
#step4:- compute and place the new centroid of each cluster
#step5:- reassign each data point to the new closest centroid;if any assignment took place goto step4 otherwise go to fin.
#we keep rotating the perpendicular line until no point is on the line and all the points are in their correst region
#in the end remove the centroid and the line and the model is ready


#random initialization trap
#solution is k-means++

#choosing the right number of clusters:-
#wcss(within clusters sum of squares)=E distance(p1,c1)^2+p2 of c2 and so on;p1 in cluster 1,E is the summation

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the mall dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i , init = 'k-means++',max_iter=300,n_init=10,random_state=0)#object of the kmeans class; n_cluster is the number of clusteri.e i ;init :-random initialization method(we do not want to fall in the random initialization trap
    #;max_iter:max no. of iterations;n_init:- no. of times the kmeans algo will run with different inital centorids )  
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("the elbow method")
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()

#applying the kmeans to the mall dataset with 5 clusters
kmeans=KMeans(n_clusters=5, init = 'k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

#visualizing the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='carefull')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300, c='yellow',label='centroids')
plt.title('clusters of clients')
plt.xlabel("annual income")
plt.ylabel('spending score(1-100)')
plt.show()









