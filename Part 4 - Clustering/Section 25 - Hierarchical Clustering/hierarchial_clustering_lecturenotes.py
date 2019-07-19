# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 12:11:09 2019

@author: Akshay
"""
#HC:-clusters similar to kmeans; process is different
#2 types:- 1 agglomerative:- bottom up approach , 2. divisive :- top-bottom
#1.agglomerative:- 
#step1:- make each datapoint a single point cluster
#step2:- take two closest data points and make them one cluster(n-1 cluster)
#step3:-take two closest data points and make them one cluster(n-2 cluster)
#step4:- repeat step 3 until only one cluster is left

#purpose of HC ??? 
#how do dendograms work ???
#it is a graph in shape of hierarchial graphs

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the mall dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#using the dendogram method to find the optimal no. of clusters
#we import scipy which contains lib to make dendrograms
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))#we create the variable dendrogram, linkage is the algo itself of hierarchial method
#on second term we add method = 'ward' it tries to minimize the variance withing each cluster
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('euclidian distances')
plt.show()

#now we have the clusters = 5 , we can now fit the hierarchial clustering to the mall dataset 
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5,affinity='euclidean', linkage='ward')
y_hc= hc.fit_predict(X)

#visualising the clusters results
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='carefull')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='sensible')
plt.title('clusters of clients')
plt.xlabel("annual income")
plt.ylabel('spending score(1-100)')
plt.show()













