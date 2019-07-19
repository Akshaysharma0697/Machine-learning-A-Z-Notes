# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:13:02 2019

@author: Akshay
"""
#nlp is a branch of ml where we do some analysis on some texts
#we will be able to predict in which category an article belong to in case of newspaper and so on
#here is an example of reviews of a restaurant   
#TEXT CLEANING PROCESS
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#tsv = ab seperated values , csv =  comma separated values
#if we use the delimiter as csv comma then it will be a problem for it , so we use tsv 
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter= '\t',quoting=3)#this quoting=3 we are ignoring the double quotes

#cleaning the texts for making it a generalization model
#we clean the first review and then make a for loop to clean the other reviews
import re
import nltk#library useful for removing texts
from nltk.corpus import stopwords#corpus = collection of written texts
from nltk.stem.porter import PorterStemmer
corpus = []#a list of 1000 reviews
for i in range(0,1000):
    
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])#cleaning all the commas , question marks etc and just keeping the letters only
    #[^a-zA-Z] means that we do not want to remove the letters from a-z ,then we add ' 'so that the removed item is replaced with space, then we add the first review
    #step2:- changing all the letters from upper case to the lower case
    review = review.lower()
    #step3:- we remove the non significant words i.e not useful for predicting the review is either a negative or positive review
    #eg:-  this, the , and , so on
    review =  review.split()
    ps = PorterStemmer()#object of class stemmer
    #step4:- steming process is about keeping only the root words of the words or removing sparsity(i.e not in past , future or present tense)
    review =  [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]#taking all the words in the review that are not in the stopwords list(english words not relevant to texts)
    #step5:-
    review = ''.join(review)
    corpus.append(review)

#creating the bag of words model:-
    #removing the duplicates and creating only one cloumn for each word and we will make a table 1000 rows corresponding to the reviews and many  columns including different words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500 )
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,1].values 

#training the test sets to the model
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
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

the accuracy of our model is this =(1+97)/200





