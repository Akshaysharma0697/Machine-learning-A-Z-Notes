# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:35:37 2019

@author: Akshay
"""
# MULTI ARMED BANDIT PROBLEM:-
#if you have 500 adds which one is the best ?
#if you have d arms. for eg. arms are ads that we dusplay to users each time they connect to a web page
#each tme a user connects to this web page , that makes a round
#t each n round , we choose one add to display to the user
#at each round n , ad i gives a reward r(n) for all {0,1},r(n)=1 if the user clicked on the ad , 0 if didnt display
#our goal is to maximize the total reward we get over many rounds

#Upper Confidence Bound ALGO:-
#we first assume that all give the same result
#we then assume the confience band adn assume that it includes the the expected value falls inside the confidence round

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.CSV')#ctr = click through rate of ads
#this data is simulation , we have no real data
#these algo decides which ad to show to the user in the next round based on previous ones

#implementing the UCB 
import math
N=10000
d=10
ads_selected = []
#step1:-at each round n , we consider two numbers for each ad i :
 #   Ni(n):- the number of rewards the ad was selected upto round n
  #  Ii(n):- the sum of rewards of the ad i upto round n
numbers_of_selections = [0] * d #create a vetor of size d containing 0  
sums_of_rewards = [0] * d 
total_reward = 0
#step2:- from these 2 numbers we compute
#the avg rewards of ad i upto round n   :- ri(n)= Ri(n)/Ni(n)
#the confidence interval also
for n in range(0, N):
    ad = 0
    max_upper_bound=0
    for i in range(0,d):
        if (numbers_of_selections[i] >0) :
            average_reward= sums_of_rewards[i]/ numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i])#since n starts at 0
            upper_bound = average_reward + delta_i
        else:
             upper_bound = 1e400#a large value 
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] +1
    reward = dataset.values[n,ad]  
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward +reward
  
  
 #visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('ads')
plt.ylabel('no. of times each add was selected')
plt.show()






 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  