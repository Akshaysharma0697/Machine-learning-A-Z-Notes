# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 18:28:02 2019

@author: Akshay
"""
# thompson sampling 
#in ucb we had a conficence box , it is a deterministic algo , requires update at every round , 
# in thompson we choose by taking the better  range , it is a probabilistic algo , can accomodate delayed feedback, better empirical evidence
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.CSV')#ctr = click through rate of ads
#this data is simulation , we have no real data
#these algo decides which ad to show to the user in the next round based on previous ones

#implementing the Thompson sampling
import random
N=10000
d=10
ads_selected = []
#step1:-at each round n , we consider two numbers for each ad i :
 #   Ni(n):- the number of times  the ad got reward 1  upto round n
  #  Ii(n):- the number of times the ad got reward 0 upto round n
numbers_of_rewards_1 =[0] * d
numbers_of_rewards_0 =[0] * d
total_reward = 0
#step2:- for each ad i 
for n in range(0, N):
    ad = 0
    max_random=0#maximum random draw
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] +1 ,numbers_of_rewards_0[i]+1)#this will give us random draws of the beta distribution
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward==1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] +1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] +1
    total_reward = total_reward +reward
  
  
 #visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('ads')
plt.ylabel('no. of times each add was selected')
plt.show()

 




























