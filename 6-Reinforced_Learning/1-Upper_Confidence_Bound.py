# Reinforced Learning

# A/B Test
# Example :
# AlphaGo
# One Armed Bandit

# Random distribution
# Let's try to find the best one that can be clicked according to the number of clicks on the ads.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading data sets----------------------------------------------
advertisement = pd.read_csv('data/Ads_CTR_Optimisation.csv')

# Random Selection----------------------------------------------

import random

N = 10000 # 10.000 total row in dataaset 
d = 10 # total 10 ads clicked
total = 0
selections = []

for n in range(0,N):
    ad = random.randrange(d)
    selections.append(ad)
    # if our dataset n. row = 1, award variable is equal 1 and total award increase +1
    # n. row, award variable is equal 0 and total award some value
    award = advertisement.values[n,ad]
    total += award

print('Total Award: ')
print(total)

plt.hist(selections)
plt.show()

# UCB----------------------------------------------
import math

N = 10000 # 10.000 clicked
d = 10 # total 10 ads

# Ri(n) : total rewards from ad i so far
awards = [0] * d # whole ads awards is equal 0

# Ni(n) : Number of clicks on ad number i so far
clickeds = [0] * d # updated clicks

total = 0 # total awards
selections = []

for n in range(1,N):
    ad = 0 #selected ads
    max_ucb = 0

    for i in range(0,d):
        if(clickeds[i]>0):
            avg_award = awards[i]/ clickeds[i]
            delta = math.sqrt(3/2*math.log(n)/clickeds[i]) 
            ucb = avg_award + delta
        else :
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i

    selections.append(ad)
    clickeds[ad] +=1
    award = advertisement.values[n,ad]
    awards[ad] += award 
    total += award

print('Total Award: ')
print(total)

plt.hist(selections)
plt.show()













