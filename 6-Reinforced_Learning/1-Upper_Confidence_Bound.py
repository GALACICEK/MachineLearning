# Reinforced Learning

# A/B Test
# Example :
# AlphaGo
# One Armed Bandit

# Let's try to find the best one that can be clicked according to the number of clicks on the ads.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading data sets----------------------------------------------
veriler = pd.read_csv('data/Ads_CTR_Optimisation.csv')

# Random Selection----------------------------------------------

import random

N = 10000
d = 10
toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)
    # if our dataset n. row = 1, odul variable is equal 1
    secilenler.append(ad)
    odul = veriler.values[n,ad]
    toplam += odul

print('Total Prize')
print(toplam)

plt.hist(secilenler)
plt.show()

# UCB----------------------------------------------
import math

N = 10000 # 10.000 clicked
d = 10 # total 10 ads
#Ri(n)
oduller = [0] * d # whole ads prize is equal 0
#Ni(n)
tiklamalar = [0] * d # updated clicks
toplam = 0 # total prize
secilenler = []
for n in range(0,N):
    ad = 0 #selected ad
    max_ucb = 0
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ortalama = oduller[i]/ tiklamalar[i]
            delta = math.sqrt(3/2*math.log(n)/tiklamalar[i]) 
            ucb = ortalama + delta
        else :
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i

    secilenler.append(ad)
    tiklamalar[ad] +=1
    odul = veriler.values[n,ad]
    oduller[ad] += odul 
    toplam += odul

print('Total Prize')
print(toplam)

plt.hist(secilenler)
plt.show()













