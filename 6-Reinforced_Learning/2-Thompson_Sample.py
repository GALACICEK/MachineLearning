# Reinforced Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading data sets----------------------------------------------
advertisement = pd.read_csv('data/Ads_CTR_Optimisation.csv')

# Thompson Sample ----------------------------------------------
# https://www.nowpublishers.com/article/DownloadSummary/MAL-070
import random

N = 10000 # 10.000 clicked
d = 10 # total 10 ads

total = 0 # total awards
selections = []

ones = [0] *d
zeros = [0] *d

for n in range(1,N):
    ad = 0 #selected ads
    max_th = 0

    for i in range(0,d):
        rand_beta = random.betavariate(ones[i]+1, zeros[i]+1)
        if rand_beta>max_th:
            max_th = rand_beta
            ad = i

    selections.append(ad)
    award = advertisement.values[n,ad]
    if award ==1:
        ones[ad] +=1
    else :
        zeros[ad] +=1
    total += award

print('Total Award: ')
print(total)

plt.hist(selections)
plt.show()




