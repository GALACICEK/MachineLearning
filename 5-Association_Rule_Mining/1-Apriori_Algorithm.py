# Breadth First Search
# ARM (Association Rule Mining) / ARL (Association Rule Learning),
# Correlation vs Causality
# Support 
# Confidence interval
# Lift = confidence interval / support

# Frequency is important for apriori algorithm

# Usage Areas :
# Complex Event Processing
# Campaign
# Forecast Behavear
# Direced ARM
# Time Series Analysis

# source: https://github.com/ymoch/apyori/blob/master/apyori.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading data sets----------------------------------------------
veriler = pd.read_csv('data/sepet.csv', header = None)

t = []
for i in range(len(veriler)):
    t.append([str(veriler.values[i,j]) for j in range(0,20)])


# wget "https://raw.githubusercontent.com/ymoch/apyori/master/apyori.py" -OutFile "5-Association_Rule_Mining/apyori.py"
# Using apriori lib
from apyori import apriori
rules = apriori(t, min_support = 0.01, min_confidence =0.2, min_lift =3, min_length =2)

print(list(rules))
