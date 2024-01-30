#Unsupervised Learning
#Hierarchical Clustering
# Agglomerative : bottom approach
# - Accept each data as a separate cluster, and then moving forward by combining
# Divisive : top approach
# - Accept all data as a single cluster, and then forward by dividing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Loading data sets----------------------------------------------
veriler = pd.read_csv('data/musteriler.csv')

X = veriler.iloc[:,3:].values


#Agglomerative Clustering----------------------------------------------

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_predic = ac.fit_predict(X)

print(y_predic)

#Scater Clustering----------------------------------------------
plt.scatter(X[y_predic == 0,0],X[y_predic == 0,1],s=100, c='red')
plt.scatter(X[y_predic == 1,0],X[y_predic == 1,1],s=100, c='blue')
plt.scatter(X[y_predic == 2,0],X[y_predic == 2,1],s=100, c='purple')
plt.scatter(X[y_predic == 3,0],X[y_predic == 3,1],s=100, c='yellow')
plt.title('Agglomerative Clustering')
plt.show()

#Dendogram----------------------------------------------

import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()





