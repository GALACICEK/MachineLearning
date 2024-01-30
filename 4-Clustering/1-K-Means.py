#Clustering
#customer segmentation
#Collaboration Filter : Same behavior ability
#market segmentation
#healthcare and image processing segmentation
#https://scikit-learn.org/stable/modules/clustering.html

#Unsupervised Learning
#K-Means
# Give the number of clusters
# Determine the number of clusters is important
# Optimum decided Clusters Numbers, WCSS : within-cluster sums of square 
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Loading data sets----------------------------------------------
veriler = pd.read_csv('data/musteriler.csv')

X = veriler.iloc[:,3:].values

#K-Means Clustering----------------------------------------------

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)

#Find Optimum Cluster number----------------------------------------------
sonuclar = []
for i in range(1,11):
    kmeans = KMeans (n_clusters=i, init='k-means++', random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)
plt.show()

#Plot Optimum Clusters----------------------------------------------
kmeans = KMeans (n_clusters=4, init='k-means++', random_state=123)
y_predic = kmeans.fit_predict(X)

print(y_predic)

plt.scatter(X[y_predic == 0,0],X[y_predic == 0,1],s=100, c='red')
plt.scatter(X[y_predic == 1,0],X[y_predic == 1,1],s=100, c='blue')
plt.scatter(X[y_predic == 2,0],X[y_predic == 2,1],s=100, c='purple')
plt.scatter(X[y_predic == 3,0],X[y_predic == 3,1],s=100, c='yellow')
plt.title('KMeans')
plt.show()
