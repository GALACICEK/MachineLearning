'''
Principal ComPonent Analysis

Noise Filter
Visualizing
Future Extraction
Future Eleminate / Translate

Stock Market Analysis
Gealth Data / Genetic Data etc.

Algorithm:
    - Reducing requested Matrix let' say k,
    - Make Data Standardisation ,
    - Take Covariance or Correlation for eigen value and eigen vector
    - Eigen Value align big to small and take as many as k
    - Create W projection matrix from eigen value choosen number k 
    - Translate the orijinal datasets X using W and obtain the k-diamention space Y

    
Component axes that maximize the variance
-Unsupervised

- https://sebastianraschka.com/Articles/2014_pca_step_by_step.html

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Preprocessing ----------------------------------------------
#Loading data sets
datas = pd.read_csv('data\Wine.csv')

X = datas.iloc[:,0:13].values
y = datas.iloc [:,13].values

# Split test and train variables--------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#Datas Scaler---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#PCA ---------------------------------------------------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X_trainPCA = pca.fit_transform(X_train)
X_testPCA = pca.transform(X_test)

# Logistic Regression 
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

classifierPCA = LogisticRegression(random_state=0)
classifierPCA.fit(X_trainPCA,y_train)


#Predicts
y_pred = classifier.predict(X_test)
y_predPCA = classifierPCA.predict(X_testPCA)

#Confusion Matrix 
from sklearn.metrics import confusion_matrix

print("actual/without PCA")
cm = confusion_matrix(y_test,y_pred)
print(cm)

print("actual/with PCA")
cmp1 = confusion_matrix(y_test, y_predPCA)
print(cmp1)

print("without PCA/ with PCA")
cm2 = confusion_matrix(y_pred, y_predPCA)
print(cm2)




