# K-NN (K nearest neighborhood)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Loading data sets----------------------------------------------
veriler = pd.read_csv('data/veriler.csv')


X = veriler.iloc[:,1:4].values
y= veriler.iloc[:,4:].values

# Split test and train variables--------------------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=0)


#Datas Scaler---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#KNN ----------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)


#Confusion Matrix---------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
