#Support Vector Machine

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

#SVC kernel='linear'----------------------------------------------------------------------------
from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

#Confusion Matrix---------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("SVC linear: \n",cm)


#SVC kernel='rbf'----------------------------------------------------------------------------
from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

#Confusion Matrix---------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("SVC rbf: \n",cm)


#SVC kernel='poly'----------------------------------------------------------------------------
from sklearn.svm import SVC

svc = SVC(kernel='poly')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

#Confusion Matrix---------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("SVC poly: \n",cm)
