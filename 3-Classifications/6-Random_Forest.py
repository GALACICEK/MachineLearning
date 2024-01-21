# Random_Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

#Loading data sets----------------------------------------------
veriler = pd.read_csv('data/veriler.csv')

X = veriler.iloc[:,1:4]
y= veriler.iloc[:,4:]

# Split test and train variables--------------------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=0)

#Datas Scaler---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Random Forest Classifier entropy----------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier 

rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)


from sklearn import metrics
y_proba = rfc.predict_proba(X_test)
fpr, tpr, thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')

#Confusion Matrix---------------------------------------------------------------------------
cm_rfc = confusion_matrix(y_test,y_pred)
print('RFC_Entropy')
print(cm_rfc)

#Random Forest Classifier gini----------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier 

rfc2 = RandomForestClassifier(n_estimators=10, criterion='gini')
rfc2.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

#Confusion Matrix---------------------------------------------------------------------------
cm_rfc = confusion_matrix(y_test,y_pred)
print('RFC_Gini')
print(cm_rfc)

