# Decission_Tree
# Quinlan's ID3 Algoritm
#   Information Gain finding decission tree startpoint which independent values
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

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

#Decision Tree Classifier 'entropy'----------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)

#Confusion Matrix---------------------------------------------------------------------------
cm_dtc = confusion_matrix(y_test,y_pred)
print('DTC_Entropy')
print(cm_dtc)

# Tree Shape Visualization ----------------------------------------
from sklearn import tree

plt.figure(figsize=(12, 8))
tree.plot_tree(dtc, feature_names=X.columns.tolist(), filled=True, rounded=True)
plt.show()

#Decision Tree Classifier 'gini'----------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

dtc2 = DecisionTreeClassifier(criterion='gini')
dtc2.fit(X_train,y_train)

y_pred = dtc2.predict(X_test)

#Confusion Matrix---------------------------------------------------------------------------
cm_dtc = confusion_matrix(y_test,y_pred)
print('DTC_Gini')
print(cm_dtc)

# Tree Shape Visualization ----------------------------------------
from sklearn import tree

plt.figure(figsize=(12, 8))
tree.plot_tree(dtc2, feature_names=X.columns.tolist(), filled=True, rounded=True)
plt.show()


