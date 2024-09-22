'''
LDA (Linear Discriminant Analysis)

Maximizing the component axes for class separation

This algorithm is Like PCA

- Supervised
- https://sebastianraschka.com/Articles/2014_python_lda.html
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

#LDA ---------------------------------------------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 2)
X_trainLDA = lda.fit_transform(X_train,y_train)
X_testLDA = lda.transform(X_test)

# Logistic Regression 
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

classifierLDA = LogisticRegression(random_state=0)
classifierLDA.fit(X_trainLDA,y_train)


#Predicts
y_pred = classifier.predict(X_test)
y_predLDA = classifierLDA.predict(X_testLDA)

#Confusion Matrix 
from sklearn.metrics import confusion_matrix

print("actual/without LDA")
cm = confusion_matrix(y_test,y_pred)
print(cm)

print("actual/with LDA")
cm1 = confusion_matrix(y_test, y_predLDA)
print(cm1)

print("without LDA/ with LDA")
cm2 = confusion_matrix(y_pred, y_predLDA)
print(cm2)




