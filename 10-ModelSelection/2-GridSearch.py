'''

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Preprocessing ----------------------------------------------
#Loading data sets
dataset = pd.read_csv('data\Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Split test and train variables--------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Datas Scaler---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM ---------------------------------------------------------------------------
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

#Predicts
y_pred = classifier.predict(X_test)

#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Split validation Confusion Matrix:")
print(cm)


#Grid Search ---------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

p = [{'C': [1,2,3,4,5],'kernel':['linear','rbf']},
     {'C': [1,2,3,4,5],'kernel':['rbf'], 'gamma':[1, 0.5, 0.1,0.01, 0.001]} ]

gs = GridSearchCV(estimator=classifier,
                  param_grid=p,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

grid_search = gs.fit(X_train,y_train)

bestsuccess = grid_search.best_score_
print(bestsuccess)
bestparameters = grid_search.best_params_
print(bestparameters)

