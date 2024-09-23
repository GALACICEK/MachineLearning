'''
https://xgboost.readthedocs.io/en/stable/

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Preprocessing ----------------------------------------------
#Loading data sets
datas = pd.read_csv('data\Churn_Modelling.csv')

X = datas.iloc[:,3:13].values
Y = datas.iloc [:,13].values

# Encoder Categoric -> Numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

le = LabelEncoder()
X[:,1] =le.fit_transform(X[:,1])

le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
X = X[:,1:]


# Split test and train variables--------------------------------------------------
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, random_state=0)

#XGBoost---------------------------------------------------------------------------
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


#Confusion Matrix 
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)

