# Example for Multiple Linear Regression


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Preprocessing ------------------------------------------
veriler = pd.read_csv('data/odev_tenis.csv')
print(veriler)

""" Encoding transform First Option
#outlook variable categoric to numeric----------------------------
outlook = veriler.iloc[:,0:1].values
#print(outlook)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

outlook[:,0] = le.fit_transform(veriler.iloc[:,0])
#print(outlook)

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)

#windy variable categoric to numeric----------------------------
windy = veriler.iloc[:,-2:-1].values
#print(windy)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,-1] = le.fit_transform(veriler.iloc[:,-1])
#print(windy)


#play variable categoric to numeric----------------------------
play = veriler.iloc[:,-1:].values
#print(play)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

play[:,-1] = le.fit_transform(veriler.iloc[:,-1])
#print(play)"""

# Encoding transform Second Option
from sklearn import preprocessing

# Whole variables transform to LabelEncoding but we dont prefer to numerical variables change
#Therefore we need to choise we want columns
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

# outlook column transform OneHotEncoder in our data
outlook=veriler2.iloc[:,:1]

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
#print(outlook)

havadurumu = pd.DataFrame(data=outlook, index = range(14), columns=['overcast','rainy','sunny'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonveriler = pd.concat([sonveriler,veriler2.iloc[:,-2:]],axis=1)


# Split test and train variables--------------------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler['humidity'],test_size=0.33, random_state=0)


# Modelling------------------------------------------

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
#print(y_pred)
#print(y_test)


# Backward Elimination-----------------------------------------
import statsmodels.api as sm

X = np.append(arr=np.ones((len(sonveriler),1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)

X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,:-1],X_l).fit()

sonveriler = sonveriler.iloc[:,1:]

# windy colums pvalues is bigest therefore we eliminate
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,:-1],X_l).fit()
print(model.summary())

x_train = x_train.iloc[:,-1:]
x_test = x_test.iloc[:,-1:]

reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)



