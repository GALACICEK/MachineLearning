# Support Vector Regression (SVR)
# First useing classification problems
# Purpose : select line that will give the max offset margin data point
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Preprocessing------------------------------------------_
veriler = pd.read_csv('data/maaslar.csv')

# Dataframe Slicing And Transform to Array
X= veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values


#Datas Scaler--------------------------------------------------
from sklearn.preprocessing import StandardScaler

sc1 =StandardScaler()
X_scaler = sc1.fit_transform(X)
sc2 =StandardScaler()
y_scaler = sc2.fit_transform(y)

# SVR kernel = rbf ---------------------------------------
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(X_scaler, y_scaler)

# SVR kernel = linear ---------------------------------------
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
svr_lin.fit(X_scaler,y_scaler)

# SVR kernel = poly ---------------------------------------
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
svr_poly.fit(X_scaler,y_scaler)
# Visualization------------------------------------------

plt.scatter(X_scaler,y_scaler,color = "red")
plt.plot(X_scaler,svr_rbf.predict(X_scaler), color = "steelblue")
plt.title("'rbf' curve on SVR")
plt.show()

plt.scatter(X_scaler,y_scaler,color = "red")
plt.plot(X_scaler,svr_lin.predict(X_scaler), color = "steelblue")
plt.title("'linear' curve on SVR")
plt.show()

plt.scatter(X_scaler,y_scaler,color = "red")
plt.plot(X_scaler,svr_poly.predict(X_scaler), color = "steelblue")
plt.title("'poly' curve on SVR")
plt.show()

# Predictions------------------------------------------

print(svr_rbf.predict(np.array(11.0).reshape(1, -1)))
print(svr_rbf.predict(np.array(6.0).reshape(1, -1)))

print(svr_lin.predict(np.array(11.0).reshape(1, -1)))
print(svr_lin.predict(np.array(6.0).reshape(1, -1)))

print(svr_poly.predict(np.array(11.0).reshape(1, -1)))
print(svr_poly.predict(np.array(6.0).reshape(1, -1)))












