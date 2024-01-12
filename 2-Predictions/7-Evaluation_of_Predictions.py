# Evaluation of Predictions

# Adjusted R^2 Score
# Adjusted R^2 = 1- (1-R^2) x ((n-1)/(n-p-1))
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

#Preprocessing-------------------------------------------
veriler = pd.read_csv('data/maaslar.csv')

# Dataframe Slicing And Transform to Array
X= veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values


"""# Linear Regression------------------------------------------"""
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Visualization------------------------------------------

plt.scatter(X,y,color = "red")
plt.plot(X,lin_reg.predict(X), color = "steelblue")
plt.title("Linear Regression")
plt.show()

# R2 Score----------------------------------------
print("Polynomial Regression R2 Score:")
print(r2_score(y,lin_reg.predict(X)))


"""# Polynomial Regression------------------------------------------"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Second Degree Polynomial
poly_reg2 = PolynomialFeatures(degree=2)
X_poly = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)


# Visualization------------------------------------------

plt.scatter(X,y,color = "red")
plt.plot(X,lin_reg2.predict(poly_reg2.fit_transform(X)), color = "steelblue")
plt.title("Second Degree Polynomial Regression")
plt.show()

# R2 Score----------------------------------------
print("Linear Regression R2 Score:")
print(r2_score(y,lin_reg2.predict(poly_reg2.fit_transform(X))))


"""# SVR ---------------------------------------"""
#Datas Scaler--------------------------------------------------
from sklearn.preprocessing import StandardScaler

sc1 =StandardScaler()
X_scaler = sc1.fit_transform(X)
sc2 =StandardScaler()
y_scaler = sc2.fit_transform(y)

# SVR ---------------------------------------
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(X_scaler, y_scaler)

# Visualization------------------------------------------

plt.scatter(X_scaler,y_scaler,color = "red")
plt.plot(X_scaler,svr_rbf.predict(X_scaler), color = "steelblue")
plt.title("'rbf' curve on SVR")
plt.show()

# R2 Score----------------------------------------
print("SVR R2 Score:")
print(r2_score(y_scaler,svr_rbf.predict(X_scaler)))


"""# Decision Tree----------------------------------------"""

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,y)

# Visualization------------------------------------------
Z = X + 0.5
K = X - 0.4
plt.scatter(X,y,color = "red")
plt.plot(X,r_dt.predict(X), color = "steelblue")

plt.plot(X,r_dt.predict(Z),color = "green")
plt.plot(X,r_dt.predict(K), color = "yellow")

plt.title("Decision Tree")
plt.show()

# R2 Score----------------------------------------
print("Decision Tree Regression R2 Score:")
print(r2_score(y,r_dt.predict(X)))


"""# Random Forest----------------------------------------"""
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,y.ravel())


# Visualization------------------------------------------
Z = X + 0.5
K = X - 0.4
plt.scatter(X,y,color = "red")
plt.plot(X,rf_reg.predict(X), color = "steelblue")

plt.plot(X,rf_reg.predict(Z),color = "green")
plt.plot(X,rf_reg.predict(K), color = "yellow")

plt.title("Random Forest Regression")
plt.show()

# R2 Score----------------------------------------
print("Random Forest Regression  R2 Score:")
print(r2_score(y,rf_reg.predict(X)))


""" OutPuts:
Linear Regression R2 Score:
0.6690412331929894
Polynomial Regression R2 Score:
0.9162082221443942
SVR R2 Score:
0.7513836788854973
Decision Tree Regression R2 Score:
1.0
Random Forest Regression  R2 Score:
0.9704434230386582
"""

