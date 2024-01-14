# Regression_Example
# calculate tu use MLR(multiple linear), PR(polynomial), SVR(support vector), DT(dessicion tree), RF(random forest) techniqies
# Find eveluation parameters

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
import statsmodels.api as sm

#Preprocessing-------------------------------------------
veriler = pd.read_csv('data/maaslar_yeni.csv')

# Dataframe Slicing And Transform to Array
X= veriler.iloc[:,[2,4]].values
#X= veriler.iloc[:,[2:5].values
y = veriler.iloc[:,5:].values

print(veriler.corr())

"""# Linear Regression------------------------------------------"""
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,y)

# R2 Score----------------------------------------
print("Linear Regression R2 Score:")
print(r2_score(y,lin_reg.predict(X)))


# Backward Elimination-----------------------------------------
print("Linear Regression Backward Elimination:")
model_lin = sm.OLS(lin_reg.predict(X),X)
print(model_lin.fit().summary())


"""# Polynomial Regression------------------------------------------"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Second Degree Polynomial
poly_reg2 = PolynomialFeatures(degree=2)
X_poly = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)



# R2 Score----------------------------------------
print("Polynomial Regression R2 Score:")
print(r2_score(y,lin_reg2.predict(poly_reg2.fit_transform(X))))

# Backward Elimination-----------------------------------------

print("Polynomial Regression Backward Elimination")
model_poly = sm.OLS(lin_reg2.predict(poly_reg2.fit_transform(X)),X)
print(model_poly.fit().summary())



"""# SVR Regression ---------------------------------------"""
#Datas Scaler--------------------------------------------------
from sklearn.preprocessing import StandardScaler

sc1 =StandardScaler()
X_scaler = sc1.fit_transform(X)
sc2 =StandardScaler()
y_scaler = sc2.fit_transform(y)

# SVR kernel='rbf' ---------------------------------------
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(X_scaler, y_scaler)


# R2 Score----------------------------------------
print("SVR R2 Score:")
print(r2_score(y_scaler,svr_rbf.predict(X_scaler)))

# Backward Elimination-----------------------------------------

print("SVR 'rbf' Regression Backward Elimination")
model_svr = sm.OLS(svr_rbf.predict(X_scaler),X_scaler)
print(model_svr.fit().summary())



"""# Decision Tree----------------------------------------"""

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,y)


# R2 Score----------------------------------------
print("Decision Tree Regression R2 Score:")
print(r2_score(y,r_dt.predict(X)))

# Backward Elimination-----------------------------------------

print("Decision Tree Backward Elimination")
model_dt = sm.OLS(r_dt.predict(X),X)
print(model_dt.fit().summary())



"""# Random Forest----------------------------------------"""
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,y.ravel())


# R2 Score----------------------------------------
print("Random Forest Regression  R2 Score:")
print(r2_score(y,rf_reg.predict(X)))


# Backward Elimination-----------------------------------------

print("Random Forest Backward Elimination")
model_rf = sm.OLS(rf_reg.predict(X),X)
print(model_rf.fit().summary())


