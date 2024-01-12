# Polynomial Regression

# y = β0+β1x+β2x^2+β3x^3+ε


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Preprocessing------------------------------------------_
veriler = pd.read_csv('data/maaslar.csv')

# Dataframe Slicing And Transform to Array
X= veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

# Linear Regression------------------------------------------
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,y)


# Polynomial Regression------------------------------------------
# Non-linear Regression
from sklearn.preprocessing import PolynomialFeatures

# Second Degree Polynomial
poly_reg2 = PolynomialFeatures(degree=2)
X_poly = poly_reg2.fit_transform(X)
#print(X_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)


# Fourth Degree Polynomial
poly_reg4 = PolynomialFeatures(degree=4)
X_poly = poly_reg4.fit_transform(X)
#print(X_poly)
lin_reg4 = LinearRegression()
lin_reg4.fit(X_poly,y)


# Visualization------------------------------------------

plt.scatter(X,y,color = "red")
plt.plot(X,lin_reg.predict(X), color = "steelblue")
plt.title("Linear Regression")
plt.show()

plt.scatter(X,y,color = "red")
plt.plot(X,lin_reg2.predict(poly_reg2.fit_transform(X)), color = "steelblue")
plt.title("Second Degree Polynomial Regression")
plt.show()

plt.scatter(X,y,color = "red")
plt.plot(X,lin_reg4.predict(poly_reg4.fit_transform(X)), color = "steelblue")
plt.title("4. Degree Polynomial Regression")
plt.show()

# Predictions------------------------------------------

print(lin_reg.predict([[11]]))#[[34716.66666667]]
print(lin_reg.predict([[6.6]]))#[[16923.33333333]]
# if educational level is 11 , employee salary 34716. Ceo education level is equal 10 and salary 50000 
# This situation, Linear regression, confused and illogical.Because we expected him to take more than the CEO salary.

print(lin_reg4.predict(poly_reg4.fit_transform([[6.6]])))#[[8146.9948718]]
print(lin_reg4.predict(poly_reg4.fit_transform([[11]])))#[[89041.66666669]]
# This situation, Polynomical regression, more logical.Because we expected him to take more than the CEO salary and done.
