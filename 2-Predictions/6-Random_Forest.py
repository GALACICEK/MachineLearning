# Random Forest
# Developed Decision Tree and can return different values in Random Forest
# But Decision Tree can return same values
# Essemble Learning 
# Majority voted Learning


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Preprocessing-------------------------------------------
veriler = pd.read_csv('data/maaslar.csv')

# Dataframe Slicing And Transform to Array
X= veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

# Random Forest----------------------------------------
# n_estimators=10 is meaning drawing 10 decision tree
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

plt.title("Random Forest")
plt.show()


# Predictions------------------------------------------

print(rf_reg.predict(np.array(11.0).reshape(1, -1)))
print(rf_reg.predict(np.array(6.0).reshape(1, -1)))
