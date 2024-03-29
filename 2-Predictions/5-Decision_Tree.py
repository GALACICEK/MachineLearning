# Decision Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Preprocessing-------------------------------------------
veriler = pd.read_csv('data/maaslar.csv')

# Dataframe Slicing
X= veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# Decision Tree----------------------------------------

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

# Tree Shape Visualization ----------------------------------------
from sklearn import tree

plt.figure(figsize=(12, 8))
tree.plot_tree(r_dt, feature_names=X.columns.tolist(), filled=True, rounded=True)
plt.show()

# Predictions------------------------------------------

print(r_dt.predict(np.array(11.0).reshape(1, -1)))
print(r_dt.predict(np.array(6.0).reshape(1, -1)))











