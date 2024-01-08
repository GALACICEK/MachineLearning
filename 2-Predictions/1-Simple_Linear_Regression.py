# Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Preprocessing ------------------------------------------
veriler = pd.read_csv('data/satislar.csv')

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values
print(satislar2)

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)


"""sc =StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc .fit_transform(y_train)
Y_test = sc.fit_transform(y_test)


print("X_train : ",X_train)
print("X_test : ",X_test)"""

# Modelling------------------------------------------

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

# from tahmin values predicted Y_test values
tahmin = lr.predict(x_test)


# Visualization------------------------------------------

x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("Aylara Göre Satışlar")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.show()

# Comment: We have drawn the line y = α+βx+ε closest to the selected values.


