'''
- pickle
- joblib
- pmml

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Preprocessing ----------------------------------------------
#Loading data sets

url = "http://bilkav.com/satislar.csv"

dataset = pd.read_csv(url)

dataset = dataset.values
X = dataset[:,0:1]
Y = dataset[:,1]

divider = 0.33

# Split test and train variables--------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,Y,test_size= divider)

# Linear Regression ---------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,y_train)

print(lr.predict(X_test))

# Pickle ---------------------------------------------------------------------------

import pickle

my_path = "model.save"

# Save the fitted model
pickle.dump(lr,open(my_path,'wb'))

# Loading the fitting model on file
load_file =pickle.load(open(my_path,'rb'))

print(load_file.predict(X_test))