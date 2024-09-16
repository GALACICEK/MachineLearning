'''
Artificial Neural Network(ANN)

Can we operate a computer like a human?

Input values convert to Standardize

Activation Functions important for Neural
- Threshold function
- Sigmoid function (0 to 1)
- Rectifier function
- hyperbolic function (- value and + value)

Input Layer -> Hidden Layer -> Output Layer

Hidden Layer:
inputs is scaling and apply activation func and examie output


Artificial Neural Network(ANN)
Perceptron : c = 1/2 * (result-predict)^2

Gradient Descendent
- Big learning rate
- Small learning rate

Stochastic Gradient Descendent
- Stochastic Stochastic : step to step reducing to give feedback
- Batch Stochastic : calculating all data and then reducing to give feedback
- Mini Batch Stochastic : divide parts all data and at the end of each divided part reducing to give feedback

Backpropagtion
- Forward Propagtion
- Back Propagtion


ANN libs:
- PyTorch           : https://pytorch.org
- TensorFlow        : https://www.tensorflow.org
- Caffe             : http://caffe.berkeleyvision.org
- Keras             : https://keras.io
- DeepLearning4J    : https://deeplearning4j.org
...


'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Preprocessing ----------------------------------------------
#Loading data sets
datas = pd.read_csv('data\Churn_Modelling.csv')

X = datas.iloc[:,3:13].values
Y = datas.iloc [:,13].values

from sklearn import preprocessing

# Encoder Categoric -> Numeric
le = preprocessing.LabelEncoder()
X[:,1] =le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],
                        remainder='passthrough')

X=ohe.fit_transform(X)
X = X[:,1:]


# Split test and train variables--------------------------------------------------
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#Datas Scaler---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#ANN ---------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

classifier = Sequential()

# Input Layer: 11 extentions 
classifier.add(Input(shape=(11,)))

# First Hidden  Layer: 6 neuron
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))

# Second Hidden  Layer: 6 neuron
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))

# Output Layer: 1
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Modelling Summarize
#classifier.summary()


classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics= ['accuracy'])

classifier.fit(X_train, y_train,epochs=50)

y_pred = classifier.predict(X_test)

y_pred= (y_pred>0.5) 

#Confusion Matrix 
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)
