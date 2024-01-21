# NaiveBayes
# conditional Probability is important in this theory.
# It allows us to work on unbalanced data sets.

# Gaussian Naive Bayes : predicted column is continuous values, numerical values
# Multinominal Naive Bayes : predicted column is nominal values
# Bernoulli Naive Bayes : predicted column is binomial values
# Source: https://scikit-learn.org/stable/modules/naive_bayes.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

#Loading data sets----------------------------------------------
veriler = pd.read_csv('data/veriler.csv')


X = veriler.iloc[:,1:4].values
y= veriler.iloc[:,4:].values

# Split test and train variables--------------------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=0)


#Gaussian Naive Bayes----------------------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred = gnb.predict(x_test)

#Confusion Matrix---------------------------------------------------------------------------
cm_gnb = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm_gnb)

#Multinominal Naive Bayes----------------------------------------------------------------------------
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(x_train,y_train)

y_pred = mnb.predict(x_test)

#Confusion Matrix---------------------------------------------------------------------------
cm_mnb = confusion_matrix(y_test,y_pred)
print('MNB')
print(cm_mnb)


# Bernoulli Naive Bayes----------------------------------------------------------------------------

from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB().fit(x_test,y_test)

y_pred= bnb.predict(x_test)

#Confusion Matrix---------------------------------------------------------------------------
cm_bnb = confusion_matrix(y_test,y_pred)
print('BNB')
print(cm_bnb)

