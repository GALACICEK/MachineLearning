# Multiple Linear Regression

# y = β0+β1x1+β2x2+β3x3+ε
# Dummy (Kukla) Variable : encoding olayı
# cinsiyet-erkek-kadın  column yalnız birini seçmek gerekir.
# şehir plakalarından oluşan column hepsini alabiliriz.

# P- value (olasılık değeri)
# H0 : null hypotesis H1: alternatif hypotesis
# p-değeri : olasılık değeri (genelde 0.05 alınır.)
# P-değeri küçüldükçe H0 hatalı olma ihtimali artar.
# P-değeri büyüdükçe H1 hatalı olma ihtimali artar.

# Değişken Seçim Yaklaşımları
# Full Variables Selection
# Backward Elimination: significance level choises and bigest p-value elimination
# Forward Selection: significance level choises and lowest p-value selection
# Bidirectional Elimination:significance level choises and lowest p-value 
# Score Comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Preprocessing ------------------------------------------
veriler = pd.read_csv('data/veriler.csv')
print(veriler)

#Encoder ulke variable categoric to numeric----------------------------
ulke = veriler.iloc[:,0:1].values
#print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
#print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#Encoder cinsiyet variable categoric to numeric----------------------------
c = veriler.iloc[:,0:-1].values
#print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
#print(c)

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

# numpy transform to dataframe
sonuc1 = pd.DataFrame(data= ulke, index = range(22), columns = ['fr', 'tr','us'])
#print(sonuc1)

Yas = veriler.iloc[:,1:4].values
#print(Yas)

sonuc2 = pd.DataFrame(data= Yas, index = range(22), columns = ['boy', 'kilo','yas'])
#print(sonuc2)

#Dummy variable choise one column
sonuc3 = pd.DataFrame(data= c[:,:1], index = range(22), columns = ['cinsiyet'])
#print(sonuc3)

#concat dataframes
s1=pd.concat([sonuc1,sonuc2],axis=1)
s2=pd.concat([s1,sonuc3],axis=1)
#print(s2)

# Split test and train variables--------------------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s1,sonuc3,test_size=0.33, random_state=0)


# Modelling------------------------------------------

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
#print(y_pred)
#print(y_test)

boy =s2.iloc[:,3:4].values

left = s2.iloc[:,:3]
right = s2.iloc[:,4:]

veri =pd.concat([left,right],axis=1)

x_train,x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

reg2 = LinearRegression()
reg2.fit(x_train,y_train)

y_pred = reg2.predict(x_test)


# Backward Elimination-----------------------------------------
import statsmodels.api as sm

X = np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()

# pvalue = x5=0.717 4.column elimination
X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()

# pvalue = x5=0.03 maybe 5.column elimination because pvalue<0.05 general acception
X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()







