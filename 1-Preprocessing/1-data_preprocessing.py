
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Loading data sets----------------------------------------------
veriler = pd.read_csv('data/eksikveriler.csv')

#print(veriler)
#print(eksik_veriler)

#Data preprocessing---------------------------------------------
boy = veriler[['boy']]
#print("boy :",boy)

boy_kilo = veriler[['boy','kilo']]
#print("boy_kilo:",boy_kilo)

#Finding NAN values------------------------------------------------
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
Yas = veriler.iloc[:,1:4].values
#print(Yas)

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
#print(Yas)

#Encoder categoric (Ordinal,nominal) var to Numeric var---------------
ulke = veriler.iloc[:,0:1].values
#print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

#print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
#print(ulke)

# numpy transform to dataframe
sonuc1 = pd.DataFrame(data= ulke, index = range(22), columns = ['fr', 'tr','us'])
print(sonuc1)

sonuc2 = pd.DataFrame(data= Yas, index = range(22), columns = ['boy', 'kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data= cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)

s1=pd.concat([sonuc1,sonuc2],axis=1)
s2=pd.concat([s1,sonuc3],axis=1)

print(s2)

# Split test and train variables--------------------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s1,sonuc3,test_size=0.33, random_state=0)


#Datas Scaler---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


