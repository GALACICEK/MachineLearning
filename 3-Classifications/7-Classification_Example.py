import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, classification_report

#Loading data sets----------------------------------------------
#veriler = pd.read_csv('data/veriler.csv')
veriler = pd.read_excel('data/Iris.xls')

# Label Encoding----------------------------------------------
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
veriler['Encoded_Species'] = label_encoder.fit_transform(veriler['iris'])

X = veriler.iloc[:, :2].values #independent variables
y= veriler.iloc[:,-1].values #dependent variable

x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5

# 2D scatter plot 
plt.figure(2, figsize=(8,6))
plt.clf()

try:
#Plot the training points
    plt.scatter(X[:,0] , X[:,1], c=y, cmap=plt.cm.Set1, edgecolors='k')
except (TypeError, ValueError) as err:
    print(f"Hata: {err}")
plt.title('Scatter Plot with Color for Species')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')


plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# 3D scatter plot 
fig = plt.figure(1,figsize=(8,6))
ax = fig.add_subplot(projection='3d')
#ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(veriler.iloc[:,0], veriler.iloc[:,1], veriler.iloc[:,2], c=y,
           cmap='Set1', edgecolor='k', s=40)

ax.set_title('IRIS DataSet')
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_zlabel('Petal Length')

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.show()


# Split test and train variables--------------------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=0)


#Datas Scaler---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Logistic Regression---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred_logr = logr.predict(X_test)


#Logistic Regression Calculation Metrics ---------------------------------------------------------------------------
cm = confusion_matrix(y_test,y_pred_logr)
print("Logistik Regression Confuse Matrix")
print(cm)


recall = recall_score(y_test, y_pred_logr, average='weighted')
f1score = f1_score(y_test,y_pred_logr,average='weighted')
report = classification_report(y_test, y_pred_logr, target_names=["Class 0", "Class 1", "Class 2"])


print(f'Accuracy : {accuracy_score(y_test, y_pred_logr)}\n Recall : {recall} \n F1_score : {f1score} \n Classification Report:\n {report}')



#KNN ----------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=11,weights='distance',metric='minkowski')
knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_test)

# KNN Calculation Metrics---------------------------------------------------------------------------

cm = confusion_matrix(y_test,y_pred_knn)
print("KNN Confuse Matrix")
print(cm)

recall = recall_score(y_test, y_pred_knn, average='weighted')
f1score = f1_score(y_test,y_pred_knn,average='weighted')
report = classification_report(y_test, y_pred_knn, target_names=["Class 0", "Class 1", "Class 2"])

print(f'Accuracy : {accuracy_score(y_test, y_pred_logr)}\n Recall : {recall} \n F1_score : {f1score} \n Classification Report:\n {report}')


#SVC kernel='rbf'----------------------------------------------------------------------------
from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred_svc = svc.predict(X_test)

#SVC Calculation Metrics---------------------------------------------------------------------------

cm = confusion_matrix(y_test,y_pred_svc)
print("SVC rbf Confuse Matrix \n",cm)

recall = recall_score(y_test, y_pred_svc, average='weighted')
f1score = f1_score(y_test,y_pred_svc,average='weighted')
report = classification_report(y_test, y_pred_svc, target_names=["Class 0", "Class 1", "Class 2"])

print(f'Accuracy : {accuracy_score(y_test, y_pred_svc)}\n Recall : {recall} \n F1_score : {f1score} \n Classification Report:\n {report}')


#Gaussian Naive Bayes----------------------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred_gnb = gnb.predict(x_test)

#Gaussian Naive Bayes Calculation Metrics---------------------------------------------------------------------------

cm_gnb = confusion_matrix(y_test,y_pred_gnb)
print('GNB Confuse Matrix')
print(cm_gnb)

recall = recall_score(y_test, y_pred_gnb, average='weighted')
f1score = f1_score(y_test,y_pred_gnb,average='weighted')
report = classification_report(y_test, y_pred_gnb, target_names=["Class 0", "Class 1", "Class 2"])

print(f'Accuracy : {accuracy_score(y_test, y_pred_gnb)}\n Recall : {recall} \n F1_score : {f1score} \n Classification Report:\n {report}')

#Decision Tree Classifier entropy----------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)

y_pred_dtc = dtc.predict(X_test)

#Decision Tree Classifier Calculation Metrics---------------------------------------------------------------------------
cm_dtc = confusion_matrix(y_test,y_pred_dtc)
print('DTC Entropy Confuse Matrix')
print(cm_dtc)

recall = recall_score(y_test, y_pred_dtc, average='weighted')
f1score = f1_score(y_test,y_pred_dtc,average='weighted')
report = classification_report(y_test, y_pred_dtc, target_names=["Class 0", "Class 1", "Class 2"])

print(f'Accuracy : {accuracy_score(y_test, y_pred_dtc)}\n Recall : {recall} \n F1_score : {f1score} \n Classification Report:\n {report}')


#Random Forest Classifier entropy----------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier 

rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train,y_train)

y_pred_rfc = rfc.predict(X_test)

from sklearn import metrics
y_proba = rfc.predict_proba(X_test)
fpr, tpr, thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')

#Random Forest Classifier Calculation Metrics---------------------------------------------------------------------------
cm_rfc = confusion_matrix(y_test,y_pred_rfc)
print('RFC Entropy Confuse Matrix')
print(cm_rfc)

recall = recall_score(y_test, y_pred_rfc, average='weighted')
f1score = f1_score(y_test,y_pred_rfc,average='weighted')
report = classification_report(y_test, y_pred_rfc, target_names=["Class 0", "Class 1", "Class 2"])

print(f'Accuracy : {accuracy_score(y_test, y_pred_rfc)}\n Recall : {recall} \n F1_score : {f1score} \n Classification Report:\n {report}')


