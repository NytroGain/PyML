from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score , confusion_matrix , precision_score, recall_score, f1_score, classification_report

from sklearn.externals.six import StringIO

import matplotlib.pyplot as plt 
from sklearn.tree import export_graphviz
import pydotplus
import time
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
#-----------------------------------------------------Read CSV File
start_time = time.time()

dataset = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')

#-----------------------------------------------------Factorize Data to float
nn = dataset.drop(['ACI'], axis=1)
#---------เตรียมข้อมูล-----------
categorical_feature_mask = nn.dtypes==object
categorical_cols = nn.columns[categorical_feature_mask].tolist()
#---------LabelEncoder-----------
le = LabelEncoder()
nn[categorical_cols] = nn[categorical_cols].apply(lambda col: le.fit_transform(col))

#----------OneHot---------
ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False )
ReadyData = ohe.fit_transform(nn)
'''
onehot = dataset.drop(['ACI'], axis=1)
onehot = pd.get_dummies(onehot)
'''
#-----------------------------------------------------Create Train and Test

X = ReadyData   #without target
y = dataset['ACI']                #target
#-----------------------------------------------------Train And Test Split
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

#----------------------------------------------------Feature Scaling
'''sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  '''


#----------------------------------------------------Training and Predictions
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=2)  
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)  

'''predict = pd.DataFrame(classifier.predict(X),columns=['Predict'])
proba = pd.DataFrame(classifier.predict_proba(X))'''

#-----------------------------------------------Evaluate
print("Confusion Matrix = ",confusion_matrix(y_test, y_pred))
print("Precision Score = ",precision_score(y_test, y_pred, average=None))
print("Recall Score = ",recall_score(y_test,y_pred, average=None))
print("Accuracy Score = ",accuracy_score(y_test, y_pred))
print("F measure = ",f1_score(y_test, y_pred, average=None))
print("TEST CLASSIFICATION RECORD")
print(classification_report(y_test, y_pred)) 
count_row = y_test.shape[0]
print("Total Example : ",count_row)
count_correctclassified = (y_test == y_pred).sum()
print('Correct classified samples: {}'.format(count_correctclassified))
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))

#----------------------------------------------------Runtime

print("---Runtime %s seconds ---" % (time.time() - start_time))

print("---------------------------------------------End-------------------------------------------------")