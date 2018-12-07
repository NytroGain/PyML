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

#-----------------------------------------------------Read CSV File
dataset = pd.read_csv('Bill.csv',sep=',',header=0, encoding='TIS-620')

#-----------------------------------------------------Create Train and Test
X = dataset.drop(['Class','Row'], axis=1)   #without target
y = dataset['Class']                #target
z = dataset['Row']
#-----------------------------------------------------Train And Test Split
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y,z, test_size=0.20)  

#----------------------------------------------------Feature Scaling
# sklearn.preprocessing import StandardScaler  
#scaler = StandardScaler()  
#scaler.fit(X_train)

#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test)  


#----------------------------------------------------Training and Predictions
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)  

predict = pd.DataFrame(classifier.predict(X),columns=['Predict'])
proba = pd.DataFrame(classifier.predict_proba(X))

#----------------------------------------------------Evaluate
print("Confusion Matrix = ",confusion_matrix(y_test, y_pred))
print("Precision Score = ",precision_score(y_test, y_pred, average=None))
print("Recall Score = ",recall_score(y_test,y_pred, average=None))
print("Accuracy Score = ",accuracy_score(y_test, y_pred))
print("F measure = ",f1_score(y_test, y_pred, average=None))
print("Test Classification Report")
print(classification_report(y_test, y_pred)) 
print(predict)
print("Proba")
print(proba)
print(X_test)
print(z_test)
#print("--------------------------X_test")
#pa = pd.DataFrame(X_test)
#con = pd.concat([z,pa],axis=1)
#print(con)
#print("---------------------X_train")
#pb = pd.DataFrame(X_train)
#con = pd.concat([z,pb],axis=1)
print("---------------------------------------------End-------------------------------------------------")