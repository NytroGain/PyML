from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score , confusion_matrix , precision_score, recall_score, f1_score, classification_report

from sklearn.externals.six import StringIO

import matplotlib.pyplot as plt 
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import train_test_split

#-----------------------------------------------------Read CSV File
dataset = pd.read_csv('Bill.csv',sep=',',header=0, encoding='TIS-620')

#-----------------------------------------------------Create Train and Test

X = dataset.drop('Class', axis=1)   #without target
y = dataset['Class']                #target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  
#----------------------------------------------------GNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)

#---------------------------------------------------Evaluate
print("Confusion Matrix = ",confusion_matrix(y_test, y_pred))
print("Precision Score = ",precision_score(y_test, y_pred, average=None))
print("Recall Score = ",recall_score(y_test,y_pred, average=None))
print("Accuracy Score = ",accuracy_score(y_test, y_pred))
print("F measure = ",f1_score(y_test, y_pred, average=None))
print("TEST CLASSIFICATION RECORD")
print(classification_report(y_test, y_pred)) 
print("---------------------------------------------End-------------------------------------------------")