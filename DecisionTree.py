from sklearn import tree
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
dataset = pd.read_csv('ForDecisionNC02.csv',sep=',',header=0, encoding='TIS-620')

#-----------------------------------------------------Factorize Data to float
dataset.SEX = pd.factorize(dataset.SEX)[0]
dataset.STATUS = pd.factorize(dataset.STATUS)[0]
dataset.AGE = pd.factorize(dataset.AGE)[0]
dataset.BRA_BRAND = pd.factorize(dataset.BRA_BRAND)[0]
dataset.TYPE_PRODUCT = pd.factorize(dataset.TYPE_PRODUCT)[0]
dataset.NEW_USED_ = pd.factorize(dataset.NEW_USED_)[0]



#-----------------------------------------------------Create Train and Test

X = dataset.drop(['ACI','OCCUP_SUB'], axis=1)   #without target
y = dataset['ACI']                #target
features_name = list(X)             #Get name of feature want to show in Tree graph
#-----------------------------------------------------Train test split

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  

#test_size 0.2 means ratio of test = 20% of 100%

#-----------------------------------------------------
from sklearn.tree import DecisionTreeClassifier  
dtree = DecisionTreeClassifier(max_depth=5)  
dtree.fit(X_train, y_train) 

y_pred = dtree.predict(X_test)  

ans  = pd.DataFrame(y_test)

pans = ans.to_csv('TestAnsMD5.csv',index=['index'])
#-------------------------------------------------Plot
dot_data = tree.export_graphviz(dtree,
                                feature_names=features_name,
                                out_file=None,
                                filled=True,
                                rounded=True,
                                class_names=['No','Yes'])
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png('treeDACIMD5T.png')


fi = dtree.feature_importances_
print(features_name)
print(fi)
#-----------------------------------------------Evaluate
print("Confusion Matrix = ",confusion_matrix(y_test, y_pred))
print("Precision Score = ",precision_score(y_test, y_pred, average=None))
print("Recall Score = ",recall_score(y_test,y_pred, average=None))
print("Accuracy Score = ",accuracy_score(y_test, y_pred))
print("F measure = ",f1_score(y_test, y_pred, average=None))
print("TEST CLASSIFICATION RECORD")
print(classification_report(y_test, y_pred)) 
print("---------------------------------------------End-------------------------------------------------")