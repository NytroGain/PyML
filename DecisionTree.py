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
import time
import pickle
#-----------------------------------------------------Read CSV File
start_time = time.time()
dataset = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')

#-----------------------------------------------------Factorize Data to float
dataset.REGION = pd.factorize(dataset.REGION)[0]
dataset.AGE = pd.factorize(dataset.AGE)[0]
dataset.YEAR_OF_PRODUCT = pd.factorize(dataset.YEAR_OF_PRODUCT)[0]
dataset.TYPE_PRODUCT = pd.factorize(dataset.TYPE_PRODUCT)[0]
dataset.NEW_USED_ = pd.factorize(dataset.NEW_USED_)[0]
dataset.COM_ROUND = pd.factorize(dataset.COM_ROUND)[0]
dataset.T25_COM_TYPE_COVERAGE = pd.factorize(dataset.T25_COM_TYPE_COVERAGE)[0]
dataset.T25_COM_INS_CODE = pd.factorize(dataset.T25_COM_INS_CODE)[0]
dataset.CLAIM_CON = pd.factorize(dataset.CLAIM_CON)[0]
dataset.INS_PAY_TYPE = pd.factorize(dataset.INS_PAY_TYPE)[0]
dataset.INS_PAY_BY = pd.factorize(dataset.INS_PAY_BY)[0]
dataset.COM_CONFIRM = pd.factorize(dataset.COM_CONFIRM)[0]
#-----------------------------------------------------Create Train and Test

X = dataset.drop(['ACI'], axis=1)   #without target
y = dataset['ACI']                #target
features_name = list(X)             #Get name of feature want to show in Tree graph
#-----------------------------------------------------Train test split

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  

#test_size 0.2 means ratio of test = 20% of 100%

#-----------------------------------------------------
from sklearn.tree import DecisionTreeClassifier  
dtree = DecisionTreeClassifier(max_depth=7)  
dtree.fit(X_train, y_train) 

y_pred = dtree.predict(X_test)  

ans  = pd.DataFrame(y_test)

#pans = ans.to_csv('TestAnsMD10.csv',index=['index'])
cname = ['No','Yes']
#-------------------------------------------------Plot
'''
dot_data = tree.export_graphviz(dtree,
                                feature_names=features_name,
                                out_file=None,
                                filled=True,
                                rounded=True,
                                class_names=dtree.classes_)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png('TesssN13.png')
'''


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
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
#----------------------------------------------------Runtime

print("---Runtime %s seconds ---" % (time.time() - start_time))

#---------------------------------------------------SaveModel

pickle.dump(dtree, open('dtree_model_pickle.p', 'wb'))

print("---------------------------------------------End-------------------------------------------------")