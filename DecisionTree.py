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
'''
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
'''

onehot = dataset.drop(['ACI'], axis=1)
onehot = pd.get_dummies(onehot)
#-----------------------------------------------------Create Train and Test

X = onehot   #without target
y = dataset['ACI']                #target
features_name = list(X)             #Get name of feature want to show in Tree graph
#-----------------------------------------------------Train test split

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  

#test_size 0.2 means ratio of test = 20% of 100%

#-----------------------------------------------------
from sklearn.tree import DecisionTreeClassifier  
dtree = DecisionTreeClassifier()  
dtree.fit(X_train, y_train) 

y_pred = dtree.predict(X_test)  

ans  = pd.DataFrame(y_pred)

#pans = ans.to_csv('TestAnsMD_OH10.csv', header=['Values'])

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
count_row = y_test.shape[0]
print("Total Example : ",count_row)
count_correctclassified = (y_test == y_pred).sum()
print('Correct classified samples: {}'.format(count_correctclassified))
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))

#----------------------------------------------------Runtime

print("---Runtime %s seconds ---" % (time.time() - start_time))

#---------------------------------------------------SaveModel

#pickle.dump(dtree, open('dtree_model_pickleNoMD.p', 'wb'))
'''
print(X_test.shape)
Shapes = pd.DataFrame(X_test)
psha = Shapes.to_csv('X_testColumn.csv',index=False, encoding='UTF-8')
'''
print("---------------------------------------------End-------------------------------------------------")

'''
loaded_model = pickle.load(open('dtree_model_pickle.p', 'rb'))
result = loaded_model.score(xxxxx)
print(result)
'''