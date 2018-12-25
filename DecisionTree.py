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
dataset = pd.read_csv('ACIdataAfterSelection.csv',sep=',',header=0, encoding='TIS-620')

#-----------------------------------------------------Factorize Data to float


onehot = dataset.drop(['Account','ACI'], axis=1)
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
dtree = DecisionTreeClassifier(max_depth=100)  
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

'''
fi = dtree.feature_importances_
print(features_name)
print(fi)
'''
#-----------------------------------------------Evaluate
print("Confusion Matrix = ",confusion_matrix(y_test, y_pred))
print("Accuracy Score = ",accuracy_score(y_test, y_pred))

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

#pickle.dump(dtree, open('dtree_model_pickleMD10.p', 'wb'))
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
#---------------------------------------------------SaveModel

#pickle.dump(dtree, open('dtree_model_Test100.p', 'wb'))