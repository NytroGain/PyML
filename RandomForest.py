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
from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.model_selection import StratifiedKFold
#-----------------------------------------------------Read CSV File
start_time = time.time()
dataset = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')

#--------------------------------------------------Preprocess OneHot
onehot = dataset.drop(['ACI'], axis=1)
onehot = pd.get_dummies(onehot)
#-----------------------------------------------------Create Train and Test

X = onehot   #without target
y = dataset['ACI']                #target
#features_name = list(X)             #Get name of feature want to show in Tree graph


#-----------------------------------------------------Train test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#-----------------------------------------------------Random Forest
from sklearn.ensemble import RandomForestClassifier

rafo = RandomForestClassifier(n_estimators=500,class_weight='balanced')
skf = StratifiedKFold(n_splits=10, shuffle=False)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    rafo.fit(X_train,y_train)
    y_pred=rafo.predict(X_test)
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

pickle.dump(rafo, open('RandomForest500.p', 'wb'))