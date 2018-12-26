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
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV,cross_val_score
#-----------------------------------------------------Read CSV File
start_time = time.time()
dataset = pd.read_csv('ACIdataAfterSelection.csv',sep=',',header=0, encoding='TIS-620')

#--------------------------------------------------Preprocess OneHot
onehot = dataset.drop(['Account','ACI'], axis=1)
onehot = pd.get_dummies(onehot)
#-----------------------------------------------------Create Train and Test

X = onehot   #without target
y = dataset['ACI']                #target
#features_name = list(X)             #Get name of feature want to show in Tree graph
#-----------------------------------------------------Train test split
kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=False)

score_array =[]
from lightgbm import LGBMClassifier
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    lgbm = LGBMClassifier(objective='binary', random_state='None')
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    print("Confusion Matrix = ",confusion_matrix(y_test, y_pred))
    score_array.append(accuracy_score(y_test, y_pred))

avg_score = np.mean(score_array,axis=0)
each_score = pd.DataFrame(score_array,columns=['Each Round'])
each_score.index = each_score.index+1
print("Accuracy Score For Each Round = ",each_score)
print("Accuracy Mean = ",avg_score)
