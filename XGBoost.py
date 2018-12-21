import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score , confusion_matrix , precision_score, recall_score, f1_score, classification_report
import pickle

from keras.callbacks import History 
from sklearn.model_selection import StratifiedKFold
#----------------------------------------------------Read CSV File
start_time = time.time()
dataset = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')

#----------------------------------------------Preprocessing
onehot = dataset.drop(['ACI'], axis=1)
onehot = pd.get_dummies(onehot)

#-----------------------------------------------------Create Train and Test

X = onehot   #without target
y = dataset['ACI']                #target
features_name = list(X)             #Get name of feature want to show in Tree graph

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error

model = XGBClassifier()
model=XGBClassifier(learning_rate=0.1,n_estimators=100)

skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train,y_train)
y_pred=model.predict(X_test)

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