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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
#-----------------------------------------------------Read CSV File
start_time = time.time()
dataset = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')
#-----------------------------------------------------Factorize Data to float

onehot = dataset.drop(['ACI'], axis=1)
onehot = pd.get_dummies(onehot)

#-----------------------------------------------------Create Train and Test

X = onehot   #without target
y = dataset['ACI']                #target

kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=False)

#----------------------------------------------------GNB
'''
gnb = GaussianNB()
score_array =[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf=gnb.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    score_array.append(accuracy_score(y_test, y_pred))

avg_score = np.mean(score_array,axis=0)
each_score = pd.DataFrame(score_array,columns=['Each Round'])
each_score.index = each_score.index+1
'''
#---------------------------------------------------Multinomial
'''
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
score_array =[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf=MNB.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    score_array.append(accuracy_score(y_test, y_pred))
avg_score = np.mean(score_array,axis=0)
each_score = pd.DataFrame(score_array,columns=['Each Round'])
each_score.index = each_score.index+1
'''
#---------------------------------------------------Bernoulli

from sklearn.naive_bayes import BernoulliNB
ber = BernoulliNB()
score_array =[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf=ber.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    score_array.append(accuracy_score(y_test, y_pred))
avg_score = np.mean(score_array,axis=0)
each_score = pd.DataFrame(score_array,columns=['Each Round'])
each_score.index = each_score.index+1

#-----------------------------------------------Evaluate
'''
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
'''

print("Accuracy Score For Each Round = ",each_score)
print("Accuracy Mean = ",avg_score)

#----------------------------------------------------Runtime

print("---Runtime %s seconds ---" % (time.time() - start_time))
print("---------------------------------------------End-------------------------------------------------")