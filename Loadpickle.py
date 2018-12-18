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
'''
tn = pickle.load(open('dtree_model_pickle.p','rb'))
print(tn)
'''

from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')

dataset = dataset.drop(['ACI'], axis=1)

nn = pd.get_dummies(dataset)
nns = nn.to_csv('OtestUTF.csv', index=False, encoding='UTF-8')
'''
enc = OneHotEncoder(handle_unknown='ignore')
print(enc.fit(dataset))
#nn = dataset.to_csv('OneHottest.csv', index=False)

'''