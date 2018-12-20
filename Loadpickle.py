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
import keras
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
import h5py
'''
tn = pickle.load(open('dtree_model_pickle.p','rb'))
print(tn)
'''
'''
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')

dataset = dataset.drop(['ACI'], axis=1)
'''
'''
nn = pd.get_dummies(dataset)
nns = nn.to_csv('OtestUTF.csv', index=False, encoding='UTF-8')
'''
'''
enc = OneHotEncoder(sparse=False)
nd = enc.fit_transform(dataset)
print(nd)
'''
''''
nd = pd.DataFrame(test)
na = nd.to_csv('OnTest.csv', index=False)
'''
#nn = dataset.to_csv('OneHottest.csv', index=False)
from keras.models import Sequential
model = Sequential()
loaded_model=model.load_weights('Neural_2.h5')
print(loaded_model)