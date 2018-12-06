from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO

import matplotlib.pyplot as plt 
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus

#-----------------------------------------------------Read CSV File
dataset = pd.read_csv('bill_authentication.csv',sep=',',header=0, encoding='TIS-620')

#-----------------------------------------------------Create Train and Test

X = dataset.drop('Class', axis=1)   #without target
y = dataset['Class']                #target
features_name = list(X)
#-----------------------------------------------------Train test split

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  

#test_size 0.2 means ratio of test = 20% of 100%

#-----------------------------------------------------
from sklearn.tree import DecisionTreeClassifier  
dtree = DecisionTreeClassifier()  
dtree.fit(X_train, y_train) 

y_pred = dtree.predict(X_test)  

ans  = pd.DataFrame(y_test)

pans = ans.to_csv('TestAns.csv',index=['index'])
#-------------------------------------------------Plot
dot_data = tree.export_graphviz(dtree,
                                feature_names=features_name,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png('tree.png')


print("end")