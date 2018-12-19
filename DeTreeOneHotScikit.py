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
dataset = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')

#--------------------------------------------------Preprocess OneHot
onehot = dataset.drop(['ACI'], axis=1)
onehot = pd.get_dummies(onehot)
#-----------------------------------------------------Create Train and Test

X = onehot   #without target
y = dataset['ACI']                #target
#features_name = list(X)             #Get name of feature want to show in Tree graph
#-----------------------------------------------------Train test split

'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  

#test_size 0.2 means ratio of test = 20% of 100%
'''
kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=False)

#-----------------------------------------------------

dtree = DecisionTreeClassifier(max_depth=10)  
score_array =[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf=dtree.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    score_array.append(accuracy_score(y_test, y_pred))

avg_score = np.mean(score_array,axis=0)

print("Accuracy Score For Each Round = ",score_array)
print("Accuracy Mean = ",avg_score)
#dtree.fit(X_train, y_train) 
'''
y_pred = dtree.predict(X_test)  

ans  = pd.DataFrame(y_pred)

#pans = ans.to_csv('TestAnsMD_OH10.csv', header=['Values'])
'''
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