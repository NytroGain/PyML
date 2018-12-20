import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score , confusion_matrix , precision_score, recall_score, f1_score, classification_report
#-----------------------------------------------------Read CSV File
start_time = time.time()
dataset = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')

#----------------------------------------------Preprocessing
onehot = dataset.drop(['ACI'], axis=1)
onehot = pd.get_dummies(onehot)
#------------------------------------Convert Output to 0,1 in one column
dataset.ACI = dataset.ACI.replace("NO",0)
dataset.ACI = dataset.ACI.replace("YES",1)
#-----------------------------------------------------Create Train and Test

X = onehot   #without target
y = dataset['ACI']                #target

#------------------------------------------------------ Count Columns of input to use in Dense
onehotCo = len(onehot.columns)
#-----------------------------------------------------Create Model
model = Sequential()

#------------------------------------Create First Hidden Layer

model.add(Dense(32, input_dim=onehotCo,init='uniform',activation='relu'))
model.add(Dropout(rate=0.1))       #Use Dropout to prevent Data overfitting

#------------------------------------Create Second Hidden Layer

model.add(Dense(32,init='uniform',activation='relu'))
model.add(Dropout(rate=0.1))       #Use Dropout to prevent Data overfitting

#------------------------------------Output Layer

model.add(Dense(1,init='uniform',activation='sigmoid'))

model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

#-----------------------------------K-Fold Cross Validation
kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=False)

score_array =[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train,y_train,batch_size=10, nb_epoch=10)
    y_pred = model.predict_classes(X_test)
    score_array.append(accuracy_score(y_test, y_pred))

avg_score = np.mean(score_array,axis=0)
each_score = pd.DataFrame(score_array,columns=['Each Round'])
each_score.index = each_score.index+1
print("Accuracy Score For Each Round = ",each_score)
print("Accuracy Mean = ",avg_score)







#----------------------------------------------------Runtime

print("---Runtime %s seconds ---" % (time.time() - start_time))





print("----------------------------------------------End------------------------------------------")
#------------------------------------------------------