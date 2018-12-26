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
import pickle
from keras import backend as Tensor
from keras import metrics
from keras.callbacks import History 

#----------------------------------------------------Read CSV File
start_time = time.time()
dataset = pd.read_csv('ACIdataAfterSelection.csv',sep=',',header=0, encoding='TIS-620')

#----------------------------------------------Preprocessing
onehot = dataset.drop(['Account','ACI'], axis=1)
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

model.add(Dense(512, input_dim=onehotCo,init='uniform',activation='relu'))
model.add(Dropout(rate=0.1))       #Use Dropout to prevent Data overfitting

#------------------------------------Create Second Hidden Layer

model.add(Dense(512,init='uniform',activation='relu'))
model.add(Dropout(rate=0.1))       #Use Dropout to prevent Data overfitting


#------------------------------------Output Layer

model.add(Dense(1,init='uniform',activation='sigmoid'))

model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

score_array =[]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)  


model.fit(X_train,y_train,batch_size=64, epochs=1000)
y_pred = model.predict(X_test)
score_array.append(accuracy_score(y_test, y_pred.round()))
k = pd.DataFrame(X_train)
l = pd.DataFrame(y_pred)
'''
jk = k.to_csv('X_train1000.csv', index=False)
jl = l.to_csv('y_pred1000.csv', index=False)
'''
avg_score = np.mean(score_array,axis=0)
each_score = pd.DataFrame(score_array,columns=['Each Round'])
each_score.index = each_score.index+1
print("Accuracy Score For Each Round = ",each_score)
print("Accuracy Mean = ",avg_score)
print("Confusion Matrix = ",confusion_matrix(y_test, y_pred.round()))
print(classification_report(y_test, y_pred.round())) 

#-----------------------------Test Count num
'''
count_row = y_test.shape[0]
print("Total Example : ",count_row)
count_correctclassified = (y_test == y_pred).sum()
print('Correct classified samples: {}'.format(count_correctclassified))
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
'''




#----------------------------------------------------Runtime

print("---Runtime %s seconds ---" % (time.time() - start_time))


# Save the model in h5 format 
model.save('TestModelT1000o512Dense.h5')


print("----------------------------------------------End------------------------------------------")
#------------------------------------------------------