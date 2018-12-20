import pandas as pd 
import sklearn
import numpy as np 

dataset = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')

#-----------------------Factorize

dataset.ACI = dataset.ACI.replace("NO",0)
dataset.ACI = dataset.ACI.replace("YES",1)
#print(dataset.ACI)
print(dataset)