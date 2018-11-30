import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import scipy.spatial.distance as sdist
import scipy.spatial.distance as sdist
import math
from sklearn import metrics
df = pd.read_csv('CSVscale01newColumn.csv',sep=',',header=0, encoding='unicode_escape')

#points = df.drop('id', axis=1)
 #or points = df[['Type1', 'Type2', 'Type3']]
kmeans = cluster.KMeans(n_clusters=3, random_state=0).fit(df)
#df['cluster'] = kmeans.labels_ #รวมแล้วมิติไม่เท่ากัน

centroids = kmeans.cluster_centers_



dists = pd.DataFrame(
    sdist.cdist(df, centroids,'euclidean'), 
    columns=['dist_{}'.format(i) for i in range(len(centroids))],
    index=df.index)
#combine = pd.concat([df, dists], axis=1)
pdctr = pd.DataFrame(centroids)
#print(dists)
kmeanlbl = pd.DataFrame((kmeans.labels_),columns=['label'])
#print(combine)
print("_________________________________________________________________________________________")
#print("point = ")
#print(df)
print("_________________________________________________________________________________________")
#print("centroids = ")
#print(centroids)
#combine = combine.to_csv('TestLongKmean.csv', index=False)

pdists = dists.to_csv('newMain_Distance3Cl.csv', index=False)
#pdctr = pdctr.to_csv('nOri_Centroid10Cl.csv', index=False)
pkmeanlbl = kmeanlbl.to_csv('newMain_Label3Cl.csv', index=False)

#_______________________________________________
print("----------------------------------Mean Square--------------------------------------------")
#____________________________________________Select Distance of Cluster in Group
j = [] #Collect all distance for between each cluster point and Centroid
m = 0
n = len(df.index)
i= 1
for i in range(n):
    m = kmeanlbl.label[i]
    
    j.append(dists.iloc[i,m])
ea = j
    
eachalldis = pd.DataFrame((ea),columns=['Distance'])
#-------------------------------------------------------
n_data = len(set(df.index))
#print(n_data)
eachpow =  eachalldis**2
eachnormal = []
eachnormal = abs(eachalldis)
print("Mean Squared Error",eachpow.sum()/n_data)
print("Root Mean Squared Error",math.sqrt(eachpow.sum()/n_data))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df, kmeans.labels_))

print("-----------------------------Finished-------------------------------")
#distandcentr = pd.concat([dists, pdctr], axis = 1)
#distandcentr = distandcentr.to_csv('Kmean5Clustr.csv', index=False)