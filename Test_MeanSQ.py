import numpy as np
import pandas as pd
import sklearn.cluster as cluster
from sklearn.metrics import mean_squared_error
import scipy.spatial.distance as sdist
import math
from sklearn import metrics


df = pd.read_csv('CSVscale01newColumn.csv',sep=',',header=0, encoding='unicode_escape')

#points = df.drop('id', axis=1)
 #or points = df[['Type1', 'Type2', 'Type3']]
kmeans = cluster.KMeans(n_clusters=10, random_state=0).fit(df)
#df['cluster'] = kmeans.labels_ #รวมแล้วมิติไม่เท่ากัน

centroids = kmeans.cluster_centers_

#----------------------------------Loop to calculate Distance between Cluster poiint and all centroid
dists = pd.DataFrame(
    sdist.cdist(df, centroids,'euclidean'), 
    columns=['dist_{}'.format(i) for i in range(len(centroids))],
    index=df.index)
#combine = pd.concat([df, dists], axis=1)
#pdctr = pd.DataFrame(centroids)
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
#eachalldis = eachalldis.to_csv('EachForCluster.csv', index=False)

#___________________________________________Show All Cluster Group
#n_clusters_ = len(set(kmeanlbl.label))  - (1 if -1 in kmeanlbl.label else 0)
#print('Estimated number of clusters: %d' % n_clusters_)
#print(eachalldis)
#___________________________________________n data
n_data = len(set(df.index))
#print(n_data)
eachpow =  eachalldis**2
eachnormal = []
eachnormal = abs(eachalldis)
#print("Mean Squared Error",eachpow.sum()/n_data)
#print("Root Mean Squared Error",math.sqrt(eachpow.sum()/n_data))
#print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df, kmeans.labels_))
#print("Mean Absolute Error",eachnormal/n_data)
print(df.describe())


 #__________________________________________________

print("-----------------------------Finished-------------------------------")
#distandcentr = pd.concat([dists, pdctr], axis = 1)
#distandcentr = distandcentr.to_csv('Kmean5Clustr.csv', index=False)