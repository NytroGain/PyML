import numpy as np
import pandas as pd
import sklearn.cluster as cluster
from sklearn.metrics import mean_squared_error
import scipy.spatial.distance as sdist

df = pd.read_csv('Main01.csv',sep=',',header=0, encoding='unicode_escape')

#points = df.drop('id', axis=1)
 #or points = df[['Type1', 'Type2', 'Type3']]
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(df)
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
j = [] #Collect all distance for between each cluster point and Centroid
m = 0
n = len(df.index)
i= 1
for i in range(n):
    m = kmeanlbl.label[i]
    #print(dists.iloc[i,m])
    j.append(dists.iloc[i,m])
ea = j
    
eachalldis = pd.DataFrame((ea),columns=['each label of cluster'])
eachalldis = eachalldis.to_csv('EachForCluster.csv', index=False)
    #j += dists.columns[0:]
 #__________________________________________________

print("-----------------------------Finished-------------------------------")
#distandcentr = pd.concat([dists, pdctr], axis = 1)
#distandcentr = distandcentr.to_csv('Kmean5Clustr.csv', index=False)