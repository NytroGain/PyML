import numpy as np
import pandas as pd
import sklearn.cluster as cluster
from sklearn.metrics import mean_squared_error
import scipy.spatial.distance as sdist

df = pd.read_csv('Main_Edition.csv',sep=',',header=0, encoding='unicode_escape')

#points = df.drop('id', axis=1)
 #or points = df[['Type1', 'Type2', 'Type3']]
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(df)
#df['cluster'] = kmeans.labels_ #รวมแล้วมิติไม่เท่ากัน

centroids = kmeans.cluster_centers_

dists = pd.DataFrame(
    sdist.cdist(df, centroids,'euclidean'), 
    columns=['dist_{}'.format(i) for i in range(len(centroids))],
    index=df.index)
#combine = pd.concat([df, dists], axis=1)
#pdctr = pd.DataFrame(centroids)
#print(dists)
kmeanlbl = pd.DataFrame(kmeans.labels_)
#print(combine)
print("_________________________________________________________________________________________")
#print("point = ")
#print(df)
print("_________________________________________________________________________________________")
#print("centroids = ")
#print(centroids)
#combine = combine.to_csv('TestLongKmean.csv', index=False)

print("----------------------------------Mean Square--------------------------------------------")
j = 0
m = 0
n = len(df.index)
for i in range(5):
    m = kmeanlbl[i]
    print(dists.iloc[i,m])

    #j += dists.columns[0:]
    

#y_true = centroids
#y_predict = dists
#mean_squared_error(y_true, y_predict)


#dists = dists.to_csv('C_Distance2Cl.csv', index=False)
#pdctr = pdctr.to_csv('C_Centroid2Cl.csv', index=False)
#kmeanlbl = kmeanlbl.to_csv('C_Label2Cl.csv', index=False)
print("-----------------------------Finished-------------------------------")
#distandcentr = pd.concat([dists, pdctr], axis = 1)
#distandcentr = distandcentr.to_csv('Kmean5Clustr.csv', index=False)