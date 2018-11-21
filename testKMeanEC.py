import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import scipy.spatial.distance as sdist

df = pd.DataFrame({'Type1': [0.0, 0.0, 473.6, 0.0, 0.0, 514],
 'Type2': [0.0, 63.72, 174.0, 996.0, 524.91, 221],
 'Type3': [0.0, 0.0, 31.6, 160.92, 0.0, 351],
 'Type4': [0.0, 76.0, 21.6, 60.92, 20.0, 20],
 'Type5': [0.0, 51.0, 74.3, 10.92, 10.0, 31],
 'id': [1000, 10001, 10002, 10003, 10004, 10005]})

points = df.drop('id', axis=1)
# or points = df[['Type1', 'Type2', 'Type3']]
kmeans = cluster.KMeans(n_clusters=5, random_state=0).fit(points)
df['cluster'] = kmeans.labels_

centroids = kmeans.cluster_centers_
dists = pd.DataFrame(
    sdist.cdist(points, centroids,'euclidean'), 
    columns=['dist_{}'.format(i) for i in range(len(centroids))],
    index=df.index)
combine = pd.concat([df, dists], axis=1)

print(combine)
print("_________________________________________________________________________________________")
print("point = ")
print(points)
print("_________________________________________________________________________________________")
print("centroids = ")
print(centroids)
#combine = combine.to_csv('TestEuZ.csv', index=False)