import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import scipy.spatial.distance as sdist

df = pd.read_csv('(MainCSV)zrO(1-0)Edition.csv',sep=',', encoding='unicode_escape')
an = pd.DataFrame(df)

kmeans = cluster.KMeans(n_clusters=5, random_state=0).fit(an)
an['cluster'] = kmeans.labels_

centroids = pd.DataFrame(kmeans.cluster_centers_)

#centroids = kmeans.cluster_centers_
#ทามม่ายด้ายยยยยยยยยยย
#dists = pd.DataFrame(
#    sdist.cdist(an, centroids), 
#    columns=['dist_{}'.format(i) for i in range(len(centroids))],
#    index=an.index)
#combine = pd.concat([an, dists], axis=1)
#ฮว้ากกกกกกกกกกกกกกกก
#print(combine)
print("_________________________________________________________________________________________")
print("point = ")
#print(points)
print("_________________________________________________________________________________________")
#print("centroids = ")
#print(centroids)
#combine = combine.to_csv('TestEuZ.csv', index=False)
print(df)
#df.to_csv('TestEuZII.csv', index=False)
print(centroids)
centroids.to_csv('TestCentroid.csv', index=False)