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
j = 0
m = 0
n = len(df.index)
i= 1
for i in range(n):
    m = kmeanlbl[i]
    print(dists.iloc[i,m])

    #j += dists.columns[0:]
 #__________________________________________________
 #------------------------------------Loop To Select cluster and distance
#c0 = []
#c1 = []
#finClus = []
#testmerge = pd.concat([df,kmeanlbl,dists], axis = 1)
#print(testmerge)
#testmerge = testmerge.to_csv('MergeError.csv', index=False)
#for row in testmerge['label']:
 #   if row == 0:
  #      finClus.append([testmerge.dist_0]) #testmerge.dist_0.loc
   # else:
    #    finClus.append([testmerge.dist_1])
#gn = pd.DataFrame((finClus), columns=['distance'])
#print(gn)
#gn = gn.to_csv('GN.csv', index=False)
 #   print(dists.iloc[i])


#dists = dists.to_csv('C_Distance2Cl.csv', index=False)
#pdctr = pdctr.to_csv('C_Centroid2Cl.csv', index=False)
#kmeanlbl = kmeanlbl.to_csv('C_Label2Cl.csv', index=False)
print("-----------------------------Finished-------------------------------")
#distandcentr = pd.concat([dists, pdctr], axis = 1)
#distandcentr = distandcentr.to_csv('Kmean5Clustr.csv', index=False)