from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import mean_squared_error

#-----------------------------Load File-------------------------------------
df = pd.read_csv('Book2.csv', sep=',', header=0)


#-----------------------------PCA ------------------------------------------
pca = PCA(n_components=9).fit(df)
pca_2d = pca.transform(df)

#------------------------------K-mean
kmeans = KMeans(n_clusters=9, random_state=0).fit(pca_2d)

#y_true = kmeans.cluster_centers_
#y_pred = kmeans
#mean_squared_error(y_true, y_pred, multioutput='raw_value')

#print("Mean Square Error = ", mean_squared_error)
#-----------------------------DBSCAN-----------------------------------------

#db = DBSCAN(eps=0.08, min_samples=3,algorithm='kd_tree', n_jobs=-1).fit(pca_2d)

#----------------------------Plot Graph--------------------------------------
#plt.scatter(pca_2d[:,0],pca_2d[:,1], c=db.labels_, cmap='rainbow') 
#plt.show()


#df['Cluster'] = db.labels_
#df.to_csv('TestBook2.csv', index=False)
#np.savetxt('Book2DBS.txt', db.labels_)

#------------------------------Evaluate---------------------------------------
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(df, db.labels_))                            #ทำได้แต่ 1 มิติ
#print("Completeness: %0.3f" % metrics.completeness_score(df, db.labels_))                          #ทำได้แต่ 1 มิติ
#print("V-measure: %0.3f" % metrics.v_measure_score(df, db.labels_))                                #ทำได้แต่ 1 มิติ
#print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(df, db.labels_))                  #ทำได้แต่ 1 มิติ
#print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(df, db.labels_))   #ทำได้แต่ 1 มิติ
#print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_2d, kmeans.labels_))

#---------------------------

print (pca.explained_variance_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())


print('----------------------------------------END DBSCAN----------------------------------------')
#-----Birch
#brc = Birch(branching_factor=2000, n_clusters=None, threshold=0.5,
#compute_labels=True)
#brc.fit(df) 


#brc.predict(df)
#----------------------------------------------------
#MS
#clustering = MeanShift(bandwidth=5).fit(df)

#clustering.labels_
#clustering 
#------------------------------------------------------
#Spectral                           Memory Error
#clustering = SpectralClustering(n_clusters=2,
    #    assign_labels="discretize",
    #    random_state=0).fit(df)
#clustering.labels_

#clustering 
#----------------------------------
#Agglo                              Memory Error
#clustering = AgglomerativeClustering().fit(df)
#clustering 
#clustering.labels_


#plt.scatter(pca_2d[:,0],pca_2d[:,1], c=db.labels_, cmap='rainbow') 
#plt.show()