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


df = pd.read_csv('Book2.csv', sep=',', header=0)


pca = PCA(n_components=2).fit(df)
pca_2d = pca.transform(df)

#DBSCAN

db = DBSCAN(eps=0.08, min_samples=3,algorithm='kd_tree', n_jobs=-1).fit(pca_2d)

plt.scatter(pca_2d[:,0],pca_2d[:,1], c=db.labels_, cmap='rainbow') 
plt.show()
#np.savetxt('Book2DBS.txt', db.labels_)
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