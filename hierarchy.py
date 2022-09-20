# -*- coding: utf-8 -*-
"""
hircarchical clustering practice
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv("C:\\Users\\SAI\\Desktop\\practice\\Wholesale customers data.csv")
data.head()

#lets normalize the data, its nessesary because in the data some feature are with the hight
#magnitude value, that feature will affect to data and model is getting baise toward 
#that feature so first we have to normalize the data in one scale

from sklearn.preprocessing import normalize
data_scale = normalize(data)
data_scale = pd.DataFrame(data_scale, columns= data.columns)
data_scale.head()

#now the data is in the scale now we have to create the dendogram figure which we can choose the 
#how many group or cluster we need to select

import scipy.cluster.hierarchy as sch
plt.figure(figsize = (10,7))
plt.title("dendogram")
dend = sch.dendrogram(sch.linkage(data_scale, method = 'ward'))

#x axis contains the samples and y axis contains the difference of sample
#the vertical line with the maximum distance is blue line so we decide to our threshold vlaue is 6
#cut from 6 a threshold line

plt.figure(figsize = (10,7))
plt.title("dendogram")
dend = sch.dendrogram(sch.linkage(data_scale,method='ward'))
plt.axhline(y=6,color='r',linestyle='--')

#we have two cluster as this line cuts the dendrogram at two points. now lets apply hierarchical clustering
#with two cluster

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 2,affinity = 'euclidean',linkage = 'ward')
predict = cluster.fit_predict(data_scale)

#this is all about hierarchical clustering