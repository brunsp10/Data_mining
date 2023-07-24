# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 12:09:21 2023

@author: bruns
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np

from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import scale


# Import data
cancer = pd.read_csv("C:\\Users\\bruns\\Desktop\\Data_Mining\\Project_messing\\cancer_gene_expression.csv")
data = cancer.iloc[:,2:]

all_results=pd.DataFrame(cancer['type'])

# Cluster on all
k_all=KMeans(n_clusters=5).fit(data).labels_
a_all=AgglomerativeClustering(n_clusters=5).fit(data).labels_

all_results['kc_all']=k_all
all_results['ac_all']=a_all

k_plot = []
inertia_plot = []



def calculate_error(labels, centers, samples):
    output = 0
    samples_array=samples.to_numpy()
    for i in range(0,len(samples)):
        c = labels[i]
        output = output + (np.linalg.norm(centers[c]-samples_array[i]))**2
    return math.sqrt(output * 1/len(labels))

# PCA
data = cancer.iloc[:,2:]
data.head()
X = scale(data)
y= cancer.iloc[:,1]

# apply PCA
pca = decomposition.PCA(n_components=.8)
X = pca.fit_transform(X)

loadings = pd.DataFrame(pca.components_.T *  np.sqrt(pca.explained_variance_), index=cancer.columns[2:])
loadings = loadings.abs()
loadings
loadings.sort_values(by = [0],ascending=False).head(5).iloc[:,0]

# Cluster on PCA 5 genes
data_5 = cancer[['38447_at', '1561171_a_at', '1554616_at', '1556507_at', '1569522_at']]
kc_5=KMeans(n_clusters=5).fit(data_5).labels_
ac_5=AgglomerativeClustering(n_clusters=5).fit(data_5).labels_

all_results['ac_5']=ac_5
all_results['kc_5']=kc_5

# Cluster on PCA 10 genes
data_10 = cancer[['38447_at', '1561171_a_at', '1554616_at', '1556507_at', '1569522_at',
                  '1552915_at', '1552863_a_at', '1570266_x_at', '1560631_at', '216535_at']]
kc_10=KMeans(n_clusters=5).fit(data_10).labels_
ac_10=AgglomerativeClustering(n_clusters=5).fit(data_10).labels_
all_results['ac_10']=ac_10
all_results['kc_10']=kc_10

# Cluster on PCA 20 genes
data_20 = cancer[['38447_at', '1561171_a_at', '1554616_at', '1556507_at', '1569522_at',
                  '1552915_at', '1552863_a_at', '1570266_x_at', '1560631_at', '216535_at',
                  '237425_at', '1557841_at', '1562256_at', '216849_at', '1566923_at', '1561133_at',
                  '1556374_s_at', '215484_at', '1568613_at', '1559167_x_at']]
kc_20=KMeans(n_clusters=7).fit(data).labels_
ac_20=AgglomerativeClustering(n_clusters=5).fit(data_20).labels_
all_results['ac_20']=ac_20
all_results['kc_7']=kc_20


types = cancer["type"]

test2=np.array([types,kc_5,kc_10,kc_20, k_all, ac_5, ac_10, ac_20, a_all])

test2=test2.T

def calculate_error_max(labels, centers, samples):
    output = 0
    samples_array=samples.to_numpy()
    for i in range(0,len(samples)):
        c=labels[i]
        distance = (np.linalg.norm(centers[c]-samples_array[i]))**2
        if distance > output:
            output = distance
    return output


test_labels=KMeans(n_clusters=7).fit(data).labels_
test_centers=KMeans(n_clusters=7).fit(data).cluster_centers_
to_test=data

calculate_error(test_labels, test_centers, to_test)


data = data_10.to_numpy();

data = [x/np.linalg.norm(x) for x in data]

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarities = cosine_similarity(data)





#%% Gonzalez

from scipy.spatial import distance
import random

opt_centroids = []
opt_cost = 1_000_000_000_000_000_000

data = np.array(data_20)

for q in range(0,100):
    rand_i = random.randint(0,129)
    centroids = []
    centroids.append(data[rand_i])
    
    while len(centroids) < 5:
        furthest = data[1]
        max_dist = 0
        for x in data:
            sum = 0
            for y in centroids:
                if (distance.euclidean(x,y) < 0.1):
                    sum = 0
                    break
                sum = sum + distance.euclidean(x,y)
            if sum > max_dist:
                max_dist = sum
                furthest = x
        centroids.append(furthest)
    _centroids = []
    assigned_centroids = []
    
    for x in data:
        min_dist = math.inf
        centroid_index = 5
        for y in range(0,5):
            if distance.euclidean(x, centroids[y]) < min_dist:
                min_dist = distance.euclidean(x, centroids[y])
                centroid_index = y
        assigned_centroids.append(centroid_index)
    
    means_cost = 0
    
    for index in range(0,len(data)):
        cost = distance.euclidean(data[index], centroids[assigned_centroids[index]])
        means_cost = means_cost + cost**2
        
    means_cost=math.sqrt((1/len(data))*means_cost)
    if means_cost < opt_cost:
        opt_cost=means_cost
        opt_centroids=assigned_centroids

all_results['gc_20']=opt_centroids

all_results.to_csv("C:\\Users\\bruns\\Desktop\\Data_Mining\\Project_messing\\resultsertvrtwerct.csv")
