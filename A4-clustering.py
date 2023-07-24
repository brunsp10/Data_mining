# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 07:19:03 2023

@author: bruns
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import complete, single, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.spatial import distance
import math
from sklearn.cluster import KMeans

#### 1A single ####

c1_all = open("C:\\Users\\bruns\\Desktop\\Data_Mining\\HW4\\C1.txt")
c1_string = c1_all.readlines()
test = c1_string[0].split("\t")
c1 = []

for string in c1_string:
    nums = []
    string = string.split("\t")
    nums.append(float(string[1]))
    nums.append(float(string[2]))
    c1.append(nums)

y = pdist(c1)
Z = single(y)
fcluster(Z, .8, criterion="distance")

x = []
y = []
for index in c1:
    x.append(index[0])
    y.append(index[1])
    
plt.figure(figsize=(12,8))
plt.scatter(x,y)
plt.scatter(x, y, c = AgglomerativeClustering(n_clusters=3, linkage="complete").fit(c1).labels_,s=250)
plt.title("Figure 1D: Complete-Linkage Clusters Using AgglomerativeClustering")
plt.show()

#### 1A Complete ####

y = pdist(c1)
Z = complete(y)
fcluster(Z,2.1, criterion="distance")


x = []
y = []
for index in c1:
    x.append(index[0])
    y.append(index[1])
    

plt.figure(figsize=(12,8))
plt.scatter(x,y)
plt.scatter(x, y, c = fcluster(Z,2.1, criterion="distance"),s=250)
plt.title("Figure 1B: Complete-Linkage Clusters")
plt.show()

#### 1B ####

AgglomerativeClustering(n_clusters=3, linkage="single").fit(c1).labels_
AgglomerativeClustering(n_clusters=3, linkage="complete").fit(c1).labels_



#### 2A ####
c2_all = open("C:\\Users\\bruns\\Desktop\\Data_Mining\\HW4\\C2.txt")
c2_string = c2_all.readlines()
test = c2_string[0].split("\t")
c2 = []

for string in c2_string:
    nums = []
    string = string.split("\t")
    nums.append(float(string[1]))
    nums.append(float(string[2]))
    c2.append(nums)

x = []
y = []
for index in c2:
    x.append(index[0])
    y.append(index[1])

c2 = [np.array(x) for x in c2]
centroids = []
centroids.append(c2[0])

while len(centroids) < 4:
    furthest = c2[1]
    max_dist = 0
    for x in c2:
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


centroids = [np.array([-2.7694973, 2.6778586]), np.array([-2, 14]),np.array([-0.4032861, -5.4479696]), np.array([4.6102579, 0.6921337])]





assigned_centroids = []

for x in c2:
    min_dist = math.inf
    centroid_index = 5
    for y in range(0,4):
        if distance.euclidean(x, centroids[y]) < min_dist:
            min_dist = distance.euclidean(x, centroids[y])
            centroid_index = y
    assigned_centroids.append(centroid_index)

centers_cost = 0
for index in range(0,len(c2)):
    cost = distance.euclidean(c2[index], centroids[assigned_centroids[index]])
    if cost > centers_cost:
        centers_cost = cost

means_cost = 0

for index in range(0,len(c2)):
    cost = distance.euclidean(c2[index], centroids[assigned_centroids[index]])
    means_cost = means_cost + cost**2
    
math.sqrt((1/len(c2))*means_cost)

x = []
y = []
for index in c2:
    x.append(index[0])
    y.append(index[1])
    
c_x=[]
c_y=[]
for index in centroids:
    c_x.append(index[0])
    c_y.append(index[1])


plt.figure(figsize=(12,8))
plt.scatter(x,y,c=assigned_centroids)
plt.scatter(c_x, c_y, marker="D", s = 100, c="white",edgecolor="black")
plt.title("Figure 2: Results of Gonzalez Clustering. Centroids Shown as White Diamonds")
plt.show()

##### 2B #####
def check_clusters(x, y):
    for index in range(len(x)-1):
        if (x[index] == x[index+1] and y[index] != y[index+1]):
            return False
        elif (x[index] != x[index+1] and y[index] == y[index+1]):
            return False
    return True

kmeans_centers = []
inertias = []
all_assignments=[]
count = 0

for i in range(0,50):
    point_assignments = []
    assignments = {0:[], 1:[], 2:[], 3:[]}
    kmeans = KMeans(n_clusters=4).fit(c2)
    if check_clusters(kmeans.predict(c2), assigned_centroids):
        count = count + 1
    centers = kmeans.cluster_centers_
    kmeans_centers.append(centers)
    for x in c2:
        assigned_center = 5
        min_dist = math.inf
        for y in range(len(centers)):
            if distance.euclidean(x, centers[y]) < min_dist:
                min_dist = distance.euclidean(x, centers[y])
                assigned_center = y
        assignments[assigned_center].append(x)
        point_assignments.append(assigned_center)
    all_assignments.append(point_assignments)
    print(kmeans.inertia_)
    means_cost = 0
    
    for i in range(len(centers)):
        assigned_points = assignments[i]
        means_cost = 0
        dist = 0
        for point in assigned_points:
            dist = 0
            dist = distance.euclidean(point,centers[i])
            means_cost = means_cost + dist**2
            total_cost = math.sqrt((1/len(c2))*means_cost)
    
        inertias.append(total_cost)


x = np.sort(inertias)
y = np.arange(len(inertias))/float(len(inertias))

plt.figure(figsize=(12,8))
plt.xlabel("4-means cost")
plt.ylabel("Cumulative Density")
plt.ticklabel_format(useOffset=False)
plt.plot(x,y)
plt.title("Figure 3: CDF of 4-means cost for 50 runs of k-means++ clustering")
plt.show()

##### 2C.i #####

def lloyd_2(centers, data):
    assignments = {0:[], 1:[], 2:[], 3:[]}
    new_centers = []
    point_assignments = []
    for x in data:
        assigned_center = 5
        min_dist = math.inf
        for y in range(len(centers)):
            if distance.euclidean(x, centers[y]) < min_dist:
                min_dist = distance.euclidean(x, centers[y])
                assigned_center = y
        assignments[assigned_center].append(x)
        point_assignments.append(assigned_center)
    for i in range(len(centers)):
        assigned_points = assignments[i]
        means_cost = 0
        dist = 0
        for point in assigned_points:
            dist = 0
            dist = distance.euclidean(point,centers[i])
            means_cost = means_cost + dist**2
            total_cost = math.sqrt((1/len(c2))*means_cost)
    
        inertias.append(total_cost)
    
    for y in range(len(centers)):
        assigned_points = assignments[y]
        total = np.array((0,0))
        for point in assigned_points:
            total = total + point
        new_centers.append(total/float(len(assigned_points)))
    
    return point_assignments, total_cost, new_centers

lloyd_centers = [c2[0], c2[1], c2[2], c2[3]]


points2 = []
total_cost = 0
count = 0
while count < 100:
    points2, total_cost,lloyd_centers = lloyd_2(lloyd_centers,c2)
    count = count + 1

x = []
y = []
for index in c2:
    x.append(index[0])
    y.append(index[1])
    
c_x=[]
c_y=[]
for index in lloyd_centers:
    c_x.append(index[0])
    c_y.append(index[1])

plt.figure(figsize=(12,8))
plt.scatter(x,y,c=points2)
plt.scatter(c_x, c_y, marker="D", s = 100, edgecolors='black',c="white")
plt.title("Figure 4: Results of Lloyd's Algorithm Using First 4 Points As Initial Clusters. Centroids Shown As White Diamonds")
plt.show()


####### 2.c.ii ############


lloyd_centers_2 = centroids

points2 = []
total_cost = 0
count = 0
while count < 100:
    points2, total_cost,lloyd_centers_2 = lloyd_2(lloyd_centers_2,c2)
    count = count + 1

x = []
y = []
for index in c2:
    x.append(index[0])
    y.append(index[1])
    
c_x=[]
c_y=[]
for index in lloyd_centers_2:
    c_x.append(index[0])
    c_y.append(index[1])

plt.figure(figsize=(12,8))
plt.scatter(x,y,c=points2)
plt.scatter(c_x, c_y, marker="D", s = 100, c='white',edgecolor='black')
plt.title("Figure 5: Results of Lloyd's Algorithm Using Gonzalez Centroids As Initial Clusters. Centroids Shown As White Diamonds")
plt.show()



##### 2.c.iii #####
def lloyd_2(centers, data):
    assignments = {0:[], 1:[], 2:[], 3:[]}
    new_centers = []
    point_assignments = []
    for x in data:
        assigned_center = 5
        min_dist = math.inf
        for y in range(len(centers)):
            if distance.euclidean(x, centers[y]) < min_dist:
                min_dist = distance.euclidean(x, centers[y])
                assigned_center = y
        assignments[assigned_center].append(x)
        point_assignments.append(assigned_center)
    means_cost = 0
    
    for i in range(len(centers)):
        dist = 0
        assigned_points = assignments[i]
        for point in assigned_points:
            dist = dist + distance.euclidean(point,centers[i])
        means_cost = means_cost + dist**2
    total_cost = math.sqrt((1/len(c2))*means_cost)
    
    for y in range(len(centers)):
        assigned_points = assignments[y]
        total = np.array((0,0))
        for point in assigned_points:
            total = total + point
        new_centers.append(total/float(len(assigned_points)))
    
    return point_assignments, total_cost, new_centers

count = 0
all_cost = []
assignments=[]

for i in range(0,len(kmeans_centers)):
    cents = kmeans_centers[i]
    for a in range(0,50):
        assignments, cost, cents = lloyd_2(cents, c2)
    if check_clusters(assignments,all_assignments[i]):
        count = count + 1
    all_cost.append(cost)

x = np.sort(all_cost)
y = np.arange(len(all_cost))/float(len(all_cost))

plt.figure(figsize=(12,8))
plt.xlabel("4-means cost")
plt.ylabel("Cumulative Density")
plt.title("Figure 6: CDF of 4-Means Cost for Lloyd's Algorithm Using K-Means++ Centroids as Starting Points")
plt.plot(x,y)
plt.ticklabel_format(useOffset=False)
plt.show()





