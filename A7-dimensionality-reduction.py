# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:00:25 2023

@author: bruns
"""

import sklearn.datasets as dt
import random
random.seed(75)
import matplotlib.pyplot as plt
import pandas as pd


def prepare_data_problem_1():    
    # Downloads from https://www.gapminder.org/data/
    cm_path = 'C:\\Users\\bruns\\Desktop\\Data_Mining\\HW7\\child_mortality_0_5_year_olds_dying_per_1000_born-1.csv'
    fe_path = 'C:\\Users\\bruns\\Desktop\\Data_Mining\\HW7\\children_per_woman_total_fertility-1.csv'
    cm = pd.read_csv(cm_path).set_index('country')['2017'].to_frame()/10
    fe = pd.read_csv(fe_path).set_index('country')['2017'].to_frame()
    child_data = cm.merge(fe, left_index=True, right_index=True).dropna()
    child_data.columns = ['mortality', 'fertility']
    child_data.head()
    print (child_data)

    return child_data

data = prepare_data_problem_1()

### 1i ###
from numpy import linalg as LA
import numpy as np

data = np.array(data)

means = data.mean(axis = 0)

data_centered = data - means

u, s, vT = LA.svd(data_centered)

plt.scatter(data_centered[:,0], data_centered[:,1])
origin = np.array([[0,0],[0,0]])
plt.quiver(*origin,vT.T[0], vT.T[1], scale = 5.5, color = ['red', 'orange'])
plt.title("Cenetered Fertility vs Child Mortality for 186 Countires in 2017")
plt.xlabel("Child Mortality")
plt.ylabel("Fertility")
plt.show()

### 1ii ###

from sklearn.decomposition import TruncatedSVD

truncate = TruncatedSVD(n_components=1)
approx_data = truncate.fit_transform(data_centered)

approx_data = truncate.inverse_transform(approx_data)

def joint_scatter_plot(data, approx_data):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(data[:,0],data[:,1], color='b')
    ax1.scatter(approx_data[:,0],approx_data[:,1], color='r')
    plt.title("Cenetered Fertility vs Child Mortality for 186 Countires in\n2017 with Best Rank-1 Approximation")
    plt.xlabel("Child Mortality")
    plt.ylabel("Fertility")
    plt.show()

joint_scatter_plot(data_centered, approx_data)



### 2 ###
from matplotlib import pyplot as plt

def prepare_data_problem_2():
    '''
        Fetch and downsample RCV1 dataset to only 500 points.
        https://scikit-learn.org/stable/datasets/real_world.html#rcv1-dataset 
    '''
    rcv1 = dt.fetch_rcv1()

    # Choose 500 samples randomly
    sample_size = 500
    row_indices = random.sample(list(range(rcv1.data.shape[0])),sample_size)
    data_sample = rcv1.data[row_indices,:]

    print(f'Shape of the input data: {data_sample.shape}') # Should be (500, 47236)
    return data_sample

data_2 = prepare_data_problem_2()

epsilons = np.arange(0.1, 0.99, 0.2)

from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.metrics.pairwise import euclidean_distances

johnson_lindenstrauss_min_dim(500, eps=epsilons)

dists = euclidean_distances(data_2)
dists_vec = []

for i in range(0,499):
    for j in range(i+1, 500):
        dists_vec.append(dists[i][j])

### 2a ###
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.metrics.pairwise import euclidean_distances

# Get the data
data_2 = prepare_data_problem_2()

# Check if epsilon values make sense
epsilons = np.arange(0.1, 0.99, 0.2)
johnson_lindenstrauss_min_dim(500, eps=epsilons)
# Output was [5326,  690,  298,  190,  153] so
# they should all be good

# Make a vector of pairwise distances
dists = euclidean_distances(data_2)
dists_vec = []
for i in range(0,499):
    for j in range(i+1, 500):
        dists_vec.append(dists[i][j])

# The following 3 variables are used for storing
# values and plotting data
ep_plot = []
mean_plot = []
dist_dict = {}
for ep in epsilons:
    dist_dict[str(ep)] = []

# For each value of epsilon, transform the data
# and make a vector of the new pairwise distances
for ep in epsilons:
    transformer = GaussianRandomProjection(eps=ep)
    data_t = transformer.fit_transform(data_2)
    dists_t = euclidean_distances(data_t)
    dists_vec_t = []
    for i in range(0,499):
        for j in range(i+1, 500):
            dists_vec_t.append(dists_t[i][j])
            
    # Keep track of all the pairwise distances for
    # each value for epsilon
    for i in range(0,len(dists_vec)):
        dist_dict[str(ep)].append(np.abs(dists_vec[i]-dists_vec_t[i]))
    ep_plot.append(ep)
    mean_plot.append(np.mean(dist_dict[str(ep)]))
    
    # Make a histogram for the differences for each
    # individual value of epsilon
    plt.title("Histogram of Differences when Epsilon = %.2f" % ep)
    plt.xlabel("Difference in Distance")
    plt.ylabel('Count')
    plt.hist(dist_dict[str(ep)], bins = 15)
    plt.show()

# Plot mean absolute difference versus epsilon
plt.title("Mean Absolute Difference vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel('Mean Absolute Difference')
plt.plot(ep_plot, mean_plot)
plt.show()

### 2b ###
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.metrics.pairwise import euclidean_distances

# Get the data
data_2 = prepare_data_problem_2()

# Check if epsilon values make sense
epsilons = np.arange(0.1, 0.99, 0.2)
johnson_lindenstrauss_min_dim(500, eps=epsilons)
# Output was [5326,  690,  298,  190,  153] so
# they should all be good

# Make a vector of pairwise distances
dists = euclidean_distances(data_2)
dists_vec = []
for i in range(0,499):
    for j in range(i+1, 500):
        dists_vec.append(dists[i][j])

# The following 3 variables are used for storing
# values and plotting data
ep_plot = []
mean_plot = []
dist_dict = {}
for ep in epsilons:
    dist_dict[str(ep)] = []

# For each value of epsilon, transform the data
# and make a vector of the new pairwise distances
for ep in epsilons:
    transformer = SparseRandomProjection(eps=ep)
    data_t = transformer.fit_transform(data_2)
    dists_t = euclidean_distances(data_t)
    dists_vec_t = []
    for i in range(0,499):
        for j in range(i+1, 500):
            dists_vec_t.append(dists_t[i][j])
            
    # Keep track of all the pairwise distances for
    # each value for epsilon
    for i in range(0,len(dists_vec)):
        dist_dict[str(ep)].append(np.abs(dists_vec[i]-dists_vec_t[i]))
    ep_plot.append(ep)
    mean_plot.append(np.mean(dist_dict[str(ep)]))
    
    # Make a histogram for the differences for each
    # individual value of epsilon
    plt.title("Histogram of Differences when Epsilon = %.2f" % ep)
    plt.xlabel("Difference in Distance")
    plt.ylabel('Count')
    plt.hist(dist_dict[str(ep)], bins = 15)
    plt.show()

# Plot mean absolute difference versus epsilon
plt.title("Mean Absolute Difference vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel('Mean Absolute Difference')
plt.plot(ep_plot, mean_plot)
plt.show()