# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:18:37 2023

@author: bruns
"""
import numpy as np

filename = 'C:\\Users\\bruns\\Desktop\\Data_Mining\\HW8\\web-Google_10k.txt'
with open(filename,'r') as input_file: 
    # The first 4 lines are metadata about the graph that you do not need 
    # After the metadata, the next lines are edges given in the format: `node1\tnode2\n` where node1 points to node2
    lines = [item.replace('\n','').split('\t') for item in input_file] 
    edges = [[int(item[0]),int(item[1])] for item in lines[4:]]

    nodes_with_duplicates = [node for pair in edges for node in pair]
    nodes = sorted(set(nodes_with_duplicates))

    # There are 10K unique nodes, but the nodes are not numbered from 0 to 10K!!! 
    # E.g. there is a node with the ID 916155 
    # So you might find these dictionaries useful in the rest of the assignment
    node_index = {node: index for index, node in enumerate(nodes)}
    index_node = {index: node for node, index in node_index.items()}

edges_as_index = []

for pair in edges:
    out_index = node_index[pair[0]]
    in_index = node_index[pair[1]]
    edges_as_index.append([out_index, in_index])

#### 1A ####
# Make a dictionary to hold counts
edges_out = {}

# Loop over edges and count occurences of origin node
for pair in edges:
    out_node = pair[0]
    if out_node not in edges_out.keys():
        edges_out[out_node] = 1
    else:
        edges_out[out_node]+=1
        
#### 1B ####
# Make a set of all nodes
dead_ends = set(node_index.keys())

# If a node has any out-edges, remove it
# from the set
for node in edges_out.keys():
    dead_ends.remove(node)

dead_ends = list(dead_ends)
print(len(dead_ends))

#### 1C ####

# Convert dead-ends to indices
dead_end_indices = set({})
for node in dead_ends:
    dead_end_indices.add(node_index[node])

# Convert edges to index values
endpoints_of_indices = {}
for pair in edges_as_index:
    if pair[0] not in endpoints_of_indices.keys():
        endpoints_of_indices[pair[0]] = set({pair[1]})
    else:
        endpoints_of_indices[pair[0]].add(pair[1])

# Make an array of degrees for indices
M = np.zeros((10_000,10_000))
for j in range(0,10_000):
    # Set values for dead-end indices to uniform
    if j in dead_end_indices:
        for i in range(0,10_000):
            M[i, j] = 1/10_000
    else:
        for i in range(0,10_000):
            if i in endpoints_of_indices.keys() and j in endpoints_of_indices[i]:
                M[j, i] = 1/len(endpoints_of_indices[i])
            else:
                pass

# Initialize r
r = [1/10_000 for i in range(0,10_000)]
r = np.array(r).T

# Calculate matrix for power iteration with teleportation
beta = 0.85
e=np.ones(10_000)
A = (beta*M + (1-beta)*(1/10_000)*e*(e.T))

# Iterate until threshold reached
epsilon = 0.0001
while True:
    r_new = A@r
    if (np.linalg.norm(r_new-r,1)<epsilon):
        break
    else:
        r = r_new

# Find index with maximum PageRank score
max_rank = r_new[0]
max_index = 0
for i in range(0,len(r_new)):
    if r_new[i] > max_rank:
        max_rank = r_new[i]
        max_index = i

# Convert index to node and print it
max_node = index_node[max_index]
print(max_node)

#### 1D ####

# Find edges that point to each node
edges_in = {}
for pair in edges:
    in_node = pair[1]
    if in_node not in edges_in.keys():
        edges_in[in_node] = 1
    else:
        edges_in[in_node]+=1

# Get edges that point to the node with
# the highest PageRank score
print(edges_in[max_node])

count = 0
for pair in edges:
    if pair[1] == 486980:
        count +=1


