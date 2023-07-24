# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:21:33 2023

@author: bruns
"""
import numpy as np
import math
from random import uniform as unif
import matplotlib.pyplot as plt

########## Problem 1A ##################

#t = 200 hash functions
#threshold is 0.75

b = math.floor(-math.log(200,0.75))
# Gives b = 18, round to 20 so that r=t/b is an integer
# That made a bad graph (sharpest increase around 0.8), try b = 10

b = 10
r = 20

def f(js, b, r):
    return 1-(1-js**b)**r

s_vals = list(np.linspace(0, 1, 1000))
y = [f(s,b,r) for s in s_vals]
plt.figure(figsize=(12,8))
plt.xlabel("Jaccard Similarity")
plt.ylabel("Probability of the pair being a candidate")
plt.plot(s_vals,y)
plt.show()

########## Problem 1B ##################


AB = 0.77
AC = 0.25
AD = 0.33
BC = 0.20
BD = 0.55
CD = 0.91

print("A-B:\t" + str(f(AB,b,r)))
print("A-C:\t" + str(f(AC,b,r)))
print("A-D:\t" + str(f(AD,b,r)))
print("B-C:\t" + str(f(BC,b,r)))
print("B-D:\t" + str(f(BD,b,r)))
print("C-D:\t" + str(f(CD,b,r)))

########### Problem 2 ##################

def make_vector(d):
    vec = []
    for x in range(0,d//2):
        u1 = unif(0,1)
        u2 = unif(0,1)
        # Create two Gaussian random variables from the two uniforms above
        vec.append(math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2))
        vec.append(math.sqrt(-2*math.log(u1))*math.sin(2*math.pi*u2))
    vec = np.array(vec)
    # Get vector length then normalize each index to create a unit vector
    length = np.linalg.norm(vec)
    normal_vec = []
    for x in vec:
        normal_vec.append(x/length)
    return normal_vec
         
# Make 200 random unit vectors for d=100
unit_vectors = []
for n in range(0,200):
    unit_vectors.append(np.array(make_vector(100)))

# Compute every pair of dot products and keep a list of them
dot_products = []
for i in range(len(unit_vectors)):
    for j in range(i,len(unit_vectors)):
        if (j == i):
            continue
        dot_products.append(np.dot(unit_vectors[i], unit_vectors[j]))

# Method from https://www.geeksforgeeks.org/how-to-calculate-and
#-plot-a-cumulative-distribution-function-with-matplotlib-in-python/        
x = np.sort(dot_products)
y = np.arange(len(dot_products))/float(len(dot_products))

plt.figure(figsize=(12,8))
plt.xlabel("Dot product")
plt.ylabel("Cumulative Density")
plt.plot(x,y)
plt.show()


####### Problem 3 #############

# Read in contents
f2=open('C:\\Users\\bruns\\Desktop\\Data_Mining\\HW3\\R.txt')
r_contents = f2.readlines()

# Split each input string into an array of individual numbers
prob_3 = [str.split(x," ") for x in r_contents]

for index in range(len(prob_3)):
    input_str=prob_3[index]
    # Convert each string number to a float
    floats = [float(x) for x in input_str]
    floats = np.array(floats)
    
    # Get size of vector and normalize all elements to make a unit vector
    size = np.linalg.norm(floats)
    normalized_vec=[]
    for x in floats:
        normalized_vec.append(x/size)
    prob_3[index] = np.array(normalized_vec)


def get_angular_similarity(a,b):
    # Assume that a and b are already unit vectors
    dot_prod = np.dot(a,b)
    return 1-(1/math.pi)*np.arccos(dot_prod)

count = 0

angular_similarities = []

# Compute each angular similarity and count those above 0.75
for i in range(len(prob_3)):
    for j in range(i,len(prob_3)):
        if (j == i):
            continue
        if (get_angular_similarity(prob_3[i], prob_3[j]) > 0.75):
            count += 1
        angular_similarities.append(
            get_angular_similarity(prob_3[i], prob_3[j]))

x = np.sort(angular_similarities)
y = np.arange(len(angular_similarities))/float(len(angular_similarities))

plt.figure(figsize=(12,8))
plt.xlabel("Angular similarity")
plt.ylabel("Cumulative Density")
plt.plot(x,y)
plt.show()



########## Prob 3b ################

count = 0

angular_similarities = []
for i in range(len(unit_vectors)):
    for j in range(i,len(unit_vectors)):
        if (j == i):
            continue
        if (get_angular_similarity(unit_vectors[i], unit_vectors[j]) > 0.75):
            count += 1
        angular_similarities.append(get_angular_similarity(
            unit_vectors[i], unit_vectors[j]))

x = np.sort(angular_similarities)
y = np.arange(len(angular_similarities))/float(len(angular_similarities))

plt.figure(figsize=(12,8))
plt.xlabel("Angular similarity")
plt.ylabel("Cumulative Density")
plt.plot(x,y)
plt.show()




