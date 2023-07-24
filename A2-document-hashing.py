# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 21:08:18 2023

@author: bruns
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def n_grams_sklearn_words():
    # Let's read some data 
    url = "https://raw.githubusercontent.com/koaning/icepickle/main/datasets/imdb_subset.csv"
    df = pd.read_csv(url) # This is how you read a csv file to a pandas frame
    corpus = list(df['text']) 
    corpus_small = corpus[:4] # This is a list of 4 movie review
    #Create a count vectorizer object
    vectorizer = CountVectorizer(ngram_range = (2,2), analyzer = 'word')
    #Return a matrix of counts of word ngrams
    return vectorizer.fit_transform(corpus_small).toarray()

# Get the feature count matrix
x=n_grams_sklearn_words()

# Count nonzero entries in the feature count matrix for each review
for doc in range(0,4):
    count = 0
    for index in range(0,len(x[doc])):
        if (x[doc][index] > 0):
            count = count + 1
    print(str(doc + 1) + "\t" + str(count))


# Use something like this to use the matrix to calculate JS
intersection = 0
union = 0
xor = 0
neither = 0
for index in range(0,len(x[0])):
    if (x[0][index] >= 1 and x[1][index] >= 1):
        intersection = intersection + 1
        union = union + 1
    elif ((x[0][index] >= 1 or x[1][index] >= 1) and not
          (x[0][index] >= 1 and x[1][index] >= 1)):
        xor = xor + 1
        union = union + 1
    elif (x[0][index] >= 1 or x[1][index] >= 1):
        union = union + 1
    else:
        neither = neither + 1

### 1b  
def generalized_similarity(intersection,union,xor,neither,x,y,z,z_prime):
    numerator = x * intersection + y * neither + z*xor
    denominator = x * intersection + y * neither + z_prime*xor
    return numerator/denominator


#Use that function to compute JS for all pairs of reviews
pairs = [[1,2], [1,3], [1,4], [2,3],
         [2,4], [3,4]]

for pair in pairs:
    y = pair[0] - 1
    z = pair[1] - 1
    # Counting variables for cardinalities
    intersection = 0
    union = 0
    xor = 0
    neither = 0
    for index in range(0,len(x[y])):
        # Calculate intersection, add to union
        if (x[y][index] >= 1 and x[z][index] >= 1):
            intersection = intersection + 1
            union = union + 1
        # Calculate symmetric difference, add to union
        elif ((x[y][index] >= 1 or x[z][index] >= 1) and not
              (x[y][index] >= 1 and x[z][index] >= 1)):
            union = union + 1
            xor = xor + 1
        # Find number of items in neither set
        else:
            neither = neither + 1
    JS = generalized_similarity(intersection = intersection, 
                                union = union, xor = xor, 
                                neither = neither, 
                                x = 1, y = 0, z = 0, z_prime = 1)
    print(str(pair[0]) + " and " + str(pair[1]) + ": " + str(JS))

### Prob 2
import time
import numpy as np
import random

f1=open('C:\\Users\\bruns\\Desktop\\Data_Mining\\HW2\\D1.txt')
d1_contents = f1.read()

f2=open('C:\\Users\\bruns\\Desktop\\Data_Mining\\HW2\\D2.txt')
d2_contents = f2.read()

def get_3grams(input_str):
    index = 0
    test_set = set({})
    while index < (len(input_str)-2):
        test_set.add(input_str[index:index+3])
        index += 1
    return test_set

set1=get_3grams(d1_contents)
set2=get_3grams(d1_contents)

def h(input_3g, t):
    prod = 1
    for c in input_3g:
        prod = prod * ord(c)
    return prod % t

# Use this to choose t a values I guess
t20 = [random.randint(1, (ord(' ')**3)) for val in range(0,20)]
t60 = [random.randint(1, (ord(' ')**3)) for val in range(0,60)]
t150 = [random.randint(1, (ord(' ')**3)) for val in range(0,150)]
t300 = [random.randint(1, (ord(' ')**3)) for val in range(0,300)]
t600 = [random.randint(1, (ord(' ')**3)) for val in range(0,600)]


v1_20 = [np.inf for val in range(0,20)]
v1_60 = [np.inf for val in range(0,60)]
v1_150 = [np.inf for val in range(0,150)]
v1_300 = [np.inf for val in range(0,300)]
v1_600 = [np.inf for val in range(0,600)]

v2_20 = [np.inf for val in range(0,20)]
v2_60 = [np.inf for val in range(0,60)]
v2_150 = [np.inf for val in range(0,150)]
v2_300 = [np.inf for val in range(0,300)]
v2_600 = [np.inf for val in range(0,600)]

t1 = time.time()
for gram in set1:
    for index in range(0,20):
       if (h(gram,t600[index]) < v1_600[index]):
           v1_600[index] = h(gram,t600[index])

for gram in set2:
    for index in range(0,600):
       if (h(gram,t600[index]) < v2_600[index]):
           v2_600[index] = h(gram,t600[index])

count = 0

for i in range(0,600):
    if (v1_600[i] == v2_600[i]):
        count = count + 1

t2 = time.time()

print(count/600)
print((t2-t1) * 1_000)

set3 = set({})

for x in set1:
    set3.add(x)
    
for x in set2:
    set3.add(x)

# Variables for tracking set cardinalities
intersection = 0
union = 0
xor = 0
neither = 0
for ngram in set3:
    # Calculate set intersection, add to union
    if (ngram in set1 and ngram in set2):
        intersection = intersection + 1
        union = union + 1
    # Calculate symmetric difference, add to union
    elif ((ngram in set1 or ngram in set2) and not
          (ngram in set1 and ngram in set2)):
        union = union + 1
        xor = xor + 1
    # Find number of elements in universe in neither
    else:
        neither = neither + 1
JS = generalized_similarity(intersection = intersection, 
                            union = union, xor = xor, 
                            neither = neither, 
                            x = 1, y = 0, z = 0, z_prime = 1)
print(str(pair[0]) + " and " + str(pair[1]) + ": " + str(JS))








