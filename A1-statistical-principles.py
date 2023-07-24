# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:19:13 2023

@author: bruns
"""
### 1A ###
import random
def pick_until_repeat(x):
    count = 0
    picked_nums = set({})
    while True:
        rand_int = random.randint(1,x)
        if rand_int not in picked_nums:
            picked_nums.add(rand_int)
            count = count + 1
        else:
            return count

print(pick_until_repeat(10_000))

### 1B ###
# from https://www.geeksforgeeks.org/how-to-calculate-and-plot-a-cumulative-distribution-function-with-matplotlib-in-python/
import matplotlib.pyplot as plt
import numpy as np
import time

trial_counts = []
t1 = time.time()
for i in range(0,500):
    trial_counts.append(pick_until_repeat(10_000))
t2 = time.time()
x = np.sort(trial_counts)
y = np.arange(500)/float(500)

plt.figure(figsize=(12,8))
plt.xlabel("Number of trials")
plt.ylabel("Cumulative Density")
plt.plot(x,y)
plt.show()

### 1C ###
print(sum(x)/len(x))

### 1D.ii ###
print((t2 - t1)*1_000)

### 1D.iii ###

time.sleep(3)
trial_counts = []
plt.figure(figsize=(6,4))

m_vals = [500, 3333, 5000, 7500, 10000]
n_vals = list(np.linspace(10_000, 1_000_000, 15))
n_vals = [int(x) for x in n_vals]

for m in m_vals:
    times = []
    for n in n_vals:
        t1 = time.time()
        for i in range(0,m):
            trial_counts.append(pick_until_repeat(n))
        t2 = time.time()
        times.append((t2-t1)*1000)
    plt.plot(n_vals,times, label = str(m) + " trials")

plt.xlabel("Size of n")
plt.ylabel("Time to complete with m trials (ms)")
plt.legend(loc="upper center", bbox_to_anchor = (0.5, -0.15))    

### 2A ###
def pick_all_numbers(x):
    domain = set(np.arange(1,x+1,1))
    count = 0
    while len(domain) > 0:
        rand_int = random.randint(1,x)
        if rand_int in domain:
            domain.remove(rand_int)
        count = count + 1
    return count

print(pick_all_numbers(1000))

### 2B ###
import matplotlib.pyplot as plt
import numpy as np
import time

time.sleep(3)

trial_counts = []
t1 = time.time()
for i in range(0,500):
    trial_counts.append(pick_all_numbers(1000))
t2 = time.time()
x = np.sort(trial_counts)
y = np.arange(500)/float(500)

plt.figure(figsize=(6,4))
plt.xlabel("Number of trials")
plt.ylabel("Cumulative Density")
plt.plot(x,y)
plt.show()

### 2C ###
print(sum(x)/len(x))

### 2C.ii ###
print(t2-t1)

### 2C.iii ###

time.sleep(3)
trial_counts = [] 
plt.figure(figsize=(6,4))

m_vals = [500, 1000, 2500, 4000, 5000]
n_vals = list(np.linspace(1_000, 20_000, 7))
n_vals = [int(x) for x in n_vals]

for m in m_vals:
    times = []
    for n in n_vals:
        t1 = time.time()
        for i in range(0,m):
            trial_counts.append(pick_all_numbers(n))
        t2 = time.time()
        times.append((t2-t1))
    plt.plot(n_vals,times, label = str(m) + " trials")

plt.xlabel("Size of n")
plt.ylabel("Time to complete with m trials (seconds)")
plt.legend(loc="upper center", bbox_to_anchor = (0.5, -0.15))

### 3A ###
import math
estimate = int(math.sqrt(10_000*2*.5))

estimate_prob = 1 - (1 - 1/10_000)**(math.comb(estimate,2))

while estimate_prob <= 0.5:
    estimate = estimate + 1
    estimate_prob = 1 - (1 - 1/10_000)**(math.comb(estimate,2))

def product_of_probabilities(n, k):
    i = 1
    product = 1
    for i in range(1,k):
        product = product * ((n-i)/n)
    return 1 - product

estimate = int(math.sqrt(10_000*2*.5))

estimate_prob = product_of_probabilities(10_000, 119)

while estimate_prob <= 0.5:
    estimate = estimate + 1
    estimate_prob = product_of_probabilities(10_000, estimate)

### 3B ###
gamma = 0.577
print(1000*(gamma + np.log(1000)))


