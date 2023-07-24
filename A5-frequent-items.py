# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:38:39 2023

@author: bruns
"""

import pandas as pd
import random
import time

s1=open("C:\\Users\\bruns\\Desktop\\Data_Mining\\HW5\\S1.txt")
s2=open("C:\\Users\\bruns\\Desktop\\Data_Mining\\HW5\\S2.txt")
s1=s1.read()
s2=s2.read()

########## Real values ##########

actual_s1 = {}

for c in s1:
    if c not in actual_s1:
        actual_s1[c] = 1
    else:
        actual_s1[c] = actual_s1[c] + 1

actual_s2 = {}

for c in s2:
    if c not in actual_s2:
        actual_s2[c] = 1
    else:
        actual_s2[c] = actual_s2[c] + 1

df = pd.DataFrame(data=actual_s2, index=[0])

df = (df.T)

print (df)

df.to_excel('C:\\Users\\bruns\\Desktop\\Data_Mining\\HW5\\dict2.xlsx')

################ 1A S1 ################

counters = {}

for c in s1:
    if len(counters)<11 and c not in counters:
        counters[c] = 1
    elif c in counters:
        counters[c] = counters[c] + 1
    elif c not in counters and 0 in counters.values():
        for key in counters:
            if counters[key]==0:
                del counters[key]
                counters[c] = 1
                break
    else:
        for key in counters:
            counters[key] = counters[key] - 1

df = pd.DataFrame(data=counters, index=[0])

df = (df.T)

print (df)

df.to_excel('C:\\Users\\bruns\\Desktop\\Data_Mining\\HW5\\dictASDASd.xlsx')


################ 1A S2 ################

counters = {}

for c in s2:
    if len(counters)<11 and c not in counters:
        counters[c] = 1
    elif c in counters:
        counters[c] = counters[c] + 1
    elif c not in counters and 0 in counters.values():
        for key in counters:
            if counters[key]==0:
                del counters[key]
                counters[c] = 1
                break
    else:
        for key in counters:
            counters[key] = counters[key] - 1
            
df = pd.DataFrame(data=counters, index=[0])

df = (df.T)

print (df)

df.to_excel('C:\\Users\\bruns\\Desktop\\Data_Mining\\HW5\\dictASDASd.xlsx')

################ 1B S1 ################
all_dicts_s1 = {}

def rand_hash(x, a, b, c, d,e):
    return (a*x**b + c*x**e + d*x)**2 % 12

while len(all_dicts_s1) < 1:
    run_dict = {}
    count = 0
    for j in range(0,12):
        run_dict[j] = [0,set()]
    a = random.randint(-10000,10000)
    b = random.randint(5,100)
    e = random.randint(5,100)
    z = random.randint(-10000,10000)
    d = random.randint(-10000,10000)
    for c in s1:
        hashed = rand_hash(ord(c), a, b, z, d, e)
        run_dict[hashed][0] = run_dict[hashed][0]+1
        run_dict[hashed][1].add(c)
    for j in range(0,12):
        if (run_dict[j][0] == 0):
            count = count + 1
    if count == 0:
        all_dicts_s1[len(all_dicts_s1)] = run_dict

f3 = open('C:\\Users\\bruns\\Desktop\\Data_Mining\\HW5\\dict3.txt','w')
for key in all_dicts_s1.keys():
    f3.write(str(all_dicts_s1[key]) + "\n")

f3.close()

################ 1B S2 ################
all_dicts_s2 = {}

def rand_hash(x, a, b, c, d,e):
    return (a*x**b + c*x**e + d*x)**2 % 12

while len(all_dicts_s2) < 6:
    run_dict = {}
    count = 0
    for j in range(0,12):
        run_dict[j] = [0,set()]
    a = random.randint(100,10000)
    b = random.randint(5,100)
    e = random.randint(5,100)
    z = random.randint(100,10000)
    d = random.randint(100,10000)
    for c in s2:
        hashed = rand_hash(ord(c), a, b, z, d, e)
        run_dict[hashed][0] = run_dict[hashed][0]+1
        run_dict[hashed][1].add(c)
    for j in range(0,12):
        if (run_dict[j][0] == 0):
            count = count + 1
    if count <= 0:
        all_dicts_s2[len(all_dicts_s2)] = run_dict
        print(len(all_dicts_s2))


f3 = open('C:\\Users\\bruns\\Desktop\\Data_Mining\\HW5\\dict4.txt','w')
for key in all_dicts_s2.keys():
    f3.write(str(all_dicts_s2[key]) + "\n")

f3.close()


f3 = open('C:\\Users\\bruns\\Desktop\\Data_Mining\\HW5\\dict3.txt')
test = f3.readlines()
f3.close()
