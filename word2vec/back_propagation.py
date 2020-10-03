# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:05:17 2020

@author: sphil

Back-Propagation Basics
"""


import numpy as np
import matplotlib.pyplot as plt 

V = 2 #Input layer size
N = 8 #Hidden layer size
M = 2 #Output layer size

#load train data
train = np.loadtxt("data/train.txt")
np.random.shuffle(train)

# Hidden Layer
W1 = np.random.random((N, V)) #Input -> Hidden
W2 = np.random.random((M, N)) #Hidden -> Output

#Parameters

fetch = 100 #fetch number
n = 0.01 #learning rate


#functions
def Sigmoid(v): #Sigmoid Function
    return (np.exp(v)/(1 + np.exp(v)))

def singleNN(v, W): #apply neural network
    u = np.dot(W, v) #net input
    y = Sigmoid(u) #activate net input
    return y


def SGD(n, v, dE, W): #Stochastic Gradient Descent 
    for ii in range(len(dE)): #dE implies gradient
        for jj in range(len(v)):
            W[ii][jj] = W[ii][jj] - (n * dE[ii] * v[jj])
    return W


def isTrue(v, t): #to check accuracy
    if ((t == 0) and (v[0] >= v[1])) or ((t == 1) and (v[0] <= v[1])):
        return True
    return False

for i in range(fetch):
    for j in range(len(train)):
        x = np.array([train[j][0], train[j][1]]) #input x
        c = train[j][2] #true value
        h = singleNN(x, W1) #hidden layer vector
        y = singleNN(h, W2) #output layer vector

        if c == 0 :
            t = np.array([1, 0])
        else:
            t = np.array([0, 1])
        e = y - t
        
        #update hidden -> output
        EI2 = np.zeros(np.shape(e))
        for k in range(len(e)):
            EI2[k] = e[k]*y[k]*(1-y[k])

        
        W2 = SGD(n, h, EI2, W2) #apply SGD

        #update input -> hidden
        EI1 = np.zeros(np.shape(h))
        for k in range(len(h)):
            for m in range(len(EI2)):
                EI1[k] = EI1[k] + (EI2[m] * W2[m][k] * h[k] * (1 - h[k]))
        

        W1 = SGD(n, x, EI1, W1) #apply SGD

#load test data
test = np.loadtxt("data/test.txt")
np.random.shuffle(test)

total = len(test)
p_count = 0
n_count = 0

for i in range(len(test)):
    x = np.array([test[i][0], test[i][1]])
    c = test[i][2]
    h = singleNN(x, W1)
    y = singleNN(h, W2)
    
    if isTrue(y, c):
        p_count += 1
    else:
        n_count += 1
        
accuracy = p_count/total

print("Accuracy is "+str(accuracy)+".")
