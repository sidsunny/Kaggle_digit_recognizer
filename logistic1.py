# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:39:11 2016

@author: Siddharth
"""

import numpy as np
from scipy.optimize import fmin_bfgs
import pandas as pd

def sigmoid(X):
    den = (1.0 + np.exp(-1.0*X))
    return 1.0 / den

def costfunction(theta, X, y, lmda):
    z = np.dot(X, theta)
    h = sigmoid(z)
    m, n = X.shape
 
    return (-np.dot((y.T) , np.log(h)) - np.dot(((1-y).T) , np.log(1-h)))/m + lmda * (np.sum(np.power(theta[1:], 2))) / (2*m)
 
def derivative(theta, X, y, lmda):
    #h = (m,1), X = (m,n+1), y = (m,1), theta = (n+1,1), grad = (n+1,1)
    z = np.dot(X, theta)
    h = sigmoid(z)
    m,n = X.shape
    grad = np.dot((X.T), (h-y))/m
    grad[1:] = grad[1:] + lmda * theta[1:] / m
    return grad


def oneVsAll(X, y, num_labels, lmda):
    m, n = X.shape
    X = np.append(np.ones((m, 1)), X, axis = 1)
    all_theta = np.zeros((num_labels,n+1))
    lmda
    for c in range(0,num_labels):
        initial_theta = np.zeros((n+1,1))
        myargs = (X,y==c, lmda)
        theta_opt = fmin_bfgs(costfunction, initial_theta, fprime=derivative, maxiter=61, args=myargs)
        all_theta[c,:] = theta_opt
    return all_theta

def predictOneVsAll(all_theta, X):
    m, n = X.shape
    p = np.zeros((m, 1))
    X = np.append(np.ones((m, 1)), X, axis = 1)
    p = np.argmax(sigmoid(np.dot(X,all_theta.T)), axis=1)
    return p

def printaccuracy(p, y):
    count = 0
    #print y
    for i in range(len(y)):
        if p[i]==y[i]:
            count+=1
    print ((count*100)/len(y))

print('Training One-vs-All Logistic Regression...')
df = pd.read_csv('train.csv', header=0)
y = df['label'].values
X = df.drop(['label'], axis=1).values
m, n = X.shape

num_labels = 10
lmda = 0.1
all_theta = oneVsAll(X, y, num_labels, lmda)

pred = predictOneVsAll(all_theta, X)
print (pred.shape, y.shape)
printaccuracy(pred, y)
#print('Training Set Accuracy: ', np.mean(float(pred == y)) * 100)