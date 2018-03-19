#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:59:23 2018

@author: santosh
"""

import pandas as pd
import numpy as np
from numpy import matrix
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
import scipy as sp
from scipy.optimize import minimize


df = pd.DataFrame.from_csv('1challenge.csv')
df0 = df.loc[df['label'] == 1.0]
df1 = df.loc[df['label'] == 0.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]
df2= df.loc[((df['label'] == 0.0) | (df['label'] == 1.0))]
print(df0.shape)
print(df1.shape)
print(dftest.shape)
TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TrainingData  = df2.as_matrix(columns=None)
finalData = dftest.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1'])
#plt.plot(TrainingData0[:,0], TrainingData0[:,1], 'x', color='r')
#plt.plot(TrainingData1[:,0], TrainingData1[:,1], 'x', color='b')
#plt.plot(TestData[:,0], TestData[:,1], 'o', color='k')
#plt.axis('equal')
x=matrix([np.ones(10000),TrainingData[:,0],TrainingData[:,1]])
x=np.transpose(x)
theta=matrix([0.5,0.5,0.5])
theta=np.transpose(theta)
y=matrix([TrainingData[:,2]])
y=np.transpose(y)

def sigmoi(h):
    z=np.exp(-h)
    g=np.true_divide(1,(1+z))
    return g

def sigmoid(z):
    g = np.array([z]).flatten()
    g =  1/(1+(np.e**-g))
    return g

def costJ(theta, X, y):
    m = len(y)
    hypothesis = sigmoid(X.dot(theta).T)
    J = np.sum(y*np.log(hypothesis)+(1-y)*np.log(1-hypothesis))/m
    return -J
finalData=np.transpose(matrix([np.ones(5000),finalData[:,0],finalData[:,1]]))
#finalData=np.transpose(finalData)
#finding optimum theta
#yy=np.transpose(x)*x
#yy1=inv(yy)
#yy2=np.transpose(x)*y
#theta=yy1*np.transpose(x)*y
#endData=finalData*theta
initial_theta=([0,0,0])
# cost function
def cost(t1,x,y):
      #print(t1)
      t2=matrix(t1)
      t2=(np.transpose(t2))
      h=x*t2
      y1= np.sum(np.multiply(y,np.log(sigmoi(h)))+np.multiply((1-y),(np.log(1-sigmoi(h)))))
      cos=-(y1)/10000
      return(cos)
 
def regCostFunction(theta, X, y, _lambda = 0.1):
    m = len(y)
    h = sigmoid(X.dot(theta))
    tmp = np.copy(theta)
    tmp[0] = 0 
    reg = (_lambda/(2*m)) * np.sum(tmp**2)

    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + reg

def regGradient(theta, X, y, _lambda = 0.1):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X.dot(theta))
    tmp = np.copy(theta)
    tmp[0] = 0
    reg = _lambda*tmp /m

    return ((1 / m) * X.T.dot(h - y)) + reg

#theta, cost1, _, _, _ = \
    #    sp.optimize.fmin(lambda t: cost(t, x, y), initial_theta, **options)
#Result = minimize(f=costJ,x0=initial_theta,args = (x,y),maxiter = 1)
#optimiz.fmin(lambda t: cost(x,theta,y), initial_theta, **options)
result = minimize(fun = regCostFunction, x0 = initial_theta, args = (x, y),
                         method = 'TNC', jac = regGradient)


yy=np.ones(5000)
for ii in range(0,5000):
   yy[ii]=sigmoi(finalData[ii]*result.x)
   
