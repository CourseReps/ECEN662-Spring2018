#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 09:52:33 2018

@author: humberto
"""
import pandas as pd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=np.nan)
#np.set_printoptions(edgeitems=3,infstr='inf',linewidth=75, nanstr='nan', precision=8,suppress=False, threshold=1000, formatter=None)
def estimate_theta(Data):
    estimate = np.sum(Data, axis=1)/(N*Data.shape[1])
    return estimate
#Read the csv file
df = pd.DataFrame.from_csv("3challenge_original.csv")
dftraining = df.loc[~np.isnan(df['label'])]
dftesting = df.loc[np.isnan(df['label'])]
print(dftraining.shape)
print(dftesting.shape)

#Convert data into numpy arrays
TrainingData = dftraining.as_matrix(columns=None)
TestData = dftesting.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])
#Experiments per sample element.
N = 40;
#Now use the MVUE which is T(X) = sum(x_i)/N
#Where x_i is the number of successes for each element of the sample (each element of row)
#and N is the total number of experiments per element of the sample (in each sample there are N realizations)
estimatedTheta = estimate_theta(TestData)
print 'The estimated theta is ', estimatedTheta

#Now we label the data with the estimated theta (for this problem, theta is probability of success)
TestData_labeled = np.c_[TestData,estimatedTheta]
dftest = pd.DataFrame(TestData_labeled,columns=['Y0', 'Y1','Y2','Y3','Y4','Y5','Y6','Y7','label'] )
#
#Estimates for the training data
estimated_theta_for_training_data = estimate_theta(TrainingData[:,0:8])
print 'The estimated theta is for training set ', estimated_theta_for_training_data
given_theta_for_training_set = TrainingData[:,8]
error_variance = np.var(given_theta_for_training_set - estimated_theta_for_training_data)
print 'Error variance with respect to given values for theta'
print 'Variance = ',error_variance

##Data visualization
plt.hist(estimatedTheta, 20, normed=1, facecolor='green', alpha=0.5)
x = np.linspace(0.001,1.01, 100)
#alpha an beta were given in the problem.
plt.plot(x, beta.pdf(x, 2, 5),'r-', lw=5, alpha=0.6, label='beta pdf')
#plt.hist(estimatedTheta)
#plt.axis('equal')
plt.grid()
#plt.show()
#
###Write the csv
df = pd.concat([dftraining, dftest], join='outer', ignore_index=True)
df.to_csv("3challenge-1.csv")