#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:38:12 2018

@author: Andrew
"""
import xlrd
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
'''
Pick five elements in the equation
Ni, Cr, Mo, Si, Mn
5, 11, 12, 13, 14, 15
'''
data = xlrd.open_workbook('SFE_Dataset.xlsx')
table = data.sheets()[0]
nrows = table.nrows
ncols = table.ncols
predictors = []

def transform(sample_col):
    row = len(sample_col)
    col = len(sample_col[0])
    return [[sample_col[i][j] for i in range(row)] for j in range(col)]

for i in range(ncols - 1):
    predictors.append(table.col_values(i)[1:])
headers = table.row_values(0)[0:-1]
SFE = table.col_values(ncols - 1)[1:]



def selected(sample, predictors, headers):
    sample = transform(sample)
    Head = []
    index = []
    for i in sample:
        for j in range(len(predictors)):
            if i == predictors[j]:
                Head.append(headers[j])
                index.append(j)
    return Head, index

        
k = 0
for j in range(len(predictors)):
    zeroCT = 0
    for value in predictors[j - k]:
        if value == 0:
            zeroCT += 1
        if zeroCT / (nrows - 1) > 0.4:
            del(predictors[j - k])
            del(headers[j - k])
            k += 1
            break

Samples = transform(predictors)
k = 0
for i in range(len(Samples)):
    for element in Samples[i - k]:
        if element == 0:
            del(Samples[i - k])
            del(SFE[i - k])
            k += 1
            break
    
predictors = transform(Samples)
Sample_F = SelectKBest(f_regression, k=5).fit_transform(Samples, SFE)
Sample_MI = SelectKBest(mutual_info_regression, k=5).fit_transform(Samples, SFE)
headers_F, index_F = selected(Sample_F, predictors, headers)
headers_MI, index_MI = selected(Sample_MI, predictors, headers)

'''
ridge regression
'''
reg = linear_model.Ridge (alpha = 1)
reg.fit(Sample_F, SFE)
SFE_hat_F = reg.predict(Sample_F)
reg.fit(Sample_MI, SFE)
SFE_hat_MI = reg.predict(Sample_MI)

'''
without penalty
'''
reg = linear_model.LinearRegression()
reg.fit(Sample_MI, SFE)
SFE_hat_MI_L = reg.predict(Sample_MI)
reg.fit(Sample_F, SFE)
SFE_hat_F_L = reg.predict(Sample_F)

number = [i for i in range(211)]
l = Sample_F.shape
plt.figure(1)

plt.plot(range(1,212), SFE, 'k', label='SFE')
#plt.plot(range(1,212), SFE_hat_F, 'b')
plt.plot(range(1,212), SFE_hat_MI_L, 'r', label = 'SFE Ordinary LS')
plt.plot(range(1,212), SFE_hat_MI, 'b', label = 'SFE Ridge regression')
plt.legend()

'''
for f_regression result
'''
plt.figure(2)
plt.plot(range(1,212), SFE, 'k', label = 'SFE')
plt.plot(range(1,212), SFE_hat_F, 'b', label = 'SFE Ridge regression')
plt.plot(range(1,212), SFE_hat_F_L, 'r', label = 'SFE Ordinary LS')
plt.legend()







