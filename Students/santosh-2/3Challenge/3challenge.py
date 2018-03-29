#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 19:06:02 2018

@author: santosh
"""

import pandas as pd
import numpy as np
from numpy import matrix

df = pd.DataFrame.from_csv('3challenge-1.csv')
df_test=df.loc[pd.isnull(df['label'])]
df_train=df.loc[~(pd.isnull(df['label']))]
TrainingData  = df_train.as_matrix(columns=None)
TestData  = df_test.as_matrix(columns=None)
x=TestData[:,:8]
y=sum(np.transpose(x))/(8*40)
a1=matrix(df_test.index)
#
for m in range(0, 5000):
  df['label'][a1[0,m]]=y[m]

#using maximum likelihood estimator

df.to_csv('3challenge.csv')


