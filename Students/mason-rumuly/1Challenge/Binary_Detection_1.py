# Mason Rumuly
# Challenge 1
#
# Binary detection

import pandas as pd
import numpy as np
import Naive_Bayes
import matplotlib.pyplot as plt


# Import Data
nbc = Naive_Bayes.NaiveBayes()

df = pd.DataFrame.from_csv("1challenge.csv")
df0 = df.loc[df['label'] == 1.0]
df1 = df.loc[df['label'] == 0.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]

# Display data sizes
print(df0.shape)
print(df1.shape)
print(dftest.shape)

# Convert to numpy arrays
TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1'])

# Classify

nbc.train(df0)
nbc.train(df1)
result = nbc.test(dftest)

TestData0 = []
TestData1 = []
for i in range(len(result)):
    if result[i] == 'Y0':
        TestData0.append(dftest[i, :])
    elif result[i] == 'Y1':
        TestData1.append(dftest[i, :])


# Visualize Data
plt.plot(TestData0[:,0], TestData0[:,1], 'x', color='r')
plt.plot(TestData1[:,0], TestData1[:,1], 'x', color='b')

# plt.plot(TrainingData0[:,0], TrainingData0[:,1], 'x', color='r')
# plt.plot(TrainingData1[:,0], TrainingData1[:,1], 'x', color='b')
# plt.plot(TestData[:,0], TestData[:,1], 'o', color='k')
plt.axis('equal')
plt.show()
