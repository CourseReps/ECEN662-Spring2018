# Mason Rumuly
# Challenge 1
#
# Binary detection

# Assumptions:
# Gaussian distribution of features
#

import pandas as pd
import numpy as np
import Selective_Bayes
import matplotlib.pyplot as plt


# Import Data
df = pd.DataFrame.from_csv("1challenge.csv")
df0 = df.loc[df['label'] == 0.0]
df1 = df.loc[df['label'] == 1.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]

# Display data sizes
print(df0.shape)
print(df1.shape)
print(dftest.shape)

# Convert to numpy arrays
TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1'])

# --------------------------------------------------------------------------------
# Classify
nbc = Selective_Bayes.SelectiveBayes()

# make training set; use addition and subtraction of the parameters to capture dependencies
TrainingData = np.concatenate((TrainingData0, TrainingData1), 0)
TrainingData = TrainingData[:, :-1]  # trim off label column
TrainingData = np.concatenate((TrainingData,
                # sum of variables
                np.reshape(np.add(TrainingData[:, 0], TrainingData[:, 1]), (np.shape(TrainingData)[0], 1)),
                # difference of variables
                np.reshape(np.subtract(TrainingData[:, 0], TrainingData[:, 1]), (np.shape(TrainingData)[0], 1)),
                # product of variables
                np.reshape(np.multiply(TrainingData[:, 0], TrainingData[:, 1]), (np.shape(TrainingData)[0], 1)),
                # ratio of variables
                np.reshape(np.multiply(TrainingData[:, 0], TrainingData[:, 1]), (np.shape(TrainingData)[0], 1)),
                # return label column
                np.reshape(np.concatenate((TrainingData0, TrainingData1), 0)[:, -1], (np.shape(TrainingData)[0], 1))
            ), 1)
nbc.train(TrainingData)
# do the same to the test data
TestData = np.concatenate((TestData,
                # sum of variables
                np.reshape(np.add(TestData[:, 0], TestData[:, 1]), (np.shape(TestData)[0], 1)),
                # difference of variables
                np.reshape(np.subtract(TestData[:, 0], TestData[:, 1]), (np.shape(TestData)[0], 1)),
                # product of variables
                np.reshape(np.multiply(TestData[:, 0], TestData[:, 1]), (np.shape(TestData)[0], 1)),
                # ratio of variables
                np.reshape(np.multiply(TestData[:, 0], TestData[:, 1]), (np.shape(TestData)[0], 1))
            ), 1)
result = nbc.test(TestData)

print(nbc.rejected_features(), nbc.coherence())

TestData0 = []
TestData1 = []
for i in range(len(result)):
    if result[i][0] == 0:
        TestData0.append(TestData[i, :])
        dftest.at[10000 + i, 'label'] = 0.0
    elif result[i][0] == 1:
        TestData1.append(TestData[i, :])
        dftest.at[10000 + i, 'label'] = 1.0
TestData0 = np.array(TestData0)
TestData1 = np.array(TestData1)

print(np.shape(TestData0))
print(np.shape(TestData1))

# --------------------------------------------------------------------------------

# Visualize Data
plt.plot(TestData0[:,0], TestData0[:,1], 'x', color='r')
plt.plot(TestData1[:,0], TestData1[:,1], 'x', color='b')
# plt.plot(TrainingData0[:,0], TrainingData0[:,1], 'x', color='r')
# plt.plot(TrainingData1[:,0], TrainingData1[:,1], 'x', color='b')
# plt.plot(TestData[:,0], TestData[:,1], 'o', color='k')
plt.axis('equal')
plt.show()

# output to csv
print(dftest)
df = pd.concat([df0, df1, dftest], join='outer', ignore_index=True)
df.to_csv("1challenge.csv")
