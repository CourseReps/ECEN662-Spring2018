import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gaussian_functions as gauss

# Please note: this was started by using the code provided in the python notebook for this challenge.


# Import data from file:
df = pd.DataFrame.from_csv("1challenge.csv")
df0 = df.loc[df['label'] == 1.0]
df1 = df.loc[df['label'] == 0.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]
print(df0.shape)
print(df1.shape)
print(dftest.shape)


# Convert data to python data types:
TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1'])

# My stuff:
print("fitting Gaussian to 0 data:")
mean0, cov0 = gauss.fit_gaussian(TrainingData0[:, 0:2])

print("fitting Gaussian to 1 data:")
mean1, cov1 = gauss.fit_gaussian(TrainingData1[:, 0:2])

P0 = 0.6
P1 = 0.4

# Evaluate the gaussians, and the A value in the gaussian takes care of the scaling.
output0 = gauss.evaluate_2d_gaussian(mean0, cov0, TestData, P0)
output1 = gauss.evaluate_2d_gaussian(mean1, cov1, TestData, P1)

decisionIs0 = output0 > output1
dataLength = decisionIs0.shape[0]
num0s = np.sum(decisionIs0)
print("There were %i 0 answers and %i 1 answers!" %(num0s, dataLength - num0s))

startInd = 10000
dataOut = pd.DataFrame(columns=dftest.columns, index=range(startInd, startInd + dataLength - 1))
for i in range(0, dataLength):
    if(decisionIs0[i]):
        dataOut.loc[startInd + i] = pd.Series({'Y0':TestData[i, 0], 'Y1':TestData[i, 1], 'label':0.0})
    else:
        dataOut.loc[startInd + i] = pd.Series({'Y0':TestData[i, 0], 'Y1':TestData[i, 1], 'label':1.0})

# Plot the data:
plt.plot(TrainingData0[:, 0], TrainingData0[:, 1], 'x', color='r')
plt.plot(TrainingData1[:, 0], TrainingData1[:, 1], 'x', color='b')
plt.plot(TestData[:, 0], TestData[:, 1], '.', color='y')
plt.axis('equal')
plt.show()


# Output after everything else:
df = pd.concat([df0, df1, dataOut], join='outer', ignore_index=True)
df.to_csv("1challenge.csv")

