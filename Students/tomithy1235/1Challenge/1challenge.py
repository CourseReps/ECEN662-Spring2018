import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gaussian_functions as gauss

# Please note: this was started by using the code provided in the python notebook for this challenge.


# Import data from file:
df = pd.DataFrame.from_csv("1challenge-in.csv")
df0 = df.loc[df['label'] == 1.0]
df1 = df.loc[df['label'] == 0.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]
print(df0.shape)
print(df1.shape)
print(dftest.shape)

do_validation = False
use_priors = True

# Convert data to python data types:
TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1'])

data0len = TrainingData0.shape[0]
data1len = TrainingData1.shape[0]

# make the validation data (no harm in always doing it):
valSamplesPerSet = 500
startInd0 = data0len - valSamplesPerSet
startInd1 = data1len - valSamplesPerSet
ValidationData = np.concatenate((TrainingData0[startInd0:data0len, 0:2], TrainingData1[startInd1:data1len, 0:2]))

if(do_validation): # reduce the datasets initially
    TrainingData0 = TrainingData0[0:(data0len - valSamplesPerSet), 0:2]
    TrainingData1 = TrainingData1[0:(data1len - valSamplesPerSet), 0:2]

# My stuff:
print("fitting Gaussian to 0 data:")
mean0, cov0 = gauss.fit_gaussian(TrainingData0)

print("fitting Gaussian to 1 data:")
mean1, cov1 = gauss.fit_gaussian(TrainingData1)

train0data = TrainingData0.shape[0]
train1data = TrainingData1.shape[0]
total_train_data = train0data + train1data
P0 = train0data / total_train_data
P1 = train1data / total_train_data

if(not use_priors):
    P0 = 1
    P1 = 1

if(do_validation):
    valOut0 = gauss.evaluate_2d_gaussian(mean0, cov0, ValidationData, P0)
    valOut1 = gauss.evaluate_2d_gaussian(mean1, cov1, ValidationData, P1)

    valDecisionIs0 = valOut0 > valOut1

    correct = np.zeros(shape=(valSamplesPerSet * 2))
    for i in range(0, valSamplesPerSet * 2):
        if(i < valSamplesPerSet):
            correct[i] = valDecisionIs0[i]
        else:
            correct[i] = not valDecisionIs0[i]

    numCorrect = sum(correct)
    print("\n\nValidation test got %i / %i correct.\n\n" % (numCorrect, valSamplesPerSet * 2))

# Evaluate the gaussians, and the A value in the gaussian takes care of the scaling.
output0 = gauss.evaluate_2d_gaussian(mean0, cov0, TestData, P0)
output1 = gauss.evaluate_2d_gaussian(mean1, cov1, TestData, P1)

decisionIs0 = output0 > output1
dataLength = decisionIs0.shape[0]
num0s = np.sum(decisionIs0)
print("There were %i 0 answers and %i 1 answers!" %(num0s, dataLength - num0s))

startInd = data0len + data1len
dataOut = pd.DataFrame(columns=dftest.columns, index=range(startInd, startInd + dataLength - 1))
for i in range(0, dataLength):
    if(decisionIs0[i]):
        dataOut.loc[startInd + i] = pd.Series({'Y0':TestData[i, 0], 'Y1':TestData[i, 1], 'label':0.0})
    else:
        dataOut.loc[startInd + i] = pd.Series({'Y0':TestData[i, 0], 'Y1':TestData[i, 1], 'label':1.0})

# Plot the data:
# plt.plot(TrainingData0[:, 0], TrainingData0[:, 1], 'x', color='r')
# plt.plot(TrainingData1[:, 0], TrainingData1[:, 1], 'x', color='b')
# plt.plot(TestData[:, 0], TestData[:, 1], '.', color='y')
# plt.axis('equal')
# plt.show()


# Output after everything else:
df = pd.concat([df0, df1, dataOut], join='outer', ignore_index=True)
df.to_csv("1challenge-out.csv")

