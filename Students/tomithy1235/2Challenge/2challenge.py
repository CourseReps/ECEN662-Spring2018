import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gaussian_functions as gauss

# Please note: this was started by using the code provided in the python notebook for this challenge.


# Import data from file:
df = pd.DataFrame.from_csv("2challenge-in.csv")
df0 = df.loc[df['label'] == 1.0]
df1 = df.loc[df['label'] == 0.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]
print(df0.shape)
print(df1.shape)
print(dftest.shape)

# Note the first row of 2challenge.csv reads (order of data):
# ,Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,label

do_validation = True
validationSamplesPerSet = 5 # Very small just because I have a bug if it's false right now.
use_priors = False

gaussianDim = df.shape[1] - 1
rawDataDim = df.shape[0]

# Convert data to python data types:
TrainingData0 = df0.as_matrix(columns=None)
TrainingData1 = df1.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])

data0len = TrainingData0.shape[0]
data1len = TrainingData1.shape[0]

# It looks like all empty cells will become nan's so I can just look for those.
# I think I'll just choose random values (in the right range) so that I'm not overfitting
numBlanks0 = [0, 0, 0, 0, 0, 0, 0, 0]
numBlanks1 = [0, 0, 0, 0, 0, 0, 0, 0]
numBlanksTest = [0, 0, 0, 0, 0, 0, 0, 0]

# NOTE: this is assuming all 3 datasets have same length (true for this case)
for j in range(gaussianDim):
    for i in range(data0len):
        if (np.isnan(TrainingData0[i, j])):
            numBlanks0[j] = numBlanks0[j] + 1
            TrainingData0[i, j] = np.random.rand(1)

        if (np.isnan(TrainingData1[i, j])):
            numBlanks1[j] = numBlanks1[j] + 1
            TrainingData1[i, j] = np.random.rand(1)

        if (np.isnan(TestData[i, j])):
            numBlanksTest[j] = numBlanksTest[j] + 1
            TestData[i, j] = np.random.rand(1)

# make the validation data (no harm in always doing it):
startInd0 = data0len - validationSamplesPerSet
startInd1 = data1len - validationSamplesPerSet
ValidationData = np.concatenate((TrainingData0[startInd0:data0len, 0:gaussianDim], TrainingData1[startInd1:data1len, 0:gaussianDim]))

if(do_validation): # reduce the datasets initially
    TrainingData0 = TrainingData0[0:(data0len - validationSamplesPerSet), 0:gaussianDim]
    TrainingData1 = TrainingData1[0:(data1len - validationSamplesPerSet), 0:gaussianDim]

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
    valOut0 = gauss.evaluate_nd_gaussian(mean0, cov0, ValidationData, P0)
    valOut1 = gauss.evaluate_nd_gaussian(mean1, cov1, ValidationData, P1)

    valDecisionIs0 = valOut0 > valOut1

    correct = np.zeros(shape=(validationSamplesPerSet * 2))
    for i in range(validationSamplesPerSet * 2):
        if(i < validationSamplesPerSet):
            correct[i] = valDecisionIs0[i]
        else:
            correct[i] = not valDecisionIs0[i]

    numCorrect = sum(correct)
    print("\n\nValidation test got %i / %i correct.\n\n" % (numCorrect, validationSamplesPerSet * 2))

# Evaluate the gaussians, and the A value in the gaussian takes care of the scaling.
output0 = gauss.evaluate_nd_gaussian(mean0, cov0, TestData, P0)
output1 = gauss.evaluate_nd_gaussian(mean1, cov1, TestData, P1)

decisionIs0 = output0 > output1
dataLength = decisionIs0.shape[0]
num0s = np.sum(decisionIs0)
print("There were %i 0 answers and %i 1 answers!" %(num0s, dataLength - num0s))

startInd = data0len + data1len
dataOut = pd.DataFrame(columns=dftest.columns, index=range(startInd, startInd + dataLength - 1))
for i in range(dataLength):
    if(decisionIs0[i]):
        num = 0.0
    else:
        num = 1.0

    dataOut.loc[startInd + i] = pd.Series({'Y0': TestData[i, 0], 'Y1': TestData[i, 1],
                                           'Y2': TestData[i, 2], 'Y3': TestData[i, 3],
                                           'Y4': TestData[i, 4], 'Y5': TestData[i, 5],
                                           'Y6': TestData[i, 6], 'Y7': TestData[i, 7],
                                           'label': num})


# Output after everything else:
df = pd.concat([df0, df1, dataOut], join='outer', ignore_index=True)
df.to_csv("2challenge.csv")

