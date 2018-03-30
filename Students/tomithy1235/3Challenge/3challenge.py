import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gaussian_functions as gauss

# Please note: this was started by using the code provided in the python notebook for this challenge.





# The objective is to minimize the mean-squared error.
# Note that the parameter theta governing a sample is selected according to a Beta(2,5) distribution.
# Every element in the sample is then generated according to a Binomial(40, theta) distribution.



# Import data from file:
# df = pd.DataFrame.from_csv("2challenge-in.csv")
# df0 = df.loc[df['label'] == 1.0]
# df1 = df.loc[df['label'] == 0.0]
# dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]
# print(df0.shape)
# print(df1.shape)
# print(dftest.shape)



df = pd.DataFrame.from_csv("3challenge-1-in.csv")
dftraining = df.loc[~np.isnan(df['label'])]
dftesting = df.loc[np.isnan(df['label'])]
print(dftraining.shape)
print(dftesting.shape)


TrainingData = dftraining.as_matrix(columns=None)
TestData = dftesting.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])

observationDim = df.shape[1] - 1


trainingDataLen = TrainingData.shape[0]
testDataLen = TestData.shape[0]


#For the Beta(2,5) see here:
# https://upload.wikimedia.org/wikipedia/commons/f/f3/Beta_distribution_pdf.svg

# And sorta roughly for Binomial(40, theta):
# https://upload.wikimedia.org/wikipedia/commons/7/75/Binomial_distribution_pmf.svg


#It should be the case that the average/sum of all the X's will be a sufficient statistic


# I think the most simple way to initially do this is to use the fact that:
# Mean = n*p = 40*theta
# And just use the mean of my samples as the mean, so this says:
#theta = (sample mean)/40

# This method doesn't take into account the distribution of theta. It's likely that a better performance could be
# achieved if considering that.

# TrainingData0 = TrainingData0[0:(data0len - validationSamplesPerSet), 0:observationDim]
answers = TrainingData[:, observationDim]
TrainingData = TrainingData[:, 0:observationDim]
squaredError = 0
for i in range(trainingDataLen):
    diff = (sum(TrainingData[i])/(40*observationDim)) - answers[i]
    squaredError += pow(diff, 2)

meanSquaredError = squaredError / trainingDataLen
print("Mean squared error of training input = %.5f" % (meanSquaredError))


#As an initial version, just immediately do a mean-matching method:
theta = np.ndarray(shape=(testDataLen))
for i in range(testDataLen):
    theta[i] = sum(TestData[i])/(40*observationDim)


startInd = 0
dataOut = pd.DataFrame(columns=dftesting.columns, index=range(startInd, startInd + testDataLen - 1))

for i in range(testDataLen):
    dataOut.loc[startInd + i] = pd.Series({'Y0': TestData[i, 0], 'Y1': TestData[i, 1],
                                           'Y2': TestData[i, 2], 'Y3': TestData[i, 3],
                                           'Y4': TestData[i, 4], 'Y5': TestData[i, 5],
                                           'Y6': TestData[i, 6], 'Y7': TestData[i, 7],
                                           'label': theta[i]})


df = pd.concat([dftraining, dataOut], join='outer', ignore_index=True)
df.to_csv("3challenge-1-out.csv")
