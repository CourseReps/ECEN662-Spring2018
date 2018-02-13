# Mason Rumuly
# Challenge 2
#
# Binary detection

from Full_Bayes import FullBayes

import pandas as pd
import numpy as np

# Import Data
df = pd.read_csv("2challenge.csv")
df0 = df.loc[df['label'] == 0.0]
df1 = df.loc[df['label'] == 1.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]

# Display data sizes
print(df0.shape)
print(df1.shape)
print(dftest.shape)

# Convert to numpy arrays
TrainingData0 = df0.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'label'])
TrainingData1 = df1.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'label'])

TestData0 = df0.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])
TestData1 = df1.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])

TestData = dftest.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])

print(np.shape(TrainingData0))
print(np.shape(TrainingData1))
print(np.shape(TestData))

# train a Full Bayes instance
fb = FullBayes()
fb.train(TrainingData0)
fb.train(TrainingData1)

# test
sanity_results_0 = fb.test(TestData0)
sanity_results_1 = fb.test(TestData1)
results = fb.test(TestData)

# Print result ratio
print((sanity_results_0.count(0.0), sanity_results_0.count(1.0)))
print((sanity_results_1.count(0.0), sanity_results_1.count(1.0)))
print((results.count(0.0), results.count(1.0)))

# Save results to csv
for i in range(len(results)):
    if results[i] == 0:
        dftest.at[10000 + i, 'label'] = 0.0
    elif results[i] == 1:
        dftest.at[10000 + i, 'label'] = 1.0
df = pd.concat([df0, df1, dftest], join='outer', ignore_index=True)
df.to_csv("2challenge.csv")