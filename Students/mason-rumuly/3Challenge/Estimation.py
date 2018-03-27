# Mason Rumuly
# Challenge 3
#
# estimation

# import packages
import pandas as pd
import numpy as np

# load data
df = pd.read_csv("3challenge-1.csv")

# get numpy matrix of data
data = df.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])

# get sum for each row
sums = np.sum(data, 1)

# insert estimates using bayesian maximum likelihood (solved elsewhere)
for i in range(5000, df.shape[0]):
    df.at[i, 'label'] = (1 + sums[i])/325

# save estimates
df.to_csv("3challenge-1.csv")
