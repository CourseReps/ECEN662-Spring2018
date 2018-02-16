import pandas as pd
from scipy.stats import beta
import numpy as np

# From analyzing the data in R (the code is provided in 2Data_Analysis.R and the results are published in 2Data_Analysis.nb.html, I concluded that under both hypothesis rvs are iid and all the distributions are Beta. Also for any rv the distributions under H0 and H1 are symmetric so their shape params are switched
# Define and freeze distributions of each rv under H0 and H1
a = 2.6
b = 3.9
dists = [beta(a, b), beta(b, a)]


def mrvpdf(p):
    q0 = q1 = 1.0
    for y in p:
        if ~np.isnan(y):
            q0 = q0 * dists[0].pdf(y)
            q1 = q1 * dists[1].pdf(y)
    return q0, q1


# Read the Data from the CVS file
df = pd.DataFrame.from_csv("2challengeImputed.csv")
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]
TestData = dftest.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])

for i in range(0, 5000):
    df.values[10000+i, 8] = np.argmax(mrvpdf(TestData[i, :]))

df.to_csv("2challenge.csv")
