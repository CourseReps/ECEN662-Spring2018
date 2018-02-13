# Mason Rumuly
# Challenge 2
#
# Conduct PCA analysis of data provided
# Plot resulting cast to 2 dimensions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# iris = datasets.load_iris()

# Import Data
df = pd.read_csv("2challenge.csv")
df0 = df.loc[df['label'] == 0.0]
df1 = df.loc[df['label'] == 1.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]

# Convert to numpy arrays
TrainingData0 = df0.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])
TrainingData1 = df1.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])
TestData = dftest.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])

Data = df.as_matrix(columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])

# Just training
# X = np.concatenate((TrainingData0, TrainingData1, TestData))
# y = np.concatenate((df0.as_matrix(columns=['label']), df1.as_matrix(columns=['label'])))

# Include test data
X = Data
y = np.concatenate(df.as_matrix(columns=['label']))

# cull observations with NaN
nan_list = []
for row in range(np.shape(X)[0]):
    found = False
    for value in X[row]:
        if np.isnan(value):
            found = True
    if found:
        nan_list.append(row)
X = np.delete(X, nan_list, 0)
y = np.reshape(np.delete(y, nan_list, 0), (np.shape(X)[0],))

target_names = ['0.0', '1.0', 'Test']

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

# Plot after PCA
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0.0, 1.0], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.scatter(X_r[np.isnan(y), 0], X_r[np.isnan(y), 1], color=colors[2], alpha=.8, lw=lw, label=target_names[2])
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Dataset')

plt.show()
