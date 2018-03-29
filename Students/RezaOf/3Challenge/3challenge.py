import pandas as pd
import numpy as np

df = pd.DataFrame.from_csv("3challenge-1.csv")
df.values[5000:, 8] = (2+np.sum(df.values[5000:,0:8],1))/327
df.to_csv("3challenge-1.csv")
