import  numpy as np
import  pandas as pd

df = pd.DataFrame.from_csv('3challenge-1.csv')

df_test = df.loc[np.isnan(df['label'])]
df_train = df.loc[~np.isnan(df['label'])]
test_data = df_test.as_matrix(columns=['Y0', 'Y1','Y2','Y3','Y4','Y5', 'Y6', 'Y7'])

N = 40

label_est = np.mean(test_data,axis=1)/N
label_est = label_est.reshape((5000,1))
final_data = np.hstack((test_data,label_est))

df_test1 = pd.DataFrame(final_data,columns=['Y0', 'Y1','Y2','Y3','Y4','Y5','Y6','Y7','label'])



df = pd.concat([df_train, df_test1], join='outer', ignore_index=True)
df.to_csv("3challenge_after.csv")