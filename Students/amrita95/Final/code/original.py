import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

score_csv= '/home/amrita95/Desktop/Estimation and Detection/project/deliveries.csv'
match_csv= '/home/amrita95/Desktop/Estimation and Detection/project/matches.csv'
year = 2010

def data_loading(score_csv,match_csv,year):
    score = pd.read_csv(score_csv)
    match = pd.read_csv(match_csv)

    match = match.loc[match.season==year,:]
    match = match.loc[match.dl_applied == 0,:]

    return score,match

score, match = data_loading(score_csv,match_csv,year)

#preprocessing
score = pd.merge(score, match[['id','season', 'winner', 'result', 'dl_applied', 'team1', 'team2']], left_on='match_id', right_on='id')
score.player_dismissed.fillna(0, inplace=True)
score['player_dismissed'].loc[score['player_dismissed'] != 0] = 1
score.player_dismissed.fillna(0, inplace=True)
score['player_dismissed'].loc[score['player_dismissed'] != 0] = 1



#feature extraction
train = score.groupby(['match_id', 'inning', 'over', 'team1', 'team2', 'batting_team', 'winner'])[['total_runs', 'player_dismissed']].agg(['sum']).reset_index()
train.columns = train.columns.get_level_values(0)

train['innings_wickets'] = train.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()
train['innings_score'] = train.groupby(['match_id', 'inning'])['total_runs'].cumsum()


temp = train.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
temp = temp.loc[temp['inning']==1,:]

temp.head()
temp['inning'] = 2
temp.columns = ['match_id', 'inning', 'score_target']
train = train.merge(temp, how='left', on = ['match_id', 'inning'])
train['score_target'].fillna(-1, inplace=True)



def get_remaining_target(row):
    if row['score_target'] == -1.:
        return -1
    else:
        return row['score_target'] - row['innings_score']
train['remaining_target'] = train.apply(lambda row: get_remaining_target(row),axis=1)

train['run_rate'] = train['innings_score'] / train['over']

def get_required_rr(row):
    if row['remaining_target'] == -1:
        return -1.
    elif row['over'] == 20:
        return 99
    else:
        return row['remaining_target'] / (20-row['over'])

train['required_run_rate'] = train.apply(lambda row: get_required_rr(row), axis=1)


def get_rr_diff(row):
    if row['inning'] == 1:
        return -1
    else:
        return row['run_rate'] - row['required_run_rate']

train['runrate_diff'] = train.apply(lambda row: get_rr_diff(row), axis=1)
train['is_batting_team'] = (train['team1'] == train['batting_team']).astype('int')
train['target'] = (train['team1'] == train['winner']).astype('int')


# modelling Logistic Regression
x_cols = ['inning', 'over', 'total_runs', 'player_dismissed', 'innings_wickets', 'innings_score', 'score_target', 'remaining_target', 'run_rate', 'required_run_rate', 'runrate_diff', 'is_batting_team']

val_df = train.loc[train.match_id == 234,:]
dev_df = train.loc[train.match_id != 234,:]


dev_X = np.array(dev_df[x_cols[:]])
dev_y = np.array(dev_df['target'])
val_X = np.array(val_df[x_cols[:]])[:-1,:]
val_y = np.array(val_df['target'])[:-1]
print(dev_X.shape, dev_y.shape)
print(val_X.shape, val_y.shape)

logr = LogisticRegression()
logr.fit(dev_X,dev_y)

pred = logr.predict_proba(val_X)


# Plotting
out_df = pd.DataFrame({'Team1':val_df.team1.values})
out_df['is_batting_team'] = val_df.is_batting_team.values
out_df['innings_over'] = np.array(val_df.apply(lambda row: 'o'*row['player_dismissed']+ '  ' + str(row['inning']) + "_" + str(row['over']), axis=1))
out_df['innings_score'] = val_df.innings_score.values
out_df['innings_wickets'] = val_df.innings_wickets.values
out_df['score_target'] = val_df.score_target.values
out_df['total_runs'] = val_df.total_runs.values
out_df['predictions'] = list(pred[:,1])+[1]

coeff = logr.coef_
print(len(coeff.T))
plt.figure(1)
fig, ax1 = plt.subplots(figsize=(12,6))
ax2 = ax1.twinx()
labels = np.array(out_df['innings_over'])
ind = np.arange(len(labels))
width = 0.7
rects = ax1.bar(ind, np.array(out_df['innings_score']), width=width, color=['yellow']*20 + ['red']*20)
ax1.set_xticks(ind+((width)/2.))
ax1.set_xticklabels(labels, rotation='vertical')
ax1.set_ylabel("Innings score")
ax1.set_xlabel("Innings and over")
ax1.set_title("Win percentage prediction for Chennai Super Kings - over by over")

ax2.plot(ind+0.35, np.array(out_df['predictions']), color='b', marker='o')
ax2.plot(ind+0.35, np.array([0.5]*40), color='green', marker='o')
ax2.set_ylabel("Win percentage", color='b')
ax2.set_ylim([0,1])
ax2.grid(b=False)

plt.figure(2)

fig, ax1 = plt.subplots(figsize=(12,6))
ax2 = ax1.twinx()
labels = np.array(out_df['innings_over'])
ind = np.arange(len(labels))
width = 0.7
rects = ax1.bar(ind, np.array(out_df['total_runs']), width=width, color=['yellow']*20 + ['red']*20)
ax1.set_xticks(ind+((width)/2.))
ax1.set_xticklabels(labels, rotation='vertical')
ax1.set_ylabel("Runs in the given over")
ax1.set_xlabel("Innings and over")
ax1.set_title("Win percentage prediction for Chennai Super Kings- over by over")

ax2.plot(ind+0.35, np.array(out_df['predictions']), color='b', marker='o')
ax2.plot(ind+0.35, np.array([0.5]*40), color='green', marker='o')
ax2.set_ylabel("Win percentage", color='b')
ax2.set_ylim([0,1])
ax2.grid(b=False)

import numpy as np
import matplotlib.pyplot as plt


performance = [48.38,55.45,52.99,64.66]
y_pos = np.arange(len(x_cols)-1)

plt.figure(1)
plt.bar(y_pos,abs(coeff.T[:-1]) , align='center', alpha=0.5, color= 'Blue')
plt.xticks(y_pos,x_cols[:-1],rotation='vertical')
plt.ylabel('Weight of each feature')
plt.title('Feature Analysis')
plt.show()
plt.show()

print(coeff.T[:-1])
