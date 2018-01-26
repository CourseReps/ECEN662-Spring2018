# Challenge 1: Binary Detection 1

The purpose of Challenge 1 is to gain experience with binary detection, as it pertains to statistical inference.
This challenge assumes that you are already familiar with [Python](https://www.python.org/), and you have a working development environment.
You should also be versed in using [Git](https://git-scm.com/) and [GitHub](https://github.com/).


## Binary Detection

The file [1challenge.csv](./1challenge.csv) contains the data for Challenge 1.
Every row corresponds to a 2-dimensional sample.
Columns one and two are the components of the 2-dimensional samples; whereas the third column, if present, is a label for the training data.
The first 6000 rows are training samples for `theta = 0`.
The next 4000 rows are training samples for `theta = 1`.
Your task is to produce a label for the remaining 5000 samples.
The objective is to minimize the average number of erroneous guesses.

Complete the following tasks.
1. Develop a detection algorithm for this problem in Python, and provide estimates for the unlabeled samples.
2. Add the missing labels to your copy of the CSV files, which should be located at
  *  **ECEN662-Spring2018/ECEN662-Spring2018/Students/\<GitHubID\>/1Challenge/1challenge.csv**
3. Finally, `commit` your work to your local Git repository, `pull` latest changes from the master, and then `push` your updates.

