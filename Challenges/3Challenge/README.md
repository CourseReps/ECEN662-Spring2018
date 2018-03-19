# Challenge 3

The purpose of Challenge 3 is to gain experience with parameter estimation, as it pertains to statistical inference.


## Part 1: Parameter Estimation 1

This machine problem is based on the Beta-binomial model, and file 3challenge-1.csv contains the data for this component.
Every row corresponds to an 8-dimensional sample.
Columns one through eight are the elements of the 8-dimensional samples; whereas column nine, if present, is a label for the training data.
The first 5000 rows are training samples; your task is to produce an estimate of theta for each of the remaining 5000 samples.
The objective is to minimize the mean-squared error.
Noe that the parameter theta governing a sample is selected according to a Beta(2,5) distribution.
Every element in the sample is then generated according to a Binomial(40, theta) distribution.


### Tasks

* Develop an estimation algorithm for this problem in Python, and provide estimates for the unlabeled samples.
* Add the missing labels to your copy of the CSV files, which should be located at ECEN662-Spring2018/Students/\<GitHubID\>/3Challenge/3challenge-1.csv
* Finally, commit your work to your local Git repository, pull latest changes from the master, and then push your updates.

