# Challenge 1

The purpose of Challenge 1 is to gain experience with Python, as it pertains to statistical inference.
This challenge assumes that you are already familiar with [Python](https://www.python.org/), and you have a working development environment.
You should also be versed in using [Git](https://git-scm.com/) and [GitHub](https://github.com/).

## Binary Detection

The file 1challenge.csv contains the data for Challenge 1.
Every row corresponds to a 2-dimensional sample.
Column one and two are the elements of these 2-dimensional samples; whereas the third column, if present, is a label for the training data.
The first 6000 rows are training samples for `theta = 0`.
The next 4000 rows are training samples for `theta = 1`.
Your task is to produce a label for the remaining 5000 samples.
The objective is to minimize the average number of erroneous guesses.
