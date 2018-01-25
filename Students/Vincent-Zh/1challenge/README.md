#Guide

In my '1challenge' folder, there are two .ipynb files.

Algorithm in '1challenge.ipynb' is to assume that data in trainsets follow 2-D Gaussian Distribution. We need to estimate the mean vector and covariance matrix of each class, in order to build its PDF. Then we use Bayes Rules to classfify data in testset.

Algorithm in '1challenge_KNN.ipynb' is just the build-in KNN algorithm in 'sklearn', where K=3.

I also use training data to test the error rate. The error rates of these two algorithms are 35% and 20%, respectively. Also, the figures of result for test data show that KNN algorithm performs much better. 

When we use Bayes Rules to classify, it's important to build accurate model for each class, i.e. pdf or pmf). For this challenge, 2-D Gaussian model may not match the real distribution well. I think that's the reason the classifier didn't perform well.
