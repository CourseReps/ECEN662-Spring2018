# Mason Rumuly
# ECEN 662-600
# Final Project
#
# Implements continuous uniform distribution

from DistributionInterface import Distribution
from statistics import median, mean
from functools import reduce
import random


class UniformContinuous(Distribution):

    # returns name associated with the distribution
    @staticmethod
    def name():
        return 'Uniform Continuous'

    # estimate parameters based on list x of values
    # select argument should identify different estimation methods, if exist
    def estimate(self, sample, select=0):

        # unbiased estimator
        # (done by conversion to [0,p] problem by symmetry around sample median, unbiased estimator of midpoint)
        if select == 0:
            midpoint = mean([mean(sample), median(sample)])
            max_dist = (1/len(sample) + 1) * reduce(max, map(lambda x: abs(x - midpoint), sample))
            self.a = midpoint - max_dist
            self.b = midpoint + max_dist
            return self

        # maximum-likelihood estimator
        if select == 1:
            self.a = min(sample)
            self.b = max(sample)
            return self

        raise IndexError()

    # return the list of parameters
    def get_params(self):
        return [self.a, self.b]

    # set parameters to pre-defined value
    def set_params(self, theta):
        self.a = theta[0]
        self.b = theta[1]
        return self

    # get pdf value of distribution at a given value
    def pdf(self, x):
        if x < self.a or x > self.b:
            return 0

        return 1/(self.b-self.a)

    # get cdf value of distribution at a given value
    def cdf(self, x):
        if x < self.a:
            return 0
        if x > self.b:
            return 1
        return (x - self.a)/(self.b - self.a)

    # generate a random observation according to this distribution
    def get_observation(self):
        return random.uniform(self.a, self.b)

    # ---------------------------------------------------------------
    # REPOSITORY

    # parameters, inclusive
    a = 0  # lower limit
    b = 1  # upper limit
