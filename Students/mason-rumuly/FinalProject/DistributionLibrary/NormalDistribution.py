# Mason Rumuly
# ECEN 662-600
# Final Project
#
# Normal Distribution

from DistributionInterface import Distribution
from statistics import mean
from operator import add
from functools import reduce
from math import exp, sqrt, pi
from scipy.stats import norm
from scipy import random


class Normal(Distribution):

    # returns name associated with the distribution
    @staticmethod
    def name():
        return 'Normal'

    # estimate parameters based on list x of values
    # select argument should identify different estimation methods, if exist
    def estimate(self, sample, select=0):

        # maximum-likelihood estimator
        if select == 0:
            self.mean = mean(sample)
            self.var = reduce(add, map((lambda x: (x - self.mean)**2), sample)) / len(sample)
            return self

        # unbiased estimator
        if select == 1:
            self.mean = mean(sample)
            self.var = reduce(add, map((lambda x: (x - self.mean)**2), sample)) / (len(sample) - 1)
            return self

        raise IndexError()

    # return the list of parameters
    def get_params(self):
        return [self.mean, self.var]

    # set parameters to pre-defined value
    def set_params(self, theta):
        self.mean = theta[0]
        self.var = theta[1]
        return self

    # get pdf value of distribution at a given value
    def pdf(self, x):
        return exp((x - self.mean)**2 / (2 * self.var)) / sqrt(2 * pi * self.var)

    # get cdf value of distribution at a given value
    def cdf(self, x):
        return norm.cdf(x, self.mean, sqrt(self.var))

    # generate a random observation according to this distribution
    def get_observation(self):
        return random.normal(self.mean, sqrt(self.var))

    # ---------------------------------------------------------------
    # REPOSITORY

    # parameters
    mean = 0
    var = 1

