# Mason Rumuly
# ECEN 662-600
# Final Project
#
# Implements exponential distribution

from DistributionInterface import Distribution
from math import exp
from statistics import mean
from scipy import random


class Exponential(Distribution):

    # returns name associated with the distribution
    @staticmethod
    def name():
        return 'Exponential'

    # estimate parameters based on list x of values
    # select argument should identify different estimation methods, if they exist
    def estimate(self, sample, select=0):

        # unbiased estimator
        if select == 0:
            self.scale = mean(sample) - min(sample)  # use memoryless property here; beginning shouldn't matter
            self.loc = min(sample) - self.scale/len(sample)
            return self

        # maximum-likelihood estimator
        if select == 1:
            self.loc = min(sample)
            self.scale = mean(sample)
            return self

        raise IndexError()

    # return the list of parameters
    def get_params(self):
        return [self.loc, self.scale]

    # set parameters to pre-defined value
    def set_params(self, theta):
        self.loc = theta[0]
        self.scale = theta[1]
        return self

    # get pdf value of distribution at a given value
    def pdf(self, x):
        if x < self.loc:
            return 0

        return exp(self.scale * (self.loc - x) / self.scale) / self.scale

    # get cdf value of distribution at a given value
    def cdf(self, x):
        if x < self.loc:
            return 0

        return 1 - exp((self.loc - x) / self.scale)

    # generate a random observation according to this distribution
    def get_observation(self):
        return self.loc + random.exponential(self.scale)

    # ---------------------------------------------------------------
    # REPOSITORY

    # parameters
    loc = 0  # lower limit
    scale = 1
