# Mason Rumuly
# ECEN 662-600
# Final Project
#
# Implements Laplace distribution

from DistributionInterface import Distribution
from scipy import random
from statistics import median
from math import exp
from functools import reduce


class Laplace(Distribution):

    # returns name associated with the distribution
    @staticmethod
    def name():
        return 'Laplace'

    # estimate parameters based on list x of values
    # select argument should identify different estimation methods, if exist
    def estimate(self, sample, select=0):

        # maximum-likelihood estimator
        if select == 0:
            self.loc = median(sample)
            self.scale = reduce(lambda a, x: a + abs(x - self.loc), sample)/len(sample)
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
        return exp(-abs(x-self.loc) / self.scale) / (2*self.scale)

    # get cdf value of distribution at a given value
    def cdf(self, x):
        if x < self.loc:
            return exp((x-self.loc)/self.scale)/2
        return 1 - exp((self.loc-x)/self.scale)/2

    # generate a random observation according to this distribution
    def get_observation(self):
        return random.laplace(self.loc, self.scale)

    # ---------------------------------------------------------------
    # REPOSITORY

    # parameters
    loc = 0
    scale = 1
