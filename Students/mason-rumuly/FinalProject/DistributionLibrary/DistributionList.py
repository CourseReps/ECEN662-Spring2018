# Mason Rumuly
# ECEN 662-600
# Final Project
#
# Lists all distributions available/implemented in library by class for easy iteration

from DistributionLibrary.UniformContinuousDistribution import UniformContinuous
from DistributionLibrary.ExponentialDistribution import Exponential
from DistributionLibrary.NormalDistribution import Normal
from DistributionLibrary.LaplaceDistribution import Laplace


class DistList:
    def __len__(self):
        return 4

    def __getitem__(self, item):
        if item == 0:
            return UniformContinuous
        if item == 1:
            return Exponential
        if item == 2:
            return Normal
        if item == 3:
            return Laplace

        raise IndexError()
