# Mason Rumuly
# ECEN 662-600
# Final Project
#
# Objective Functions
# these functions are used to compare samples and distributions for goodness of fit

from functools import reduce
from operator import add
from math import log


# ----------------------------------------------------------------------------------------------------------------------
# Kullback-Leibler Divergence
# Information-theory/entropy based, rather than probability-based; paper on bayesian characterization
# Not symmetric; symmetry probably not important for this
# Outputs in well-defined range of values, 0 to 1 (0 is perfect fit, 1 is anti-fit)
# Only really works with discrete
def kullback_leibler(sample, distr):

    # calculate test statistic
    return reduce(
        add,
        map(
            (lambda y: sample.count(y) / len(sample) * log(sample.count(y) / (len(sample) * distr.pdf(y)))),
            set(sample)
        )
    )


# ----------------------------------------------------------------------------------------------------------------------
# Kolmogorov-Smirnov Test
# Sensitive to location and shape of EDFs and CDFs
# Converges to 0 if correct distribution.
# Maximum error
def kolmogorov_smirnov(sample, distr):

    # sort observations in sample to help with EDF
    sample.sort()

    # calculate test statistic, max of distances
    return reduce(
        max,
        map(
            (lambda i, y: abs(i/len(sample) - distr.cdf(y))),
            range(len(sample)), sample
        )
    )


# ----------------------------------------------------------------------------------------------------------------------
# Kuiper's Test
# improvement on Kolmogorov-Smirnov Test
# equally sensitive at tails as median, invariant under cyclic transformations of independent variable
# Maximum error
def kuiper(sample, distr):

    # sort observations in sample to help with EDF
    sample.sort()

    # calculate test statistic, max of distances
    return reduce(
        max,
        map(
            (lambda i, y: i/len(sample) - distr.cdf(y)),
            range(len(sample)), sample
        )
    ) + reduce(
        max,
        map(
            (lambda i, y: distr.cdf(y) - (i - 1)/len(sample)),
            range(len(sample)), sample
        )
    )


# ----------------------------------------------------------------------------------------------------------------------
# Anderson-Darling Test
# Essentially based on squared-error between edf and cdf
# Compares to tabulated values to reject hypothesis
# Estimation of parameters makes unable to reject, but should be good to distinguish between assumed set
# Basic statistic used here places more weight on the tails
# not general across distributions; can't be used to compare
def anderson_darling(sample, distr):

    for y in sample:
        if distr.cdf(y) <= 0 or distr.cdf(y) >= 1:
            print(distr.name(), y, distr.cdf(y), len(sample))

    # sort observations in sample
    sample.sort()

    # calculate test statistic
    return -len(sample) - reduce(
        add,
        map(
            (lambda i, y: (2*i + 1)*(log(distr.cdf(y)) + log(1 - distr.cdf(y)))),
            range(len(sample)), sample
        )
    ) / len(sample)


# ----------------------------------------------------------------------------------------------------------------------
# Cramér–von Mises Criterion
# Another version of Anderson Darling
# Actually able to distinguish most of the time; has a harder time with Laplace and Normal, but does get it
def von_mises(sample, distr):

    # sort observations in sample
    sample.sort()

    # calculate test statistic
    return (1/(12 * len(sample))) + reduce(
        add,
        map(
            (lambda i, y: ((2*i + 1)/(2*len(sample)) - distr.cdf(y))**2),
            range(len(sample)), sample
        )
    )
