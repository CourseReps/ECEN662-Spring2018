# Mason Rumuly
# ECEN 662-600
# Final Project
#
# Distribution Interface
# All distributions to be used should extend this class


class Distribution(object):

    # returns name associated with the distribution
    @staticmethod
    def name():
        raise NotImplementedError()

    # estimate parameters based on list x of values
    # select argument should identify different estimation methods, if exist
    def estimate(self, sample, select=0):
        raise NotImplementedError()

    # return the list of parameters
    def get_params(self):
        raise NotImplementedError()

    # set parameters to pre-defined value
    def set_params(self, theta):
        raise NotImplementedError()

    # get pdf value of distribution at a given value
    def pdf(self, x):
        raise NotImplementedError()

    # get cdf value of distribution at a given value
    def cdf(self, x):
        raise NotImplementedError()

    # generate a random observation according to this distribution
    def get_observation(self):
        raise NotImplementedError()

    # generate a random sample according to this distribution
    def get_sample(self, sample_size):
        return [self.get_observation() for i in range(sample_size)]
