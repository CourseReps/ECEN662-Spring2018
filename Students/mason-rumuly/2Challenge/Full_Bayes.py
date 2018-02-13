# Mason Rumuly
# Challenge 2
#
# full bayes classifier assuming normal distribution
# allows sparse data for training and classification

import numpy as np


# calculate normal distribution
def normal_dist(sample, mean, variance):
    return np.exp(-pow(sample - mean, 2)/(2.0 * variance))/np.sqrt(2 * np.pi * variance)


# Hold the classes for FullBayes implementation
class Class:
    # ------------------------------------------------------------------------------------------------------------------
    #  interface

    # add to training set. Should be a single observation, i.e. a 1d numpy array, with no label
    def train(self, identified_data):
        # initialize observations list if necessary
        if self.observations is None:
            self.observations = []

        # note state of compilation
        self.compiled = False

        # add to list of observations
        self.observations.append(identified_data)

    # test class on observation
    def test(self, unidentified_data):
        # make sure ready
        if not self.compiled:
            self.compile()

        # build smaller covariance matrix to deal with sparse data (also eliminate NaN from data)
        indices_nan = np.where(np.isnan(unidentified_data))
        sub_covariance = np.delete(self.covariance_matrix, indices_nan, axis=0)
        sub_covariance = np.delete(sub_covariance, indices_nan, axis=1)
        sub_mean = np.delete(self.means, indices_nan, axis=0)
        sub_data = np.delete(unidentified_data, indices_nan, axis=0)

        # return likelihood times number
        centered_data = np.mat(np.subtract(sub_data, sub_mean))
        exponent = centered_data * np.linalg.inv(sub_covariance) * centered_data.transpose()
        result = (np.exp(- exponent) / np.sqrt(np.linalg.det(sub_covariance)) * len(self.observations))[0][0]
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # helper functions

    # create the mean vector and covariance matrix
    def compile(self):
        # check useful (no observations no point)
        if len(self.observations) < 1:
            return

        # create mean container
        self.means = np.zeros(len(self.observations[0]))
        quantity = np.zeros(len(self.observations[0]))
        # cumulative
        for o in self.observations:
            for f in range(o.shape[0]):
                if not np.isnan(o[f]):  # for sparse data
                    quantity[f] += 1
                    self.means[f] += o[f]
        # divide to get means
        self.means = np.divide(self.means, quantity)

        # build covariance matrix
        self.covariance_matrix = np.zeros((quantity.shape[0], quantity.shape[0]))
        # do iteratively to deal with sparse matrix
        for fx in range(quantity.shape[0]):
            for fy in range(fx, quantity.shape[0]):
                n = 0  # track number of valid data points
                cumulative = 0  # keep track of sum
                for o in self.observations:
                    if not (np.isnan(o[fx]) or np.isnan(o[fy])):
                        cumulative += (o[fx] - self.means[fx])*(o[fy] - self.means[fy])
                        n += 1
                cumulative /= n
                self.covariance_matrix[fx][fy] = cumulative
                self.covariance_matrix[fy][fx] = cumulative

        # note successful compilation
        self.compiled = True

    # ------------------------------------------------------------------------------------------------------------------
    # repository

    observations = None         # Observations with this class; list of numpy 1d arrays
    means = None                # Means of features
    covariance_matrix = None    # Covariance matrix for all features
    compiled = False            # track compilation state


# classifier using a full bayesian (rather than naive) method
class FullBayes:
    # ------------------------------------------------------------------------------------------------------------------
    # interface

    # identified_data should be 2d numpy array; last column to be labels, preceding to be features
    def train(self, identified_data):
        # sanity checks on input
        dim = len(np.shape(identified_data))

        # initialize variables
        if self.class_labels is None:
            self.class_labels = []
        if self.class_list is None:
            self.class_list = []

        if dim == 1:
            label_index = np.shape(identified_data)[0] - 1
            self.get_class(identified_data[label_index]).train(np.delete(identified_data, label_index))
            return

        if dim == 2:
            # iterate over rows, put in correct class
            label_index = np.shape(identified_data)[1] - 1
            for row in identified_data:
                # check label
                self.get_class(row[label_index]).train(np.delete(row, label_index))
            return

        raise ValueError('data input should have 1 or 2 dimensions; dimensions found: ' + str(dim))

    # unidentified_data should be 2d numpy array without labels (i.e. should have no label column)
    def test(self, unidentified_data):
        # sanity checks on input
        dim = len(np.shape(unidentified_data))

        if dim == 1:
            hat = 0
            max_l = 0
            for c in self.class_labels:
                l = self.get_class(c).test(unidentified_data)
                if l > max_l:
                    max_l = l
                    hat = c
            return hat

        if dim == 2:
            hats = []
            # iterate over rows, put in correct class
            for row in unidentified_data:
                hat = 0
                max_l = 0
                for c in self.class_labels:
                    l = self.get_class(c).test(row)
                    if l > max_l:
                        max_l = l
                        hat = c
                hats.append(hat)
            return hats

        raise ValueError('data input should have 1 or 2 dimensions; dimensions found: ' + str(dim))

    # ------------------------------------------------------------------------------------------------------------------
    # helper functions

    # remove previous class data
    def clear(self):
        self.class_labels = []
        self.class_list = []

    # get class by name, add if does not exist
    def get_class(self, name):
        if name in self.class_labels:
            return self.class_list[self.class_labels.index(name)]
        self.class_labels.append(name)
        self.class_list.append(Class())
        return self.class_list[len(self.class_list) - 1]

    # ------------------------------------------------------------------------------------------------------------------
    # repository

    class_labels = None   # store class names
    class_list = None     # store classes
