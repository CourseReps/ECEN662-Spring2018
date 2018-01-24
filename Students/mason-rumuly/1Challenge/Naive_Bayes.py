# Mason Rumuly
# Challenge 1
#
# Naive bayes detector assuming normal distribution

import numpy as np


# Store an attribute for NaiveBayes
def normal_dist(sample, mean, variance):
    return np.exp(-pow(sample - mean, 2)/(2.0 * variance))/np.sqrt(2 * np.pi * variance)


class NaiveBayes:
    # --------------------------------------------------------------------------------
    # interface

    # identified_data should be 2d numpy array; last column to be labels, preceding to be features
    def train(self, identified_data):
        # sanity checks on input
        dim = len(np.shape(identified_data))
        if dim != 2:
            raise ValueError('identified_data input should have 2 dimensions; dimensions found: ' + str(dim))

        # note state of compilation
        self.compiled = False

        # Add data to training set
        if self.training_set is None:
            self.training_set = identified_data
        else:
            if np.shape(identified_data)[1] != np.shape(self.training_set)[1]:
                raise ValueError('incorrect feature quantity found: ' + str(np.shape(identified_data)[1]))
            self.training_set = np.concatenate((self.training_set, identified_data))

    # unidentified_data should be 2d numpy array without labels (i.e. should have no label column)
    def test(self, unidentified_data):
        # sanity checks on input
        dim = len(np.shape(unidentified_data))
        if dim < 1 or dim > 2:
            raise ValueError('identified_data input should have 1 or 2 dimensions; dimensions found: ' + str(dim))
        if not np.shape(unidentified_data)[1] == np.shape(self.training_set)[1] - 1:
            raise ValueError('incorrect feature quantity found: ' + str(np.shape(unidentified_data)[1]))

        # make sure internal data is ready to be used
        if not self.compiled:
            self.compile()

        # classify each row, return (ordered) list
        ids = None
        for r in range(np.shape(unidentified_data)[0]):
            # first index is guessed label, second index 'probability' of label (probability comparative, not actual)
            label_temp = [0, 0]

            # iterate over labels to find most likely
            for l in range(len(self.attribute_labels)):

                # compute label probability, assuming features are independent and gaussian
                p_temp = self.attribute_occurrences[l]
                for f in range(len(self.attribute_means[l])):
                    p_temp *=\
                        normal_dist(unidentified_data[r, f], self.attribute_means[l, f], self.attribute_vars[l, f])

                # check and update if necessary
                if p_temp > label_temp[1]:
                    label_temp = [self.attribute_labels[l], p_temp]

            # append to output
            if ids is None:
                ids = [label_temp]
            else:
                ids.append(label_temp)

        return ids

    # remove previous training data
    def clear(self):
        self.compiled = False
        self.training_set = None

    # --------------------------------------------------------------------------------
    # helper functions

    # compile training set into attribute descriptions
    def compile(self):
        size = np.shape(self.training_set)

        self.attribute_labels = []
        self.attribute_occurrences = []
        self.attribute_means = np.empty((0, size[1] - 1), float)

        # enumerate labels and get means
        for r in range(size[0]):
            if self.training_set[r, -1] not in self.attribute_labels:
                self.attribute_labels.append(self.training_set[r, -1])
                self.attribute_occurrences.append(1)
                self.attribute_means = np.append(self.attribute_means,
                                                 np.reshape(self.training_set[r, 0:-1], (1, size[1] - 1)), 0)
            else:
                i = self.attribute_labels.index(self.training_set[r, -1])
                self.attribute_occurrences[i] += 1
                if len(np.shape(self.attribute_means)) == 1:
                    self.attribute_means = np.add(self.attribute_means, self.training_set[r, 0:-1])
                else:
                    self.attribute_means[i, :] = np.add(self.attribute_means[i, :], self.training_set[r, 0:-1])


        # finalize means
        for i in range(len(self.attribute_labels)):
            self.attribute_means[i, :] = np.divide(self.attribute_means[i, :], 1.0 * self.attribute_occurrences[i])

        # get variances
        self.attribute_vars = np.zeros(np.shape(self.attribute_means))
        for r in range(size[0]):
            i = self.attribute_labels.index(self.training_set[r, -1])
            self.attribute_vars[i, :] +=\
                np.square(np.subtract(self.attribute_means[i, :], self.training_set[r, 0:-1]))

        # finalize variances
        for i in range(len(self.attribute_labels)):
            self.attribute_vars[i, :] = np.divide(self.attribute_vars[i, :], 1.0 * self.attribute_occurrences[i])

        self.compiled = True

    # --------------------------------------------------------------------------------
    # repository

    training_set = None             # store training data
    compiled = False                # note whether attributes are ready for use for classification
    attribute_labels = None         # store attribute labels (index corresponds to row for mean and var)
    attribute_occurrences = None    # store number of times attribute occurred
    attribute_means = None          # store attribute feature means
    attribute_vars = None           # store attribute feature variances
