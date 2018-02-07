# Mason Rumuly
# Challenge 1
#
# Naive bayes detector incorporating automatic feature selection (naively remove misleading features)

import Naive_Bayes as nb
import numpy as np


class SelectiveBayes(nb.NaiveBayes):

    # --------------------------------------------------------------------------------
    # Interface

    # test, leaving out any features selected against
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
                    if f not in self.features_removed:
                        p_temp *= nb.normal_dist(unidentified_data[r, f],
                                                 self.attribute_means[l, f],
                                                 self.attribute_vars[l, f])

                # check and update if necessary
                if p_temp > label_temp[1]:
                    label_temp = [self.attribute_labels[l], p_temp]

            # append to output
            if ids is None:
                ids = [label_temp]
            else:
                ids.append(label_temp)

        return ids

    # compilation with the addition of feature selection
    def compile(self):
        self.features_removed = []
        nb.NaiveBayes.compile(self)

        # get baseline coherence
        quality = self.coherence()

        # try removing each feature individually
        for f in range(np.shape(self.training_set)[1] - 1):
            if f not in self.features_removed:
                self.features_removed.append(f)
                # test with feature removed. If worse, keep feature. Otherwise, remove feature
                tq = self.coherence()
                if tq < quality:
                    del self.features_removed[-1]
                else:
                    quality = tq

    # returns the accuracy of the classifier (as a decimal) with regard to training set
    def coherence(self):
        r = self.test(self.training_set[:, :-1])
        success = 0
        for i in range(len(r)):
            if r[i][0] == self.training_set[i, -1]:
                success += 1
        return float(success)/np.shape(self.training_set)[0]

    # return the set of features removed
    def rejected_features(self):
        return self.features_removed

    # --------------------------------------------------------------------------------
    # Repository
    features_removed = []  # track features found to be detrimental or useless
