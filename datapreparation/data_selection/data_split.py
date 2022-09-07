# Copyright 2017 Pfäffli. All Rights Reserved.
#
# Created by Daniel Pfäffli on 19.04.2017
#
import numpy as np
import math
import sklearn.model_selection as skmodel_selection

def train_test_split(test_size:float, features:np.ndarray = None, targets:np.array = None, indices:list = None, take_from_end=True, random_state:int = None):
    """ Splits the given data into a training and testset. Splitting is done according to timeseries conditions.

    :param test_size: relative size of the testset
    :param features: ndarray which contains the features
    :param targets: array which contains the targetvalues
    :param indices: indices as separate list
    :param take_from_end: if true the set is taken from the end of the set. Otherwise from the begining
    :param random_state: optional, random state, if set the data are shuffled
    :return:
        feature-trainingset: ndarray with features for training,
        feature-testset: ndarray with features for testing,
        target-trainingset: array with targetvalues for training,
        target-testset: array with targetvalues for testing,
        indices-training: indices for training
        indices_test: indices for testing
    """
    assert type(test_size) is float, "Testsize must be float"
    assert indices is not None, "indices have to be set"

    feature_trainset = None
    feature_testset = None
    target_trainset = None
    target_testset = None
    indices_training = None
    indices_test = None
    test_index_start = 0
    train_index_start = 0
    test_index_end = len(indices)
    train_index_end = len(indices)
    if (random_state is not None):
        return skmodel_selection.train_test_split(
            features, targets, indices, test_size=test_size, random_state=random_state)

    if take_from_end:
        test_index_start = len(indices) - math.ceil(len(indices) * test_size)
        train_index_end = test_index_start
    else:
        test_index_end = math.ceil(len(indices) * test_size)
        train_index_start = test_index_end

    if features is not None:
        assert features is not None, "features have to be set"
        assert targets is not None, "targets have to be set"
        assert features.shape[0] == targets.shape[0], \
            "feature shape %s does not match target shape %s" % (str(features.shape), str(targets.shape))
        assert targets.shape[0] == len(indices), \
            "target shape %s does not match indices shape %i" % (str(targets.shape), len(indices))

        if features.ndim > 1:
            feature_trainset = features[train_index_start:train_index_end,:]
            feature_testset = features[test_index_start:test_index_end, :]
        else:
            feature_trainset = features[train_index_start:train_index_end]
            feature_testset = features[test_index_start:test_index_end]

        if targets.ndim>1:
            target_trainset = targets[train_index_start:train_index_end,:]
            target_testset = targets[test_index_start:test_index_end,:]
        else:
            target_trainset = targets[train_index_start:train_index_end]
            target_testset = targets[test_index_start:test_index_end]

    indices_training = indices[train_index_start:train_index_end]
    indices_test = indices[test_index_start:test_index_end]

    return feature_trainset, feature_testset, target_trainset, target_testset, indices_training, indices_test